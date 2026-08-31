"""Feature engineering with strict leakage protection.

:class:`FeatureBuilder` follows the scikit-learn ``fit`` / ``transform``
pattern. ``fit`` learns store statistics from the training data only;
``transform`` builds the known-in-advance features and joins the learned
statistics. Because every learned value comes from ``fit`` (train) and is
only *applied* in ``transform``, no information from the holdout period can
leak into the features.
"""

from __future__ import annotations

import polars as pl

# Fixed, validated category levels. Hard-coding them makes the one-hot columns
# identical for train and holdout without any column-alignment step.
_STORE_TYPES = ["a", "b", "c", "d"]
_ASSORTMENTS = ["a", "b", "c"]
_STATE_HOLIDAYS = ["0", "a", "b", "c"]

# Month abbreviations exactly as they appear in store.csv PromoInterval
# (note September is written "Sept", not "Sep").
_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Competition open years below this are treated as unknown (one store lists
# 1900, an obvious placeholder that would create a ~115-year age).
_MIN_VALID_COMP_YEAR = 1950


class FeatureBuilder:
    """Build the model feature matrix, learning store stats from train only.

    Use ``fit(train)`` then ``transform(df)``, or ``fit_transform(train)``.
    After fitting, ``feature_names_`` lists the columns that form the model
    matrix ``X``; the returned DataFrame also keeps the original columns
    (Store, Date, Sales, ...) for evaluation and error breakdowns.
    """

    def __init__(self) -> None:
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame) -> "FeatureBuilder":
        """Learn store statistics and imputation values from ``train`` only."""
        self._global_mean_ = float(train["Sales"].mean())
        self._comp_dist_median_ = float(train["CompetitionDistance"].median())

        # Store mean sales.
        self._store_mean_ = train.group_by("Store").agg(
            pl.col("Sales").mean().alias("store_mean_sales")
        )
        # Store x day-of-week mean sales.
        self._store_dow_mean_ = train.group_by(["Store", "DayOfWeek"]).agg(
            pl.col("Sales").mean().alias("store_dow_mean_sales")
        )
        # Store promo uplift = mean sales on promo days / mean on non-promo days.
        no_promo = train.filter(pl.col("Promo") == 0).group_by("Store").agg(
            pl.col("Sales").mean().alias("_np_mean")
        )
        promo = train.filter(pl.col("Promo") == 1).group_by("Store").agg(
            pl.col("Sales").mean().alias("_p_mean")
        )
        self._store_promo_uplift_ = (
            no_promo.join(promo, on="Store", how="full", coalesce=True)
            .with_columns((pl.col("_p_mean") / pl.col("_np_mean")).alias("store_promo_uplift"))
            .select(["Store", "store_promo_uplift"])
        )
        self._global_uplift_ = float(
            train.filter(pl.col("Promo") == 1)["Sales"].mean()
            / train.filter(pl.col("Promo") == 0)["Sales"].mean()
        )

        # Establish the feature column order from the built training frame.
        built = self._build(train)
        self.feature_names_ = self._collect_feature_names()
        # Sanity: every declared feature must exist in the built frame.
        missing = [c for c in self.feature_names_ if c not in built.columns]
        if missing:
            raise RuntimeError(f"Declared features missing after build: {missing}")
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build features on ``df`` using the statistics learned in ``fit``."""
        if not self.feature_names_:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        return self._build(df)

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        """Convenience: ``fit(train)`` then ``transform(train)``."""
        return self.fit(train).transform(train)

    # -------------------------------------------------------------- internal
    def _collect_feature_names(self) -> list[str]:
        """The explicit, ordered list of columns that make up ``X``."""
        calendar = ["year", "month", "day", "weekofyear", "DayOfWeek"]
        known = ["Promo", "SchoolHoliday", "promo2_active",
                 "CompetitionDistance", "competition_age_months"]
        dummies = (
            [f"StoreType_{t}" for t in _STORE_TYPES]
            + [f"Assortment_{a}" for a in _ASSORTMENTS]
            + [f"StateHoliday_{h}" for h in _STATE_HOLIDAYS]
        )
        learned = ["store_mean_sales", "store_dow_mean_sales", "store_promo_uplift"]
        return calendar + known + dummies + learned

    def _build(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add all feature columns to ``df`` (deterministic + learned joins)."""
        df = self._add_calendar(df)
        df = self._add_competition(df)
        df = self._add_promo2_active(df)
        df = self._add_dummies(df)
        df = self._join_learned(df)
        return df

    def _add_calendar(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("Date").dt.year().alias("year"),
            pl.col("Date").dt.month().alias("month"),
            pl.col("Date").dt.day().alias("day"),
            pl.col("Date").dt.week().alias("weekofyear"),
        )

    def _add_competition(self, df: pl.DataFrame) -> pl.DataFrame:
        """Competition age in months, plus distance imputation.

        Unknown or placeholder open dates (year missing or < 1950) give age 0.
        Ages are clipped at 0 so a date before the competition opened cannot
        become negative.
        """
        year = pl.col("CompetitionOpenSinceYear")
        month = pl.col("CompetitionOpenSinceMonth")
        valid = year.is_not_null() & month.is_not_null() & (year >= _MIN_VALID_COMP_YEAR)
        age = (pl.col("Date").dt.year() - year) * 12 + (pl.col("Date").dt.month() - month)
        return df.with_columns(
            pl.col("CompetitionDistance").fill_null(self._comp_dist_median_),
            pl.when(valid).then(age).otherwise(0).clip(lower_bound=0)
            .alias("competition_age_months"),
        )

    def _add_promo2_active(self, df: pl.DataFrame) -> pl.DataFrame:
        """Flag whether a store's recurring Promo2 is active on that date."""
        month_abbr = pl.col("Date").dt.month().replace_strict(
            _MONTH_ABBR, return_dtype=pl.Utf8
        )
        started = (pl.col("Date").dt.year() > pl.col("Promo2SinceYear")) | (
            (pl.col("Date").dt.year() == pl.col("Promo2SinceYear"))
            & (pl.col("Date").dt.week() >= pl.col("Promo2SinceWeek"))
        )
        in_interval = (
            pl.col("PromoInterval").fill_null("").str.split(",").list.contains(month_abbr)
        )
        active = (pl.col("Promo2") == 1) & started & in_interval
        return df.with_columns(active.fill_null(False).cast(pl.Int8).alias("promo2_active"))

    def _add_dummies(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode the fixed categorical levels (deterministic columns)."""
        exprs = (
            [(pl.col("StoreType") == t).cast(pl.Int8).alias(f"StoreType_{t}")
             for t in _STORE_TYPES]
            + [(pl.col("Assortment") == a).cast(pl.Int8).alias(f"Assortment_{a}")
               for a in _ASSORTMENTS]
            + [(pl.col("StateHoliday") == h).cast(pl.Int8).alias(f"StateHoliday_{h}")
               for h in _STATE_HOLIDAYS]
        )
        return df.with_columns(exprs)

    def _join_learned(self, df: pl.DataFrame) -> pl.DataFrame:
        """Join the fitted store statistics and fill gaps with train globals."""
        df = df.join(self._store_mean_, on="Store", how="left")
        df = df.join(self._store_dow_mean_, on=["Store", "DayOfWeek"], how="left")
        df = df.join(self._store_promo_uplift_, on="Store", how="left")

        df = df.with_columns(pl.col("store_mean_sales").fill_null(self._global_mean_))
        df = df.with_columns(
            pl.col("store_dow_mean_sales").fill_null(pl.col("store_mean_sales")),
            pl.col("store_promo_uplift").fill_null(self._global_uplift_),
        )
        return df
