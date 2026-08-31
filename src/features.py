"""Build the features, with strong protection against leakage.

`FeatureBuilder` uses the `fit` and `transform` pattern from scikit-learn.
`fit` learns the store statistics from the training data only. `transform`
builds the features that we know in advance and then joins the learned
statistics.

Because every learned value comes from `fit` (the training data) and is only
*used* in `transform`, no information from the holdout period can enter the
features.
"""

from __future__ import annotations

import polars as pl

# Fixed category levels, checked against the data. When we write them down
# here, the one-hot columns are the same for the training set and for the
# holdout set, and we never need an extra step to align the columns.
_STORE_TYPES = ["a", "b", "c", "d"]
_ASSORTMENTS = ["a", "b", "c"]
_STATE_HOLIDAYS = ["0", "a", "b", "c"]

# Short month names, exactly as they appear in the PromoInterval column of
# store.csv. Please note that September is written as "Sept", not "Sep".
_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}

# We treat an opening year below this value as unknown. One store says 1900,
# which is clearly a placeholder and would give an age of about 115 years.
_MIN_VALID_COMP_YEAR = 1950


class FeatureBuilder:
    """Build the feature matrix, and learn the store statistics from train only.

    Call `fit(train)` and then `transform(df)`, or call `fit_transform(train)`.
    After the fit, `feature_names_` holds the columns that form the model
    matrix `X`. The returned DataFrame also keeps the original columns (Store,
    Date, Sales and so on), so that we can measure the error and look at it
    from different angles.
    """

    def __init__(self) -> None:
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame) -> "FeatureBuilder":
        """Learn the store statistics and the fill values from `train` only."""
        self._global_mean_ = float(train["Sales"].mean())
        self._comp_dist_median_ = float(train["CompetitionDistance"].median())

        # Average sales per store.
        self._store_mean_ = train.group_by("Store").agg(
            pl.col("Sales").mean().alias("store_mean_sales")
        )
        # Average sales per store and weekday.
        self._store_dow_mean_ = train.group_by(["Store", "DayOfWeek"]).agg(
            pl.col("Sales").mean().alias("store_dow_mean_sales")
        )
        # How much a store gains from a promotion: the average sales on
        # promotion days divided by the average sales on normal days.
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

        # Build the training frame once, to fix the order of the columns.
        built = self._build(train)
        self.feature_names_ = self._collect_feature_names()
        # Safety check: every feature we list must really exist in the frame.
        missing = [c for c in self.feature_names_ if c not in built.columns]
        if missing:
            raise RuntimeError(f"Declared features missing after build: {missing}")
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build the features on `df`, with the statistics learned in `fit`."""
        if not self.feature_names_:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        return self._build(df)

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        """A short way to call `fit(train)` and then `transform(train)`."""
        return self.fit(train).transform(train)

    # -------------------------------------------------------------- internal
    def _collect_feature_names(self) -> list[str]:
        """The full list of columns that make up `X`, in a fixed order."""
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
        """Add every feature column to `df`: the simple ones and the joins."""
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
        """The age of the competitor in months, and a fill for the distance.

        When the opening date is missing, or when the year is below 1950, we
        do not trust it and we use an age of 0. We also cut the age at 0, so
        that a date before the opening cannot give a negative number.
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
        """Mark whether the repeating promotion Promo2 runs on that date."""
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
        """Turn the fixed category levels into one-hot columns."""
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
        """Join the learned store statistics and fill the gaps.

        A store that does not appear in the training data has no statistics of
        its own. In that case we fall back to the values of the whole training
        set, so that no row is left empty.
        """
        df = df.join(self._store_mean_, on="Store", how="left")
        df = df.join(self._store_dow_mean_, on=["Store", "DayOfWeek"], how="left")
        df = df.join(self._store_promo_uplift_, on="Store", how="left")

        df = df.with_columns(pl.col("store_mean_sales").fill_null(self._global_mean_))
        df = df.with_columns(
            pl.col("store_dow_mean_sales").fill_null(pl.col("store_mean_sales")),
            pl.col("store_promo_uplift").fill_null(self._global_uplift_),
        )
        return df
