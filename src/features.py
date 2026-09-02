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

from datetime import date

import numpy as np
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

# The same map the other way round. We use it to read the first month of a
# PromoInterval, which tells us where the promotion cycle of a store starts.
_MONTH_NUM = {name: number for number, name in _MONTH_ABBR.items()}

# Promo2 repeats every three months, so a month sits at position 0, 1 or 2 of
# the cycle. Stores outside Promo2 get this value instead, so that the model
# can keep them apart from the stores that take part.
_NO_PROMO2_CYCLE = -1

# We treat an opening year below this value as unknown. One store says 1900,
# which is clearly a placeholder and would give an age of about 115 years.
_MIN_VALID_COMP_YEAR = 1950

# How far we look for the closest state holiday. Notebook 05 showed the effect
# is short: one day before a holiday the error is 0.208, two days before it is
# already 0.129. Anything further away is the same thing to the model, so we
# stop counting and use this value.
_HOLIDAY_HORIZON = 10

# The window we use to count the closed days before and after a day. One week
# holds the normal Sunday, so a value above the usual one means an extra
# closure is near.
_CLOSURE_WINDOW = 7

# Used when a day sits at the edge of the calendar and the answer is unknown.
_UNKNOWN = -1

# How far from Easter we still count the days. Outside this window every day
# gets the same value, because the feasts we care about all sit inside it:
# Rosenmontag is 48 days before Easter and Whit Monday is 50 days after.
_EASTER_WINDOW = 60

# The column that `with_easter=True` adds.
EASTER_COLUMNS = ["days_to_easter"]

# The columns that `build_closure_calendar` produces. They are features only
# when a calendar is given to `FeatureBuilder`.
CLOSURE_COLUMNS = [
    "open_yesterday",
    "open_tomorrow",
    "closed_days_next_7",
    "closed_days_last_7",
    "days_to_next_holiday",
    "days_since_last_holiday",
]

# A closed stretch of at least this many days counts as a long closure, for
# example a renovation. The longest normal closures in this data are holiday
# blocks of three or four days, so three weeks is far above them.
_LONG_CLOSURE_DAYS = 21

# After this many days a reopening is old news. Every day further away gets
# this value, and so does a store that never had a long closure.
_REOPENING_CAP = 365

# A store needs at least this many trading days after its reopening before we
# trust statistics that are computed from those days alone.
_MIN_DAYS_AFTER_REOPENING = 60

# The column that `with_reopening=True` adds.
REOPENING_COLUMNS = ["days_since_reopening"]

# The first days of the month, for the `month_start` flag. Notebook 05
# measured the error of days 1 to 5 at 0.178 against 0.116 for days 6 to 10.
_MONTH_START_DAYS = 5

# The columns that `with_month=True` adds.
MONTH_COLUMNS = ["month_start", "days_to_month_end"]


def easter_sunday(year: int) -> date:
    """Return the date of Easter Sunday, with the usual Gregorian rule.

    Easter moves. It fell on the 31st of March in 2013, the 20th of April in
    2014 and the 5th of April in 2015, so a swing of three weeks. Several
    German feasts hang on it: Rosenmontag is 48 days before, Good Friday two
    days before, Whit Monday 50 days after.

    Our model reads a date as a month, a day and a week number, so it cannot
    line those feasts up between the years. Notebook 05 measured what that
    costs: the carnival week scores 0.335 and the week before Easter 0.213,
    against 0.09 to 0.15 for a normal week.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


def build_closure_calendar(calendar: pl.DataFrame) -> pl.DataFrame:
    """Describe, for every store and day, the closed days that sit around it.

    This is the feature family that notebook 05 asked for. Our model data holds
    open days only, so the model can see that today is a holiday but not that
    tomorrow is one. The days that carry the demand of a closed day are exactly
    the days we predict worst: a day after a closed day scores 0.187 and a day
    before one 0.171, against 0.123 for a normal day.

    Nothing here is learned from the sales. A shop plans its opening days and
    the state publishes its holidays long in advance, so every column below is
    known six weeks ahead, which is what our task allows.

    Parameters
    ----------
    calendar:
        The daily table **with the closed days kept**, so
        `DataLoader.load(drop_closed=False)`. It needs `Store`, `Date`, `Open`
        and `StateHoliday`.

    Returns
    -------
    pl.DataFrame
        One row per store and day, with `Store`, `Date` and the columns in
        `CLOSURE_COLUMNS`.

    Notes
    -----
    A day that has no row at all counts as closed. That is what a missing row
    means here: the shop was not trading, which is the case for the 180 stores
    that were away for a renovation in 2014.

    The last day of the calendar has no tomorrow, so `open_tomorrow` and
    `closed_days_next_7` are `-1` there. In the real task the same happens on
    the last day of the forecast period, which is one day out of 42.
    """
    needed = {"Store", "Date", "Open", "StateHoliday"}
    missing = needed - set(calendar.columns)
    if missing:
        raise ValueError(f"The calendar has no column(s): {sorted(missing)}")

    first_day = calendar["Date"].min()
    last_day = calendar["Date"].max()
    n_days = (last_day - first_day).days + 1

    rows = []
    for key, part in calendar.sort("Date").group_by("Store", maintain_order=True):
        store = key[0] if isinstance(key, tuple) else key
        index = np.array([(d - first_day).days for d in part["Date"].to_list()])

        # A full day-by-day axis for this store. Everything we did not see is
        # a day the shop was not trading.
        is_open = np.zeros(n_days, dtype=bool)
        is_open[index] = part["Open"].to_numpy() == 1
        is_holiday = np.zeros(n_days, dtype=bool)
        is_holiday[index] = part["StateHoliday"].to_numpy() != "0"

        rows.append(
            pl.DataFrame({
                "Store": np.full(len(index), store, dtype=np.int64),
                "Date": part["Date"],
                **{name: values[index] for name, values in
                   _closure_columns(is_open, is_holiday, n_days).items()},
            })
        )

    return pl.concat(rows).sort(["Store", "Date"])


def _closure_columns(is_open: np.ndarray, is_holiday: np.ndarray,
                     n_days: int) -> dict[str, np.ndarray]:
    """Build the six columns on one store's day-by-day axis."""
    open_yesterday = np.full(n_days, _UNKNOWN, dtype=np.int64)
    open_yesterday[1:] = is_open[:-1]
    open_tomorrow = np.full(n_days, _UNKNOWN, dtype=np.int64)
    open_tomorrow[:-1] = is_open[1:]

    # How many of the next seven days is the shop closed, and how many of the
    # last seven? A running sum makes this one subtraction per day.
    closed = (~is_open).astype(np.int64)
    total = np.concatenate([[0], np.cumsum(closed)])

    closed_next = np.full(n_days, _UNKNOWN, dtype=np.int64)
    closed_last = np.full(n_days, _UNKNOWN, dtype=np.int64)
    for i in range(n_days):
        if i + _CLOSURE_WINDOW < n_days:
            closed_next[i] = total[i + 1 + _CLOSURE_WINDOW] - total[i + 1]
        if i - _CLOSURE_WINDOW >= 0:
            closed_last[i] = total[i] - total[i - _CLOSURE_WINDOW]

    holidays = np.flatnonzero(is_holiday)
    to_next = np.full(n_days, _HOLIDAY_HORIZON, dtype=np.int64)
    since_last = np.full(n_days, _HOLIDAY_HORIZON, dtype=np.int64)
    if holidays.size:
        days = np.arange(n_days)
        after = np.searchsorted(holidays, days, side="left")
        has_after = after < holidays.size
        to_next[has_after] = holidays[after[has_after]] - days[has_after]

        before = np.searchsorted(holidays, days, side="right") - 1
        has_before = before >= 0
        since_last[has_before] = days[has_before] - holidays[before[has_before]]

    return {
        "open_yesterday": open_yesterday,
        "open_tomorrow": open_tomorrow,
        "closed_days_next_7": closed_next,
        "closed_days_last_7": closed_last,
        "days_to_next_holiday": np.minimum(to_next, _HOLIDAY_HORIZON),
        "days_since_last_holiday": np.minimum(since_last, _HOLIDAY_HORIZON),
    }


def build_reopening_calendar(calendar: pl.DataFrame) -> pl.DataFrame:
    """Count, for every store and day, the days since its last long closure.

    About 180 stores were closed for a renovation for around six months in
    2014. Notebook 05 measured what their return costs: 30 to 59 days after
    the first day back the error is 0.265, and one month later it is 0.127.
    The problem is narrow and it ends by itself, so the feature is the *time*
    since the reopening, not the fact of the renovation.

    A long closure is a closed stretch of at least `_LONG_CLOSURE_DAYS` days.
    A missing row counts as closed, exactly as in `build_closure_calendar`.
    The count starts at 0 on the first open day back and is cut at
    `_REOPENING_CAP`. A day before any long closure, and a store that never
    had one, get the cap: for the model they are all simply "not recent".

    Like the closed-day family this is reference data, not something we
    learn. Which days a store traded is a fact of the past at prediction
    time.
    """
    needed = {"Store", "Date", "Open"}
    missing = needed - set(calendar.columns)
    if missing:
        raise ValueError(f"The calendar has no column(s): {sorted(missing)}")

    first_day = calendar["Date"].min()
    last_day = calendar["Date"].max()
    n_days = (last_day - first_day).days + 1

    rows = []
    for key, part in calendar.sort("Date").group_by("Store", maintain_order=True):
        store = key[0] if isinstance(key, tuple) else key
        index = np.array([(d - first_day).days for d in part["Date"].to_list()])

        is_open = np.zeros(n_days, dtype=bool)
        is_open[index] = part["Open"].to_numpy() == 1

        # Find the closed stretches. We pad with an open day on both sides,
        # so that a stretch at the very start or end is still a stretch.
        closed = (~is_open).astype(np.int8)
        change = np.diff(np.concatenate([[0], closed, [0]]))
        starts = np.flatnonzero(change == 1)
        ends = np.flatnonzero(change == -1) - 1

        since = np.full(n_days, _REOPENING_CAP, dtype=np.int64)
        for start, end in zip(starts, ends):
            if end - start + 1 < _LONG_CLOSURE_DAYS or end + 1 >= n_days:
                continue
            back = end + 1
            # A later reopening overwrites an earlier one, so the count
            # always refers to the newest long closure before the day.
            since[back:] = np.arange(n_days - back)

        rows.append(pl.DataFrame({
            "Store": np.full(len(index), store, dtype=np.int64),
            "Date": part["Date"],
            "days_since_reopening": np.minimum(since, _REOPENING_CAP)[index],
        }))

    return pl.concat(rows).sort(["Store", "Date"])


class FeatureBuilder:
    """Build the feature matrix, and learn the store statistics from train only.

    Call `fit(train)` and then `transform(df)`, or call `fit_transform(train)`.
    After the fit, `feature_names_` holds the columns that form the model
    matrix `X`. The returned DataFrame also keeps the original columns (Store,
    Date, Sales and so on), so that we can measure the error and look at it
    from different angles.
    """

    def __init__(self, calendar: pl.DataFrame | None = None,
                 with_easter: bool = False,
                 with_reopening: bool = False,
                 with_reopening_stats: bool = False,
                 with_month: bool = False) -> None:
        """Set the builder up, with or without the closed-day family.

        Parameters
        ----------
        calendar:
            The daily table with the closed days kept, from
            `DataLoader.load(drop_closed=False)`. When it is given, the six
            columns of `CLOSURE_COLUMNS` join the feature list. When it is
            left out, the builder works exactly as it did before, so the
            earlier notebooks still produce the numbers they report.

            The calendar is reference data and not something we learn. It says
            which days a shop plans to open and when the state holidays are,
            and both are known long before the day itself.

            You may also pass a table that `build_closure_calendar` has
            already produced. The builder then uses it as it is. This saves
            time when many builders are fitted in a row, for example one per
            fold, because the calendar is the same every time.
        with_easter:
            Add `days_to_easter`, the signed number of days between the row
            and Easter Sunday of that year. Off by default, for the same
            reason as the calendar: the earlier notebooks must keep producing
            the numbers they report.
        with_reopening:
            Add the `days_since_reopening` column. Needs the calendar,
            because the reopenings are read from the closed days.
        with_reopening_stats:
            On top of the column, compute the learned store statistics from
            the days after a reopening only, when a store has enough of
            them. Notebook 07 tests the two parts apart.
        with_month:
            Add `month_start` (the first five days of the month) and
            `days_to_month_end`. Both come from the date alone.
        """
        self.feature_names_: list[str] = []
        self.with_easter = with_easter
        self.with_reopening = with_reopening
        self.with_reopening_stats = with_reopening_stats
        self.with_month = with_month
        if with_reopening_stats and not with_reopening:
            raise ValueError("with_reopening_stats needs with_reopening=True.")
        if calendar is None:
            self._closure_ = None
        elif set(CLOSURE_COLUMNS).issubset(calendar.columns):
            self._closure_ = calendar
        else:
            closure = build_closure_calendar(calendar)
            if with_reopening:
                closure = closure.join(build_reopening_calendar(calendar),
                                       on=["Store", "Date"], how="left")
            self._closure_ = closure
        if with_reopening and (
                self._closure_ is None
                or "days_since_reopening" not in self._closure_.columns):
            raise ValueError(
                "with_reopening needs the calendar. Please pass "
                "DataLoader.load(drop_closed=False), or a prebuilt table "
                "that already holds the days_since_reopening column."
            )

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame) -> "FeatureBuilder":
        """Learn the store statistics and the fill values from `train` only."""
        self._global_mean_ = float(train["Sales"].mean())
        self._comp_dist_median_ = float(train["CompetitionDistance"].median())

        # A store that has just come back from a renovation is a different
        # shop than its own history: averages over the whole training period
        # describe a store that no longer exists. So, when asked, we compute
        # the store statistics from the days after the reopening only - but
        # only for stores with enough of those days, because a mean over a
        # few weeks is noise.
        stats_train = train
        if self.with_reopening_stats:
            stats_train = self._rows_for_statistics(train)

        # Average sales per store.
        self._store_mean_ = stats_train.group_by("Store").agg(
            pl.col("Sales").mean().alias("store_mean_sales")
        )
        # Average sales per store and weekday.
        self._store_dow_mean_ = stats_train.group_by(["Store", "DayOfWeek"]).agg(
            pl.col("Sales").mean().alias("store_dow_mean_sales")
        )
        # How much a store gains from a promotion: the average sales on
        # promotion days divided by the average sales on normal days.
        no_promo = stats_train.filter(pl.col("Promo") == 0).group_by("Store").agg(
            pl.col("Sales").mean().alias("_np_mean")
        )
        promo = stats_train.filter(pl.col("Promo") == 1).group_by("Store").agg(
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
        # `promo2_tenure_weeks` and `promo_cycle_pos` are built in _add_promo2
        # but they are not in this list. In the raw data the sales of a store
        # rise by about 12 percent from the first year of Promo2 to the sixth,
        # so we expected them to help. They do not: on the four folds the
        # difference is 0.0003, which is smaller than the noise, and their
        # importance in the model is close to zero. The gradient belongs to
        # the stores that joined early, and the model already knows those
        # stores through `store_mean_sales`. Notebook 03 shows the test.
        dummies = (
            [f"StoreType_{t}" for t in _STORE_TYPES]
            + [f"Assortment_{a}" for a in _ASSORTMENTS]
            + [f"StateHoliday_{h}" for h in _STATE_HOLIDAYS]
        )
        learned = ["store_mean_sales", "store_dow_mean_sales", "store_promo_uplift"]
        closure = list(CLOSURE_COLUMNS) if self._closure_ is not None else []
        easter = list(EASTER_COLUMNS) if self.with_easter else []
        reopening = list(REOPENING_COLUMNS) if self.with_reopening else []
        month = list(MONTH_COLUMNS) if self.with_month else []
        return calendar + known + dummies + learned + closure + easter + reopening + month

    def _build(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add every feature column to `df`: the simple ones and the joins."""
        df = self._add_calendar(df)
        df = self._add_competition(df)
        df = self._add_promo2(df)
        df = self._add_dummies(df)
        df = self._join_learned(df)
        df = self._join_closure(df)
        df = self._add_easter(df)
        df = self._add_month(df)
        return df

    def _add_month(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the start-of-month flag and the distance to the month end.

        The model already has the day of the month as a raw number, so these
        two columns are a sharper way to say the same thing: `month_start`
        marks the payday shopping of the first days, and `days_to_month_end`
        counts down to the next month. Both come from the date alone.
        """
        if not self.with_month:
            return df
        year = pl.col("Date").dt.year()
        month = pl.col("Date").dt.month()
        next_month = pl.date(
            pl.when(month == 12).then(year + 1).otherwise(year),
            pl.when(month == 12).then(1).otherwise(month + 1),
            1,
        )
        return df.with_columns(
            (pl.col("Date").dt.day() <= _MONTH_START_DAYS)
            .cast(pl.Int8).alias("month_start"),
            ((next_month - pl.col("Date")).dt.total_days() - 1)
            .cast(pl.Int32).alias("days_to_month_end"),
        )

    def _rows_for_statistics(self, train: pl.DataFrame) -> pl.DataFrame:
        """Drop the pre-renovation history of the stores that came back.

        For every store with a long closure inside the training period, we
        keep only the rows after its newest reopening - if there are at least
        `_MIN_DAYS_AFTER_REOPENING` of them. Every other store keeps all of
        its rows. The result feeds the learned store statistics only; the
        model still trains on every row.
        """
        train_end = train["Date"].max()
        came_back = (
            self._closure_
            .filter((pl.col("days_since_reopening") == 0)
                    & (pl.col("Date") <= train_end))
            .group_by("Store").agg(pl.col("Date").max().alias("_back"))
        )
        joined = train.join(came_back, on="Store", how="left")
        recent = pl.col("Date") >= pl.col("_back")
        enough = (
            joined.filter(recent).group_by("Store").len()
            .filter(pl.col("len") >= _MIN_DAYS_AFTER_REOPENING)["Store"]
        )
        keep = pl.col("_back").is_null() | ~pl.col("Store").is_in(enough) | recent
        return joined.filter(keep).drop("_back")

    def _add_easter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the signed distance in days between the row and Easter Sunday.

        A negative value means the day comes before Easter. Days far away from
        Easter are all cut to the same value, so that the model does not read
        the distance as a second calendar.
        """
        if not self.with_easter:
            return df
        years = df["Date"].dt.year().unique().to_list()
        easter = {year: easter_sunday(year) for year in years}
        distance = [(d - easter[d.year]).days for d in df["Date"].to_list()]
        return df.with_columns(
            pl.Series("days_to_easter", distance, dtype=pl.Int32)
            .clip(-_EASTER_WINDOW, _EASTER_WINDOW)
        )

    def _join_closure(self, df: pl.DataFrame) -> pl.DataFrame:
        """Join the closed-day family, when a calendar was given."""
        if self._closure_ is None:
            return df
        joined = df.join(self._closure_, on=["Store", "Date"], how="left")
        empty = joined.select(pl.col(CLOSURE_COLUMNS[0]).is_null().sum()).item()
        if empty:
            raise ValueError(
                f"{empty} rows are not in the calendar. Please pass a calendar "
                "that covers every day you want to predict."
            )
        return joined

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

    def _add_promo2(self, df: pl.DataFrame) -> pl.DataFrame:
        """Describe the repeating promotion Promo2 with three columns.

        - `promo2_active` says whether the promotion runs on that date.
        - `promo2_tenure_weeks` counts the weeks since the store joined. Stores
          sell more the longer they take part, so the length matters and not
          only the fact that they joined.
        - `promo_cycle_pos` says where the month sits in the three month cycle
          of the store: 0 is the month the cycle starts, then 1 and then 2.
        """
        year = pl.col("Date").dt.year()
        week = pl.col("Date").dt.week()
        month = pl.col("Date").dt.month()
        month_abbr = month.replace_strict(_MONTH_ABBR, return_dtype=pl.Utf8)

        started = (year > pl.col("Promo2SinceYear")) | (
            (year == pl.col("Promo2SinceYear"))
            & (week >= pl.col("Promo2SinceWeek"))
        )
        # True only for a store that takes part and has already started.
        running = ((pl.col("Promo2") == 1) & started).fill_null(False)

        in_interval = (
            pl.col("PromoInterval").fill_null("").str.split(",").list.contains(month_abbr)
        ).fill_null(False)

        tenure_weeks = (year - pl.col("Promo2SinceYear")) * 52 + (
            week - pl.col("Promo2SinceWeek")
        )

        # The first month named in PromoInterval starts the cycle. We add 12
        # before the modulo so that the result is never negative.
        cycle_start = (
            pl.col("PromoInterval")
            .str.split(",")
            .list.first()
            .replace_strict(_MONTH_NUM, return_dtype=pl.Int32, default=None)
        )
        cycle_pos = (month - cycle_start + 12) % 3

        return df.with_columns(
            (running & in_interval).cast(pl.Int8).alias("promo2_active"),
            pl.when(running)
            .then(tenure_weeks)
            .otherwise(0)
            .clip(lower_bound=0)
            .cast(pl.Int32)
            .alias("promo2_tenure_weeks"),
            pl.when(running)
            .then(cycle_pos)
            .otherwise(_NO_PROMO2_CYCLE)
            .fill_null(_NO_PROMO2_CYCLE)
            .cast(pl.Int8)
            .alias("promo_cycle_pos"),
        )

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
