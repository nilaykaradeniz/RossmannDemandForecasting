from datetime import date

import polars as pl
import pytest

from src.features import FeatureBuilder, CLOSURE_COLUMNS, build_closure_calendar


def split(train):
    cut = date(2015, 4, 1)
    return train.filter(pl.col("Date") < cut), train.filter(pl.col("Date") >= cut)


def test_transform_gives_every_feature_column(train):
    fit_rows, new_rows = split(train)
    builder = FeatureBuilder().fit(fit_rows)
    built = builder.transform(new_rows)
    assert builder.feature_names_
    assert all(name in built.columns for name in builder.feature_names_)
    assert built.select(builder.feature_names_).null_count().sum_horizontal().item() == 0


def test_learned_statistics_come_from_the_training_rows_only(train):
    # Leakage test: the sales of the rows we transform must not change the
    # statistics that are joined onto them.
    fit_rows, new_rows = split(train)
    builder = FeatureBuilder().fit(fit_rows)
    learned = ["store_mean_sales", "store_dow_mean_sales", "store_promo_uplift"]

    normal = builder.transform(new_rows).select(learned)
    doubled = builder.transform(new_rows.with_columns(pl.col("Sales") * 2)).select(learned)
    assert normal.equals(doubled)


def test_unknown_store_gets_the_global_fallback(train):
    fit_rows, new_rows = split(train)
    builder = FeatureBuilder().fit(fit_rows)
    stranger = new_rows.head(5).with_columns(pl.lit(9999).alias("Store"))
    built = builder.transform(stranger)
    assert built["store_mean_sales"].null_count() == 0
    assert (built["store_mean_sales"] == builder._global_mean_).all()


def test_closure_columns_see_the_closed_day_ahead(calendar):
    closure = build_closure_calendar(calendar)
    assert set(CLOSURE_COLUMNS).issubset(closure.columns)
    # Saturday 2014-06-07 is followed by a closed Sunday.
    saturday = closure.filter((pl.col("Store") == 1) & (pl.col("Date") == date(2014, 6, 7)))
    assert saturday["open_tomorrow"][0] == 0
    # Monday 2014-06-09 follows that closed Sunday.
    monday = closure.filter((pl.col("Store") == 1) & (pl.col("Date") == date(2014, 6, 9)))
    assert monday["open_yesterday"][0] == 0
    assert monday["closed_days_last_7"][0] >= 1


def test_closure_columns_need_the_calendar_to_cover_the_rows(train, calendar):
    fit_rows, new_rows = split(train)
    short_calendar = calendar.filter(pl.col("Date") < date(2015, 4, 1))
    builder = FeatureBuilder(calendar=short_calendar, with_easter=True).fit(fit_rows)
    with pytest.raises(ValueError):
        builder.transform(new_rows)
    # After the calendar is extended, the same rows go through.
    builder.set_calendar(calendar)
    built = builder.transform(new_rows)
    assert built["open_tomorrow"].null_count() == 0


def test_easter_distance_is_zero_on_easter_sunday(train):
    builder = FeatureBuilder(with_easter=True).fit(train)
    row = train.head(1).with_columns(pl.lit(date(2015, 4, 5)).alias("Date"))
    assert builder.transform(row)["days_to_easter"][0] == 0
