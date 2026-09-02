from datetime import timedelta

import polars as pl

from src.validation import RollingOriginSplit


def test_training_days_always_come_before_the_validation_window(train):
    for fold in RollingOriginSplit(n_splits=3).split(train):
        assert fold.train["Date"].max() < fold.valid["Date"].min()


def test_every_validation_window_is_six_weeks(train):
    for fold in RollingOriginSplit(n_splits=3, horizon_days=42).split(train):
        assert (fold.valid_end - fold.valid_start).days + 1 == 42
        assert fold.valid["Date"].min() >= fold.valid_start
        assert fold.valid["Date"].max() <= fold.valid_end


def test_validation_windows_do_not_overlap(train):
    folds = list(RollingOriginSplit(n_splits=3).split(train))
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.valid_end < later.valid_start


def test_training_set_holds_every_earlier_day(train):
    # A fold trains on all history before its window, not on a sample of it.
    for fold in RollingOriginSplit(n_splits=2).split(train):
        expected = train.filter(pl.col("Date") < fold.valid_start).height
        assert fold.train.height == expected


def test_last_window_ends_on_the_last_day(train):
    folds = list(RollingOriginSplit(n_splits=2).split(train))
    assert folds[-1].valid_end == train["Date"].max()


def test_too_short_history_is_an_error(train):
    short = train.filter(pl.col("Date") > train["Date"].max() - timedelta(days=200))
    try:
        list(RollingOriginSplit(n_splits=4, min_train_days=365).split(short))
    except ValueError:
        return
    raise AssertionError("A fold with too little history should be refused.")
