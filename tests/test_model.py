from datetime import date

import numpy as np
import polars as pl
import pytest

from src.features import FeatureBuilder
from src.model import SalesModel, DEFAULT_PARAMS

FAST = dict(DEFAULT_PARAMS, eta=0.3)


@pytest.fixture(scope="module")
def built(train):
    cut = date(2015, 4, 1)
    fit_rows = train.filter(pl.col("Date") < cut)
    builder = FeatureBuilder().fit(fit_rows)
    return (builder.transform(fit_rows), builder.transform(train.filter(pl.col("Date") >= cut)),
            builder.feature_names_)


def test_predictions_are_never_negative(built):
    tr, va, names = built
    model = SalesModel(params=FAST, early_stopping_rounds=10).fit(tr, names)
    assert (model.predict(va) >= 0).all()
    assert model.best_rounds_ >= 1


def test_the_model_beats_a_flat_guess(built):
    tr, va, names = built
    model = SalesModel(params=FAST, early_stopping_rounds=10).fit(tr, names)
    pred = model.predict(va)
    error = np.sqrt(np.mean(((va["Sales"].to_numpy() - pred) / va["Sales"].to_numpy()) ** 2))
    flat = np.sqrt(np.mean(((va["Sales"].to_numpy() - tr["Sales"].mean()) / va["Sales"].to_numpy()) ** 2))
    assert error < flat / 2


def test_fixed_rounds_skips_the_inner_window(built):
    tr, _, names = built
    model = SalesModel(params=FAST, fixed_rounds=7).fit(tr, names)
    assert model.best_rounds_ == 7
    assert model.inner_scored_ is None


def test_no_refit_keeps_the_inner_window_scored(built):
    tr, _, names = built
    model = SalesModel(params=FAST, early_stopping_rounds=10, refit=False).fit(tr, names)
    inner = model.inner_scored_
    assert inner is not None and "prediction" in inner.columns
    # The inner window is the last six weeks of the training rows.
    assert inner["Date"].max() == tr["Date"].max()
    assert (tr["Date"].max() - inner["Date"].min()).days < 42


def test_missing_feature_column_is_an_error(built):
    tr, _, names = built
    with pytest.raises(ValueError):
        SalesModel(params=FAST).fit(tr, names + ["no_such_column"])
