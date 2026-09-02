import json
from datetime import date, timedelta

import polars as pl
import pytest

from src.forecaster import Forecaster, evaluate
from src.metrics import rmspe
from src.model import DEFAULT_PARAMS

FAST = dict(DEFAULT_PARAMS, eta=0.3)
CUT = date(2015, 4, 1)


@pytest.fixture(scope="module")
def fitted(train, calendar):
    before = train.filter(pl.col("Date") < CUT)
    return Forecaster(params=FAST).fit(before, calendar.filter(pl.col("Date") < CUT))


@pytest.fixture(scope="module")
def future(calendar):
    """Two weeks after the cut, closed days included, like test.csv."""
    return calendar.filter((pl.col("Date") >= CUT) & (pl.col("Date") < CUT + timedelta(days=14)))


def test_predict_scores_every_row_and_zeroes_the_closed_days(fitted, future):
    scored = fitted.predict(future)
    assert scored.height == future.height
    assert scored["prediction"].null_count() == 0
    assert (scored.filter(pl.col("Open") == 0)["prediction"] == 0).all()
    assert (scored.filter(pl.col("Open") == 1)["prediction"] > 0).all()


def test_the_forecast_is_close_to_the_real_sales(fitted, future):
    scored = fitted.predict(future).filter(pl.col("Open") == 1)
    assert rmspe(scored["Sales"], scored["prediction"]) < 0.15


def test_save_and_load_give_the_same_forecast(fitted, future, tmp_path):
    path = fitted.save(tmp_path / "model.pkl", extra={"note": "test"})
    loaded = Forecaster.load(path)
    assert loaded.predict(future)["prediction"].equals(fitted.predict(future)["prediction"])

    card = json.loads(path.with_suffix(".json").read_text())
    for key in ["fitted_at", "train_start", "train_end", "trees_per_segment", "packages", "note"]:
        assert key in card


def test_a_single_row_needs_its_opening_plan(fitted, future):
    one = future.filter((pl.col("Store") == 1) & (pl.col("Date") == CUT + timedelta(days=2)))
    plan = future.filter(pl.col("Store") == 1)
    with_plan = fitted.predict(one, planned=plan)["prediction"][0]
    in_full = fitted.predict(future).filter(
        (pl.col("Store") == 1) & (pl.col("Date") == CUT + timedelta(days=2)))["prediction"][0]
    assert with_plan == pytest.approx(in_full)


def test_predict_before_fit_is_an_error(future):
    with pytest.raises(RuntimeError):
        Forecaster().predict(future)


def test_evaluate_reports_the_overall_and_weekly_error(fitted, future):
    scored = fitted.predict(future)
    report = evaluate(scored)
    assert report["week"][0] == "all"
    open_rows = scored.filter(pl.col("Open") == 1)
    assert report["rmspe"][0] == pytest.approx(rmspe(open_rows["Sales"], open_rows["prediction"]), abs=1e-4)
    assert report.height >= 3        # "all" and at least two weeks
    assert report["rows"][0] == open_rows.height


def test_a_row_without_its_neighbours_is_refused(fitted, future):
    one = future.filter((pl.col("Store") == 1) & (pl.col("Date") == CUT + timedelta(days=5)))
    with pytest.raises(ValueError):
        fitted.predict(one)
    # With the plan around it, or with the guard switched off, it goes through.
    assert fitted.predict(one, planned=future)["prediction"][0] > 0
    assert fitted.predict(one, strict=False)["prediction"][0] > 0


def test_the_interval_holds_most_of_the_real_sales(fitted, future):
    scored = fitted.predict(future, interval=0.8).filter(pl.col("Open") == 1)
    assert (scored["low"] <= scored["prediction"]).all()
    assert (scored["prediction"] <= scored["high"]).all()
    inside = ((scored["Sales"] >= scored["low"]) & (scored["Sales"] <= scored["high"])).mean()
    assert 0.6 < inside < 0.95
    report = evaluate(scored, threshold=0.5)
    assert "coverage" in report.columns and "above_threshold" in report.columns
    assert not report["above_threshold"][0]


def test_two_seeds_average_and_reuse_the_tree_count(train, calendar, future):
    before = train.filter(pl.col("Date") < CUT)
    two = Forecaster(params=FAST, seeds=(42, 7)).fit(before, calendar.filter(pl.col("Date") < CUT))
    for models in two.models_.values():
        assert len(models) == 2
        assert models[0].best_rounds_ == models[1].best_rounds_
    scored = two.predict(future)
    assert scored["prediction"].null_count() == 0
    assert two.metadata_["seeds"] == [42, 7]
