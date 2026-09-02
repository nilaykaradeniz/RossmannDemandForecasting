"""Train the final model once, save it, and score new rows with it.

Every notebook before this one trained a model, measured it, and threw it
away. `Forecaster` is the model that stays. It holds every part that learns
from the data - the store statistics of `FeatureBuilder`, the groups of
`StoreSegmenter`, and one `SalesModel` per group - and it holds the calendar
of opening days, which is not learned but is needed to build the closed-day
columns for the days we predict.

Two ways to use it. From Python::

    forecaster = Forecaster().fit(train, calendar)
    forecaster.save("models/forecaster.pkl")
    scored = Forecaster.load("models/forecaster.pkl").predict(new_rows)

And from the command line::

    python -m src.forecaster fit
    python -m src.forecaster predict data/test.csv --out data/predictions.csv
    python -m src.forecaster evaluate actual_sales.csv data/predictions.csv

`fit` reads the data folder, trains on every open day and writes the model.
`predict` reads a file in the shape of `test.csv` and writes it back with a
`prediction` column, and with a `low` and a `high` column when an interval
is asked for. `evaluate` is the step that comes later, when the real sales
of the forecast period are known: it joins them to the predictions and
reports the error, overall and per week. Next to the model file, `save`
writes a small JSON file that says when the model was trained, on which
days, and with which packages.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import DataLoader
from src.features import (FeatureBuilder, build_closure_calendar,
                          _HOLIDAY_HORIZON, _CLOSURE_WINDOW)
from src.metrics import rmspe
from src.model import SalesModel, DEFAULT_PARAMS
from src.segments import StoreSegmenter

# The columns of the calendar that the closed-day family needs.
CALENDAR_COLUMNS = ["Store", "Date", "Open", "StateHoliday"]

# How far back the closed-day columns look. Beyond this, the past does not
# change any feature of a day we predict, so `predict` leaves it out.
CONTEXT_DAYS = max(_HOLIDAY_HORIZON, _CLOSURE_WINDOW)


class Forecaster:
    """The final model: features, store groups and one XGBoost per group.

    Parameters
    ----------
    params:
        XGBoost parameters. The defaults are the ones every notebook used;
        notebook 08 searched for better ones and found none.
    n_segments:
        How many store groups, and so how many models.
    seeds:
        The random seeds to train with. One seed is enough for a forecast,
        and it is the default. With several seeds, every group gets one
        model per seed and the forecast is their average; the models after
        the first reuse its tree count. Notebook 10 measures what the
        average is worth.
    """

    def __init__(self, params: dict | None = None, n_segments: int = 4,
                 seeds: tuple[int, ...] = (42,)) -> None:
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.n_segments = n_segments
        self.seeds = tuple(seeds)
        self.builder_: FeatureBuilder | None = None
        self.segmenter_: StoreSegmenter | None = None
        self.models_: dict[int, list[SalesModel]] = {}
        self.ratios_: dict[int, np.ndarray] = {}
        self.calendar_: pl.DataFrame | None = None
        self.train_end_ = None
        self.metadata_: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame, calendar: pl.DataFrame) -> "Forecaster":
        """Learn everything from the training rows.

        Parameters
        ----------
        train:
            The open days with their sales, from `DataLoader.load()`.
        calendar:
            The same period with the closed days kept, from
            `DataLoader.load(drop_closed=False)`. The closed-day columns are
            read from it.
        """
        self.calendar_ = _compact(calendar.select(CALENDAR_COLUMNS))
        self.train_end_ = train["Date"].max()
        self.builder_ = FeatureBuilder(calendar=self.calendar_, with_easter=True).fit(train)
        self.segmenter_ = StoreSegmenter(n_segments=self.n_segments).fit(train)
        built = self.segmenter_.transform(self.builder_.transform(train))

        self.models_, self.ratios_ = {}, {}
        for segment in range(1, self.n_segments + 1):
            rows = built.filter(pl.col("segment") == segment)
            if rows.height == 0:
                continue
            models, rounds = [], None
            for seed in self.seeds:
                model = SalesModel(params=dict(self.params, seed=seed),
                                   fixed_rounds=rounds)
                model.fit(rows, self.builder_.feature_names_)
                rounds = model.best_rounds_
                models.append(model)
            self.models_[segment] = models

            # For the interval: how far the real sales of the inner window
            # sat from the prediction of a model that had not seen them.
            inner = models[0].inner_scored_.filter(pl.col("Sales") > 0)
            self.ratios_[segment] = np.sort(
                (inner["Sales"] / inner["prediction"]).to_numpy())
            for model in models:
                model.inner_scored_ = None    # not needed any more, keeps the file small

        self.metadata_ = {
            "fitted_at": datetime.now().isoformat(timespec="seconds"),
            "train_start": str(train["Date"].min()),
            "train_end": str(self.train_end_),
            "train_rows": train.height,
            "stores": train["Store"].n_unique(),
            "features": len(self.builder_.feature_names_),
            "seeds": list(self.seeds),
            "trees_per_segment": {s: m[0].best_rounds_ for s, m in self.models_.items()},
            "stores_per_segment": {
                s: built.filter(pl.col("segment") == s)["Store"].n_unique()
                for s in self.models_},
            "interval_source": "real sales over predicted sales on the inner "
                               "window of each group, from the first seed",
            "params": self.params,
        }
        return self

    # -------------------------------------------------------------- predict
    def predict(self, rows: pl.DataFrame, planned: pl.DataFrame | None = None,
                interval: float | None = None, strict: bool = True) -> pl.DataFrame:
        """Return `rows` with a `prediction` column.

        Parameters
        ----------
        rows:
            The days to predict, in the shape of `test.csv` with the store
            facts joined, as `DataLoader.load_rows` returns them. Closed days
            (`Open == 0`) get a prediction of zero.
        planned:
            The opening plan around the days we predict: `Store`, `Date`,
            `Open` and `StateHoliday` for the future period. The closed-day
            columns need it - to say whether tomorrow is open, the model must
            know tomorrow. When it is left out, `rows` themselves are the
            plan, which is right when `rows` cover the whole period.
        interval:
            When given, for example 0.8, two more columns come back: `low`
            and `high`, the range that held that share of the real sales on
            the inner window of the group. Nothing is trained for it; it is
            the spread of the model's own recent misses, applied to the
            prediction.
        strict:
            Refuse rows whose day before is neither in the training calendar
            nor in the plan. A day that arrives alone would get "unknown"
            neighbours, and the closed-day columns would mislead the model.
            Set it to `False` only to see what that does.
        """
        if self.builder_ is None:
            raise RuntimeError("Forecaster must be fitted before predict().")
        plan = _compact((rows if planned is None else planned).select(CALENDAR_COLUMNS))
        stores = rows["Store"].unique().to_list()
        first_day = min(rows["Date"].min(), plan["Date"].min())

        # The known calendar, extended with the plan. Where both have a day,
        # the plan wins. Only the days close enough to matter are kept.
        known = self.calendar_.filter(
            pl.col("Store").is_in(stores)
            & (pl.col("Date") >= first_day - timedelta(days=CONTEXT_DAYS)))
        calendar = (
            pl.concat([known, plan])
            .unique(subset=["Store", "Date"], keep="last")
            .sort(["Store", "Date"])
        )
        if strict:
            self._check_neighbours(rows, calendar)
        self.builder_.set_calendar(build_closure_calendar(calendar))

        built = self.segmenter_.transform(self.builder_.transform(rows))
        prediction = np.zeros(built.height)
        low, high = np.zeros(built.height), np.zeros(built.height)
        for segment, models in self.models_.items():
            mask = (built["segment"] == segment).to_numpy()
            if not mask.any():
                continue
            part = built.filter(pl.col("segment") == segment)
            prediction[mask] = np.mean([m.predict(part) for m in models], axis=0)
            if interval is not None:
                tail = (1 - interval) / 2
                q_low, q_high = np.quantile(self.ratios_[segment], [tail, 1 - tail])
                low[mask] = prediction[mask] * q_low
                high[mask] = prediction[mask] * q_high
        closed = (built["Open"] == 0).to_numpy()
        prediction[closed] = low[closed] = high[closed] = 0.0

        columns = [pl.Series("prediction", prediction)]
        if interval is not None:
            columns += [pl.Series("low", low), pl.Series("high", high)]
        return rows.join(built.select(["Store", "Date"]).with_columns(columns),
                         on=["Store", "Date"], how="left")

    def _check_neighbours(self, rows: pl.DataFrame, calendar: pl.DataFrame) -> None:
        """Every day after the training period needs the day before it."""
        future = rows.filter(pl.col("Date") > self.train_end_)
        if future.height == 0:
            return
        need = future.select([pl.col("Store").cast(pl.Int32),
                              (pl.col("Date") - timedelta(days=1)).alias("Date")])
        missing = need.join(calendar.select(["Store", "Date"]), on=["Store", "Date"], how="anti")
        if missing.height:
            raise ValueError(
                f"{missing.height} rows have no opening plan for the day before them, "
                "so their closed-day columns would be unknown. Pass the plan of the "
                "days around them with planned=..., or set strict=False to see the "
                "effect."
            )

    # ----------------------------------------------------------- save, load
    def __getstate__(self) -> dict:
        """Leave the built closed-day table out of the file.

        `predict` builds that table again for the days it is asked about, so
        the file only needs the calendar it is built from.
        """
        state = dict(self.__dict__)
        if self.builder_ is not None:
            import copy
            builder = copy.copy(self.builder_)
            builder._closure_ = builder._closure_.clear()
            state["builder_"] = builder
        return state

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        """Write the model to `path`, and a JSON card next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        card = dict(self.metadata_)
        card.update({
            "python": platform.python_version(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
            "file_mb": round(path.stat().st_size / 1e6, 1),
        })
        if extra:
            card.update(extra)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        """Read a model that `save` has written."""
        with open(path, "rb") as f:
            forecaster = pickle.load(f)
        if not isinstance(forecaster, cls):
            raise TypeError(f"{path} does not hold a Forecaster.")
        return forecaster


def evaluate(scored: pl.DataFrame, actual_col: str = "Sales",
             pred_col: str = "prediction", threshold: float | None = None) -> pl.DataFrame:
    """Score a forecast against the real sales, overall and per week.

    This is the monitoring step: it runs when the real sales of a forecast
    period have arrived. Rows with zero sales are left out, as the metric
    requires. The first row of the result is the whole period, the rows
    after it are the weeks, so that a bad week shows up on its own.

    With `threshold`, a column `above_threshold` marks the rows whose error
    is larger than it - the flag that a weekly job would raise. When the
    forecast carries `low` and `high` columns, `coverage` says which share
    of the real sales fell inside that range.
    """
    rows = scored.filter(pl.col(actual_col) > 0)
    if rows.height == 0:
        raise ValueError("No rows with sales to score.")
    has_interval = "low" in rows.columns and "high" in rows.columns

    def line(name, part):
        out = {"week": name, "rows": part.height,
               "rmspe": rmspe(part[actual_col], part[pred_col])}
        if has_interval:
            inside = (part[actual_col] >= part["low"]) & (part[actual_col] <= part["high"])
            out["coverage"] = float(inside.mean())
        return out

    report = [line("all", rows)]
    weekly = rows.with_columns(pl.col("Date").dt.truncate("1w").alias("_week")).sort("_week")
    for key, part in weekly.group_by("_week", maintain_order=True):
        week = key[0] if isinstance(key, tuple) else key
        report.append(line(str(week), part))
    table = pl.DataFrame(report).with_columns(pl.selectors.float().round(4))
    if threshold is not None:
        table = table.with_columns((pl.col("rmspe") > threshold).alias("above_threshold"))
    return table


def _compact(calendar: pl.DataFrame) -> pl.DataFrame:
    """Give the calendar the smallest types that hold it."""
    return calendar.with_columns(
        pl.col("Store").cast(pl.Int32),
        pl.col("Open").cast(pl.Int8),
        pl.col("StateHoliday").cast(pl.Utf8),
    )


def _package_versions() -> dict[str, str]:
    import polars, numpy, xgboost, sklearn  # noqa: E401
    return {"polars": polars.__version__, "numpy": numpy.__version__,
            "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ------------------------------------------------------------------ command line
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the final model, or score rows with it.")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="train on every open day and save the model")
    fit.add_argument("--data", default="data", help="folder with train.csv and store.csv")
    fit.add_argument("--out", default="models/forecaster.pkl", help="where to write the model")
    fit.add_argument("--seeds", default="42", help="comma-separated seeds, one model each")

    predict = sub.add_parser("predict", help="score a file in the shape of test.csv")
    predict.add_argument("rows", help="the CSV to score")
    predict.add_argument("--model", default="models/forecaster.pkl")
    predict.add_argument("--data", default="data", help="folder with store.csv")
    predict.add_argument("--out", default=None, help="where to write the result (default: print a summary)")
    predict.add_argument("--interval", type=float, default=None,
                         help="add low and high columns, for example 0.8")

    check = sub.add_parser("evaluate", help="score a forecast against the real sales")
    check.add_argument("actual", help="CSV with Store, Date and Sales")
    check.add_argument("predictions", help="CSV written by the predict command")
    check.add_argument("--threshold", type=float, default=None,
                       help="flag the weeks whose error is above this value")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        actual = pl.read_csv(args.actual, try_parse_dates=True,
                             schema_overrides={"StateHoliday": pl.Utf8}).select(["Store", "Date", "Sales"])
        predictions = pl.read_csv(args.predictions, try_parse_dates=True)
        keep = [c for c in ["Store", "Date", "prediction", "low", "high"] if c in predictions.columns]
        joined = actual.join(predictions.select(keep), on=["Store", "Date"], how="inner")
        print(f"{joined.height:,} rows matched between the two files.")
        print(evaluate(joined, threshold=args.threshold))
        return
    if args.command == "fit":
        loader = DataLoader(args.data)
        seeds = tuple(int(s) for s in args.seeds.split(","))
        forecaster = Forecaster(seeds=seeds).fit(loader.load(), loader.load(drop_closed=False))
        path = forecaster.save(args.out)
        print(f"Model written to {path}. Trained on {forecaster.metadata_['train_rows']:,} rows, "
              f"{forecaster.metadata_['train_start']} to {forecaster.metadata_['train_end']}.")
    else:
        rows = DataLoader(args.data).load_rows(args.rows)
        scored = Forecaster.load(args.model).predict(rows, interval=args.interval)
        keep = [c for c in ["Id", "Store", "Date", "Open", "prediction", "low", "high"]
                if c in scored.columns]
        scored = scored.select(keep).with_columns(pl.selectors.float().round(2))
        if args.out:
            scored.write_csv(args.out)
            print(f"{scored.height:,} rows scored, written to {args.out}.")
        else:
            print(scored)


if __name__ == "__main__":
    # Run through the imported module, so that a model saved from the
    # command line holds `src.forecaster.Forecaster` and loads everywhere.
    from src.forecaster import main as _main
    _main(sys.argv[1:])
