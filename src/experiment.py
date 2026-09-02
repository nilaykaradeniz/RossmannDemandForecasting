"""Run one approach on every fold and collect the scores.

Every experiment in this project has the same shape: take a fold, learn
something from its training rows, predict its validation rows, and score the
result. `run_cv` does that loop, so each notebook only has to say what happens
inside a single fold.

The part that must stay visible is the fitting. Store statistics, promotion
uplifts and later the store segments are all learned from the target, so they
have to be built again inside every fold. `run_cv` therefore asks for a
function that receives the training rows of the fold and returns the scored
validation rows. Nothing is fitted outside that function.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from src.metrics import rmspe
from src.validation import RollingOriginSplit

# A function that trains on the rows of one fold and returns the validation
# rows with a `prediction` column added.
FitPredict = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def run_cv(
    df: pl.DataFrame,
    fit_predict: FitPredict,
    splitter: RollingOriginSplit | None = None,
    pred_col: str = "prediction",
    target_col: str = "Sales",
) -> tuple[pl.DataFrame, dict[int, pl.DataFrame]]:
    """Score one approach on every fold.

    Parameters
    ----------
    df:
        The clean table, with one row per store and open day.
    fit_predict:
        A function `(train, valid) -> valid_with_prediction`. It must learn
        everything it needs from `train` only.
    splitter:
        The splitter to use. A `RollingOriginSplit` with four folds by default.
    pred_col, target_col:
        The names of the prediction column and of the target column.

    Returns
    -------
    (results, scored)
        `results` holds one row per fold with its score. `scored` maps the fold
        number to the scored validation rows, so that we can look at the errors
        in more detail afterwards.
    """
    splitter = RollingOriginSplit() if splitter is None else splitter

    rows: list[dict] = []
    scored: dict[int, pl.DataFrame] = {}

    for fold in splitter.split(df):
        result = fit_predict(fold.train, fold.valid)
        if pred_col not in result.columns:
            raise ValueError(
                f"fit_predict must add a {pred_col!r} column, but it returned "
                f"{result.columns}."
            )
        scored[fold.index] = result
        rows.append(
            {
                "fold": fold.index,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
                "rows": result.height,
                "rmspe": rmspe(result[target_col], result[pred_col]),
            }
        )

    return pl.DataFrame(rows), scored


def compare(named_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Put several approaches side by side, one row per fold plus the mean.

    The last row holds the mean of each approach. A model is only convincing
    when it wins on every fold, not only on the mean, so both are shown.
    """
    if not named_results:
        raise ValueError("Please pass at least one set of results.")

    table = None
    for name, results in named_results.items():
        part = results.select(["fold", pl.col("rmspe").alias(name)])
        table = part if table is None else table.join(part, on="fold", how="full", coalesce=True)

    table = table.sort("fold").with_columns(pl.col("fold").cast(pl.Utf8))
    means = {"fold": "mean"}
    means.update({name: results["rmspe"].mean() for name, results in named_results.items()})
    return pl.concat([table, pl.DataFrame([means])], how="vertical")
