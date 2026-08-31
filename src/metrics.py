"""Measure the forecast error.

The metric of this project is RMSPE, the root mean square percentage error.
It compares every error with the size of the store, so an error of 500 units
counts much more for a small store than for a large one.

RMSPE divides by the real sales, so a row with zero sales has no error that we
can compute. We leave those rows out, and the competition does the same.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def rmspe(y_true, y_pred) -> float:
    """Return the root mean square percentage error.

    Parameters
    ----------
    y_true:
        The real sales. Rows with a value of zero are left out.
    y_pred:
        The predicted sales, in the same order as `y_true`.

    Returns
    -------
    float
        The error. A smaller number is better. A value of 0.15 means that a
        typical prediction is about 15 percent away from the real value.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    if y.shape != p.shape:
        raise ValueError(f"Shapes do not match: {y.shape} and {p.shape}.")
    if np.isnan(p).any():
        raise ValueError(
            f"{int(np.isnan(p).sum())} predictions are missing. "
            "Please give a value for every row."
        )

    keep = y != 0
    if not keep.any():
        raise ValueError("Every real value is zero, so RMSPE has no meaning here.")

    error = (y[keep] - p[keep]) / y[keep]
    return float(np.sqrt(np.mean(error**2)))


def rmspe_by(
    df: pl.DataFrame,
    group_cols: str | list[str],
    actual_col: str = "Sales",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Split the error by one or more columns.

    This shows us where the model is weak. We can group by store type, by
    month, by segment, or by anything else in the table.

    Parameters
    ----------
    df:
        A table that holds the real sales and the predictions.
    group_cols:
        The column or the columns to group by.
    actual_col, pred_col:
        The names of the two value columns.

    Returns
    -------
    pl.DataFrame
        One row per group, with the error and the number of rows behind it.
        The groups with the largest error come first. Please read a group with
        few rows with care, because its number is not stable.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    scored = df.filter(pl.col(actual_col) != 0)
    if scored.height == 0:
        raise ValueError("No row has sales above zero, so there is nothing to score.")

    squared_error = (
        (pl.col(actual_col) - pl.col(pred_col)) / pl.col(actual_col)
    ).pow(2)

    return (
        scored.with_columns(squared_error.alias("_sq"))
        .group_by(group_cols)
        .agg(
            pl.col("_sq").mean().sqrt().alias("rmspe"),
            pl.len().alias("rows"),
            pl.col(actual_col).mean().alias("mean_sales"),
        )
        .sort("rmspe", descending=True)
    )
