"""Split the data by time, in the same shape as the real task.

The competition gives us six weeks of history that we cannot see, and asks for
a forecast of those six weeks. Our validation has to look the same. If we split
the rows at random, the model sees days that come after the days it predicts,
and the score becomes far too good.

`RollingOriginSplit` therefore cuts the data by date. It builds several
validation windows of six weeks, one after the other, going back in time. Each
window uses every earlier day for training. One single window can be a lucky
draw, so we look at the average and at the spread over all windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator

import polars as pl

# Six weeks. This is the length of the competition test period, so our
# validation windows have the same length.
HORIZON_DAYS = 42


@dataclass(frozen=True)
class Fold:
    """One split of the data, with its two parts and its dates."""

    index: int
    train: pl.DataFrame
    valid: pl.DataFrame
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date

    def describe(self) -> str:
        """A short line about this fold, useful when we print the folds."""
        return (
            f"fold {self.index}: "
            f"train {self.train_start} to {self.train_end} ({self.train.height:,} rows) | "
            f"valid {self.valid_start} to {self.valid_end} ({self.valid.height:,} rows)"
        )


class RollingOriginSplit:
    """Build validation windows that move back through time.

    The newest window ends on the last day of the data. The next window ends
    `step_days` earlier, and so on. Training always uses every day before the
    window starts, so a later fold trains on more history than an earlier one.
    We use all the history we have, because the data holds only two Decembers
    and we cannot afford to throw a season away.

    Parameters
    ----------
    n_splits:
        How many validation windows we want.
    horizon_days:
        The length of one window. The default matches the competition.
    step_days:
        How far back we move between two windows. The default is the length of
        the window, which makes the windows follow each other without overlap.
    min_train_days:
        The shortest training period we accept. A fold with less history than
        this raises an error, because its score would say more about the
        missing history than about the model.
    date_col:
        The name of the date column.
    """

    def __init__(
        self,
        n_splits: int = 4,
        horizon_days: int = HORIZON_DAYS,
        step_days: int | None = None,
        min_train_days: int = 365,
        date_col: str = "Date",
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be 1 or more.")
        if horizon_days < 1:
            raise ValueError("horizon_days must be 1 or more.")
        self.n_splits = n_splits
        self.horizon_days = horizon_days
        self.step_days = horizon_days if step_days is None else step_days
        self.min_train_days = min_train_days
        self.date_col = date_col

    def split(self, df: pl.DataFrame) -> Iterator[Fold]:
        """Yield the folds, from the oldest window to the newest one.

        Parameters
        ----------
        df:
            The clean table, with one row per store and open day.

        Yields
        ------
        Fold
            The training rows and the validation rows, plus their dates.
        """
        if self.date_col not in df.columns:
            raise ValueError(f"The table has no column named {self.date_col!r}.")

        last_day: date = df[self.date_col].max()
        first_day: date = df[self.date_col].min()

        for i in reversed(range(self.n_splits)):
            valid_end = last_day - timedelta(days=i * self.step_days)
            valid_start = valid_end - timedelta(days=self.horizon_days - 1)
            train_end = valid_start - timedelta(days=1)

            history_days = (train_end - first_day).days + 1
            if history_days < self.min_train_days:
                raise ValueError(
                    f"Fold {self.n_splits - i} would train on only {history_days} days, "
                    f"but min_train_days is {self.min_train_days}. Please use fewer "
                    "splits, a shorter horizon, or a smaller min_train_days."
                )

            train = df.filter(pl.col(self.date_col) <= train_end)
            valid = df.filter(
                (pl.col(self.date_col) >= valid_start)
                & (pl.col(self.date_col) <= valid_end)
            )
            if valid.height == 0:
                raise ValueError(
                    f"The window {valid_start} to {valid_end} holds no rows."
                )

            yield Fold(
                index=self.n_splits - i,
                train=train,
                valid=valid,
                train_start=first_day,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )

    def describe(self, df: pl.DataFrame) -> str:
        """Return one line per fold, so that we can check the split by eye."""
        return "\n".join(fold.describe() for fold in self.split(df))
