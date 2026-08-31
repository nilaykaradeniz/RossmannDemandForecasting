"""Loading and cleaning of the raw Rossmann data.

The :class:`DataLoader` reads ``train.csv`` and ``store.csv``, fixes the
known data quirks, merges the store metadata, applies the project cleaning
rules, and returns one clean Polars DataFrame ready for feature building.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Columns whose values are naturally integers. We cast them explicitly so a
# left join or quoted CSV values cannot leave them as strings or floats.
_INT_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "SchoolHoliday",
]


class DataLoader:
    """Read, merge and clean the raw Rossmann CSV files.

    Parameters
    ----------
    data_dir:
        Folder that contains ``train.csv`` and ``store.csv``.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def load(self, drop_closed: bool = True) -> pl.DataFrame:
        """Return one clean, merged DataFrame.

        Steps: read both files, fix the ``StateHoliday`` type mix, merge the
        store metadata onto every daily row, then apply the cleaning rules.

        Parameters
        ----------
        drop_closed:
            When ``True`` (default) closed days (``Open == 0``) are removed, as
            the project uses only open trading days for training and metrics.
            Pass ``False`` to keep them, e.g. to validate the raw data first.
        """
        train = self._read_train()
        store = self._read_store()
        df = train.join(store, on="Store", how="left")
        df = self._clean(df, drop_closed=drop_closed)
        return df

    def _read_train(self) -> pl.DataFrame:
        """Read ``train.csv`` with dates parsed and StateHoliday as text.

        ``StateHoliday`` mixes the integer ``0`` and the string ``'0'`` in the
        raw file. Forcing the column to ``Utf8`` gives the single clean set of
        labels ``{'0', 'a', 'b', 'c'}``.
        """
        path = self.data_dir / "train.csv"
        return pl.read_csv(
            path,
            try_parse_dates=True,
            schema_overrides={"StateHoliday": pl.Utf8},
        )

    def _read_store(self) -> pl.DataFrame:
        """Read ``store.csv`` (one row of metadata per store)."""
        path = self.data_dir / "store.csv"
        return pl.read_csv(path)

    def _clean(self, df: pl.DataFrame, drop_closed: bool = True) -> pl.DataFrame:
        """Apply the project cleaning rules.

        - Optionally drop closed days (``Open == 0``); they carry no sales
          signal and are excluded from both training and evaluation.
        - Cast the natural integer columns back to ``Int64``.
        - Sort by store then date so any later time-based logic is stable.
        """
        if drop_closed:
            df = df.filter(pl.col("Open") == 1)
        df = df.with_columns(pl.col(_INT_COLUMNS).cast(pl.Int64))
        df = df.sort(["Store", "Date"])
        return df
