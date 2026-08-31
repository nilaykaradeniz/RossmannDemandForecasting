"""Read and clean the raw Rossmann files.

`DataLoader` reads `train.csv` and `store.csv`. It repairs the known problems
in the raw data, joins the store facts onto the daily rows, applies the
cleaning rules of this project, and returns one clean Polars DataFrame. The
result is ready for feature building.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# These columns hold whole numbers. We set the type ourselves, because a left
# join or quoted values in the CSV can turn them into text or into decimals.
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
    """Read, join and clean the raw Rossmann CSV files.

    Parameters
    ----------
    data_dir:
        The folder that holds `train.csv` and `store.csv`.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def load(self, drop_closed: bool = True) -> pl.DataFrame:
        """Return one clean table with the daily rows and the store facts.

        The method reads both files, repairs the type problem in
        `StateHoliday`, joins the store facts onto every daily row, and then
        cleans the result.

        Parameters
        ----------
        drop_closed:
            If `True` (the default), the closed days (`Open == 0`) are
            removed. This project trains and measures on open days only.
            Pass `False` to keep them, for example when you want to check the
            raw data first.
        """
        train = self._read_train()
        store = self._read_store()
        df = train.join(store, on="Store", how="left")
        df = self._clean(df, drop_closed=drop_closed)
        return df

    def _read_train(self) -> pl.DataFrame:
        """Read `train.csv`, with real dates and `StateHoliday` as text.

        In the raw file, `StateHoliday` mixes the number `0` and the text
        `'0'`. When we read the column as text, we get one clean set of
        labels: `'0'`, `'a'`, `'b'` and `'c'`.
        """
        path = self.data_dir / "train.csv"
        return pl.read_csv(
            path,
            try_parse_dates=True,
            schema_overrides={"StateHoliday": pl.Utf8},
        )

    def _read_store(self) -> pl.DataFrame:
        """Read `store.csv`. It holds one row of facts per store."""
        path = self.data_dir / "store.csv"
        return pl.read_csv(path)

    def _clean(self, df: pl.DataFrame, drop_closed: bool = True) -> pl.DataFrame:
        """Apply the cleaning rules of this project.

        - Remove the closed days (`Open == 0`) if the caller asks for it.
          A closed day tells us nothing about demand, and our metric (RMSPE)
          divides by the real sales, so rows with zero sales cannot stay.
        - Give the whole-number columns the type `Int64` again.
        - Sort by store and then by date, so that any later step that works
          with time gets the rows in a stable order.
        """
        if drop_closed:
            df = df.filter(pl.col("Open") == 1)
        df = df.with_columns(pl.col(_INT_COLUMNS).cast(pl.Int64))
        df = df.sort(["Store", "Date"])
        return df
