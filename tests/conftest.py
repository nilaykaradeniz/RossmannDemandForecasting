"""Small made-up data for the tests.

The real data cannot live in the repository, so the tests build a toy
version of it: a few dozen stores over about a year and a half, with a weekly
pattern, a promotion cycle, closed Sundays and a handful of holidays. The
numbers are random but the shape is the shape of the real table, so every
function of `src` can run on it in seconds.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

N_STORES = 70
FIRST_DAY = date(2014, 1, 1)
N_DAYS = 500
HOLIDAYS = {date(2014, 4, 18), date(2014, 4, 21), date(2014, 12, 25),
            date(2015, 1, 1), date(2015, 4, 3)}


def make_store_facts(n_stores: int = N_STORES) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "StoreType": rng.choice(["a", "b", "c", "d"], n_stores),
        "Assortment": rng.choice(["a", "b", "c"], n_stores),
        "CompetitionDistance": rng.integers(100, 20000, n_stores).astype(float),
        "CompetitionOpenSinceMonth": rng.integers(1, 13, n_stores).astype(float),
        "CompetitionOpenSinceYear": rng.integers(2005, 2014, n_stores).astype(float),
        "Promo2": rng.integers(0, 2, n_stores),
        "Promo2SinceWeek": rng.integers(1, 50, n_stores).astype(float),
        "Promo2SinceYear": rng.integers(2010, 2014, n_stores).astype(float),
        "PromoInterval": rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"], n_stores),
    })


def make_calendar(n_stores: int = N_STORES, n_days: int = N_DAYS,
                  first_day: date = FIRST_DAY) -> pl.DataFrame:
    """Every store on every day, closed days included, with the store facts."""
    rng = np.random.default_rng(2)
    days = [first_day + timedelta(days=i) for i in range(n_days)]
    level = rng.uniform(3000, 9000, n_stores)
    dow_factor = np.array([1.15, 1.0, 0.95, 0.95, 1.0, 0.85, 1.1])
    promo_gain = rng.uniform(1.1, 1.5, n_stores)

    rows = []
    for s in range(n_stores):
        for d in days:
            dow = d.isoweekday()
            promo = int(((d - first_day).days // 14) % 2 == 0 and dow <= 5)
            holiday = "a" if d in HOLIDAYS else "0"
            is_open = int(dow != 7 and holiday == "0")
            sales = 0
            if is_open:
                sales = level[s] * dow_factor[dow - 1] * (promo_gain[s] if promo else 1.0)
                sales = int(sales * rng.lognormal(0, 0.08))
            rows.append((s + 1, dow, d, sales, int(sales / 9) if sales else 0,
                         is_open, promo, holiday, int(d.month in (7, 8))))
    frame = pl.DataFrame(rows, schema=["Store", "DayOfWeek", "Date", "Sales", "Customers",
                                       "Open", "Promo", "StateHoliday", "SchoolHoliday"],
                         orient="row")
    return frame.join(make_store_facts(n_stores), on="Store", how="left").sort(["Store", "Date"])


@pytest.fixture(scope="session")
def calendar() -> pl.DataFrame:
    return make_calendar()


@pytest.fixture(scope="session")
def train(calendar) -> pl.DataFrame:
    """The open days only, as `DataLoader.load()` returns them."""
    return calendar.filter(pl.col("Open") == 1)


@pytest.fixture(scope="session")
def store_facts() -> pl.DataFrame:
    return make_store_facts()


@pytest.fixture
def data_dir(tmp_path: Path, calendar, store_facts) -> Path:
    """A folder in the shape of `data/`, with a small train.csv and store.csv."""
    calendar.select(["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open",
                     "Promo", "StateHoliday", "SchoolHoliday"]).write_csv(tmp_path / "train.csv")
    store_facts.write_csv(tmp_path / "store.csv")
    return tmp_path
