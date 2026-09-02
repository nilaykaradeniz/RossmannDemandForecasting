from datetime import date

import polars as pl

from src.data_loader import DataLoader


def test_load_drops_the_closed_days_and_keeps_the_store_facts(data_dir, calendar):
    df = DataLoader(data_dir).load()
    assert df.height == calendar.filter(pl.col("Open") == 1).height
    assert "StoreType" in df.columns and "CompetitionDistance" in df.columns
    assert df["StateHoliday"].dtype == pl.Utf8


def test_load_can_keep_the_closed_days(data_dir, calendar):
    assert DataLoader(data_dir).load(drop_closed=False).height == calendar.height


def test_load_rows_fills_what_a_new_file_may_leave_out(data_dir, tmp_path):
    (tmp_path / "new.csv").write_text(
        "Id,Store,Date,Open,Promo,StateHoliday,SchoolHoliday\n"
        "1,1,2015-06-01,1,1,0,0\n"
        "2,1,2015-06-02,,0,a,0\n"
    )
    rows = DataLoader(data_dir).load_rows(tmp_path / "new.csv")
    assert rows.height == 2
    assert rows["DayOfWeek"].to_list() == [1, 2]          # filled from the date
    assert rows["Open"].to_list() == [1, 1]               # a missing Open counts as open
    assert rows["StateHoliday"].to_list() == ["0", "a"]   # read as text, not as a number
    assert rows["StoreType"].null_count() == 0            # the store facts are joined
    assert rows["Date"][0] == date(2015, 6, 1)
