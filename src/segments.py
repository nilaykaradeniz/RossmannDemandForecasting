"""Group the stores by the way they react, not by the amount they sell.

The model already knows how much a store sells: `store_mean_sales` carries that.
What it handles less well is *how* a store reacts. Two stores can both sell
8,000 units a day, and still be different businesses: one may gain 60 percent
from a promotion while the other gains 15 percent.

`StoreSegmenter` therefore builds a profile of each store from ratios, not from
levels. Every value says how the store behaves compared with itself, so a large
store and a small store with the same habits land in the same group.

The work happens in three steps.

1. **A profile per store.** Seven ratios: how much a promotion helps, how the
   week is shaped, how strong the season is, how school holidays act, the size
   of the basket, and whether the store is growing.
2. **Micro cells.** We cut three of those numbers into quantile bins, which
   gives a grid of small groups of stores. Cells with too few stores are joined
   to the cell nearest to them.
3. **Macro segments.** We describe every cell by its lift, meaning how far its
   behaviour sits from the average store, and then join the cells with similar
   lift into a handful of segments.

Everything is learned in `fit`, from the training rows only. `transform` only
looks the answer up, so a segment can never carry information from the window
we are about to score.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering

# The behaviour columns. Every one is a ratio, so they do not depend on how
# much the store sells.
RESPONSE_COLS = [
    "promo_uplift",
    "school_uplift",
    "weekend_ratio",
    "season_amp",
    "dec_lift",
    "basket",
    "momentum",
]

# The three columns we cut into a grid of micro cells. Size is one of them, so
# that a segment does not mix the largest and the smallest stores.
CELL_COLS = ["mean_sales", "promo_uplift", "season_amp"]
CELL_BINS = [4, 3, 3]

RECENT_DAYS = 90  # the window we use to see whether a store is growing


def build_profiles(train: pl.DataFrame) -> pl.DataFrame:
    """Return one row per store, with its level and its behaviour ratios.

    Parameters
    ----------
    train:
        The training rows of a fold. Nothing outside this table is read.
    """
    last_day = train["Date"].max()

    def ratio(mask: pl.Expr, name: str) -> pl.DataFrame:
        """Mean sales when the mask is true, divided by the mean when it is not."""
        return train.group_by("Store").agg(
            (
                pl.col("Sales").filter(mask).mean()
                / pl.col("Sales").filter(~mask).mean()
            ).alias(name)
        )

    profiles = train.group_by("Store").agg(
        pl.col("Sales").mean().alias("mean_sales"),
        pl.len().alias("days"),
        # The basket is the money one shopper leaves behind. It separates a
        # busy store with small baskets from a quiet store with large ones.
        (pl.col("Sales").sum() / pl.col("Customers").sum()).alias("basket"),
    )
    for expr, name in [
        (pl.col("Promo") == 1, "promo_uplift"),
        (pl.col("SchoolHoliday") == 1, "school_uplift"),
        (pl.col("DayOfWeek") >= 5, "weekend_ratio"),
    ]:
        profiles = profiles.join(ratio(expr, name), on="Store", how="left")

    # The shape of the year: the gap between the best and the worst month, and
    # how much December stands out.
    monthly = (
        train.group_by(["Store", pl.col("Date").dt.month().alias("_month")])
        .agg(pl.col("Sales").mean().alias("_ms"))
        .group_by("Store")
        .agg(
            (pl.col("_ms").max() / pl.col("_ms").min()).alias("season_amp"),
            (pl.col("_ms").filter(pl.col("_month") == 12).mean()
             / pl.col("_ms").mean()).alias("dec_lift"),
        )
    )
    profiles = profiles.join(monthly, on="Store", how="left")

    # Momentum: the last three months against the same three months a year
    # earlier. A store that is growing behaves differently from a shrinking one.
    recent = train.filter(pl.col("Date") > last_day - timedelta(days=RECENT_DAYS))
    year_before = train.filter(
        (pl.col("Date") > last_day - timedelta(days=365 + RECENT_DAYS))
        & (pl.col("Date") <= last_day - timedelta(days=365))
    )
    profiles = (
        profiles.join(
            recent.group_by("Store").agg(pl.col("Sales").mean().alias("_recent")),
            on="Store", how="left",
        )
        .join(
            year_before.group_by("Store").agg(pl.col("Sales").mean().alias("_before")),
            on="Store", how="left",
        )
        .with_columns((pl.col("_recent") / pl.col("_before")).alias("momentum"))
        .drop(["_recent", "_before"])
    )

    # A store with no promotion days, or with no December in its history, ends
    # up with an empty ratio. We fill those with the middle of the column, so
    # that the store stays in the grid instead of falling out of it.
    return profiles.with_columns(
        [pl.col(c).fill_null(pl.col(c).median()).fill_nan(pl.col(c).median())
         for c in RESPONSE_COLS]
    ).sort("Store")


class StoreSegmenter:
    """Put every store into one behaviour segment.

    Parameters
    ----------
    n_segments:
        How many segments we want in the end.
    min_days:
        A store needs at least this many trading days before it may help to
        build the grid. Stores with a shorter history are placed afterwards,
        so that a broken history cannot bend the groups.
    min_cell_stores:
        A micro cell with fewer stores than this is joined to its nearest
        neighbour.
    """

    def __init__(
        self,
        n_segments: int = 4,
        min_days: int = 365,
        min_cell_stores: int = 15,
    ) -> None:
        self.n_segments = n_segments
        self.min_days = min_days
        self.min_cell_stores = min_cell_stores

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame) -> "StoreSegmenter":
        """Learn the segments from the training rows only."""
        self.profiles_ = build_profiles(train)

        long_enough = self.profiles_.filter(pl.col("days") >= self.min_days)
        self.n_short_history_ = self.profiles_.height - long_enough.height
        if long_enough.height < self.n_segments * self.min_cell_stores:
            raise ValueError("Too few stores with a long enough history to segment.")

        # Standardise the behaviour columns, using this fold only.
        values = long_enough.select(RESPONSE_COLS).to_numpy()
        self._mean_ = values.mean(axis=0)
        self._std_ = np.where(values.std(axis=0) == 0, 1.0, values.std(axis=0))

        # Step 1: a grid of micro cells.
        self._cell_edges_ = {
            col: np.quantile(long_enough[col].to_numpy(),
                             np.linspace(0, 1, bins + 1)[1:-1])
            for col, bins in zip(CELL_COLS, CELL_BINS)
        }
        cell_id = self._cell_ids(long_enough)
        cell_id = self._merge_small_cells(cell_id, self._standardise(long_enough))

        # Step 2: describe every cell by its lift, then join similar cells.
        z = self._standardise(long_enough)
        cells = sorted(set(cell_id))
        centres = np.vstack([z[cell_id == c].mean(axis=0) for c in cells])
        n_clusters = min(self.n_segments, len(cells))
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(centres)
        cell_to_group = dict(zip(cells, labels))

        groups = np.array([cell_to_group[c] for c in cell_id])
        assignment = long_enough.select("Store").with_columns(
            pl.Series("_group", groups)
        )

        # Step 3: give the groups a stable order, so that segment 2 means the
        # same thing in every fold.
        #
        # We first tried to order them by lift, the share of sales divided by
        # the share of stores. That works for value segments, where one group
        # really does carry the money. It does not work here: our groups are
        # built from behaviour, and their lifts land within a few hundredths
        # of each other, so the smallest change swaps two labels. We order by
        # the promotion uplift instead, which separates the groups clearly.
        # Lift is still reported below, because it describes the segments well
        # even though it cannot order them.
        strength = (
            long_enough.select("Store")
            .with_columns(pl.Series("_group", groups),
                          pl.Series("_uplift", long_enough["promo_uplift"].to_numpy()))
            .group_by("_group")
            .agg(pl.col("_uplift").mean())
            .sort("_uplift", descending=True)
        )
        order = {g: i + 1 for i, g in enumerate(strength["_group"].to_list())}

        self.summary_ = self._summarise(assignment, train).with_columns(
            pl.col("_group").replace_strict(order, return_dtype=pl.Int32).alias("segment")
        ).drop("_group").sort("segment")

        assignment = assignment.with_columns(
            pl.col("_group").replace_strict(order, return_dtype=pl.Int32).alias("segment")
        ).drop("_group")

        # Segment centres, used to place the stores we held back.
        self._centres_ = np.vstack([
            z[groups == g].mean(axis=0) for g in sorted(order, key=order.get)
        ])

        short = self.profiles_.filter(pl.col("days") < self.min_days)
        if short.height:
            nearest = self._nearest_segment(self._standardise(short))
            assignment = pl.concat([
                assignment,
                short.select("Store").with_columns(pl.Series("segment", nearest)),
            ])

        self.assignment_ = assignment.sort("Store")
        # Stores we have never seen go to the largest segment.
        self.default_segment_ = int(
            self.assignment_.group_by("segment").len().sort("len", descending=True)["segment"][0]
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the `segment` column by looking the store up."""
        if not hasattr(self, "assignment_"):
            raise RuntimeError("StoreSegmenter must be fitted before transform().")
        return df.join(self.assignment_, on="Store", how="left").with_columns(
            pl.col("segment").fill_null(self.default_segment_)
        )

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        return self.fit(train).transform(train)

    # -------------------------------------------------------------- internal
    def _standardise(self, profiles: pl.DataFrame) -> np.ndarray:
        return (profiles.select(RESPONSE_COLS).to_numpy() - self._mean_) / self._std_

    def _cell_ids(self, profiles: pl.DataFrame) -> np.ndarray:
        """Give every store the number of the grid cell it falls into."""
        ids = np.zeros(profiles.height, dtype=int)
        for col, bins in zip(CELL_COLS, CELL_BINS):
            index = np.digitize(profiles[col].to_numpy(), self._cell_edges_[col])
            ids = ids * bins + index
        return ids

    def _merge_small_cells(self, cell_id: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Join every cell that holds too few stores to its nearest neighbour."""
        cell_id = cell_id.copy()
        while True:
            values, counts = np.unique(cell_id, return_counts=True)
            if len(values) <= self.n_segments or counts.min() >= self.min_cell_stores:
                return cell_id
            smallest = values[counts.argmin()]
            centres = {v: z[cell_id == v].mean(axis=0) for v in values}
            others = [v for v in values if v != smallest]
            nearest = min(
                others,
                key=lambda v: float(np.linalg.norm(centres[v] - centres[smallest])),
            )
            cell_id[cell_id == smallest] = nearest

    def _nearest_segment(self, z: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(z[:, None, :] - self._centres_[None, :, :], axis=2)
        return distance.argmin(axis=1) + 1

    def _summarise(self, assignment: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
        """Describe every group, and order the groups by lift.

        Lift is the share of the sales a group carries divided by the share of
        the stores it holds. A lift above 1 means the stores of that group sell
        more than an average store.
        """
        joined = train.join(assignment, on="Store", how="inner")
        totals = joined.group_by("_group").agg(
            pl.col("Sales").sum().alias("sales"),
            pl.col("Store").n_unique().alias("stores"),
        )
        return (
            totals.with_columns(
                (pl.col("sales") / pl.col("sales").sum()).alias("sales_share"),
                (pl.col("stores") / pl.col("stores").sum()).alias("store_share"),
            )
            .with_columns((pl.col("sales_share") / pl.col("store_share")).alias("lift"))
            .sort("lift", descending=True)
        )

    def describe(self) -> pl.DataFrame:
        """Return the average behaviour of every segment, for reading by eye."""
        joined = self.profiles_.join(self.assignment_, on="Store", how="left")
        return (
            joined.group_by("segment")
            .agg(
                pl.len().alias("stores"),
                pl.col("mean_sales").mean().round(0).alias("mean_sales"),
                *[pl.col(c).mean().round(3).alias(c) for c in RESPONSE_COLS],
            )
            .sort("segment")
        )
