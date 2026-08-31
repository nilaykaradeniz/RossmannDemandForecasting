"""Train the gradient boosting model.

`SalesModel` wraps XGBoost and adds the two things this project needs.

**It learns on the logarithm of the sales.** Our metric is RMSPE, which reads
every error as a percentage. If we train on the raw sales, the model works
hardest on the large stores, because their errors are larger in absolute
numbers. In log space a step of ten percent is the same distance for every
store, so the training goal matches the metric.

**It chooses the number of trees without touching the fold.** Early stopping
needs a set to watch. If we used the validation window for that, we would pick
the tree count on the very rows we score, and the result would look better than
it is. Instead we cut a small inner window from the end of the training data,
find the best number of trees there, and then train the final model on the
whole training set with that number.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import xgboost as xgb

# A reasonable starting point. Step 8 of the plan tunes these; until then we
# keep them fixed, so that every comparison sees the same model.
DEFAULT_PARAMS: dict = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 42,
}

# Six weeks, the same length as a validation window. The inner window is only
# used to count the trees.
INNER_VALID_DAYS = 42


class SalesModel:
    """Predict daily sales with XGBoost, trained on the logarithm of the sales.

    Parameters
    ----------
    params:
        XGBoost parameters. `DEFAULT_PARAMS` is used when this is empty.
    num_boost_round:
        The largest number of trees we allow. Early stopping normally stops
        long before this.
    early_stopping_rounds:
        How many rounds without progress we accept before we stop.
    inner_valid_days:
        The length of the inner window that we use to count the trees.
    log_target:
        Train on `log1p(Sales)` and undo the logarithm when predicting. This
        is the right choice for RMSPE and it is the default. Set it to `False`
        only to show what the logarithm is worth.
    """

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
        inner_valid_days: int = INNER_VALID_DAYS,
        log_target: bool = True,
    ) -> None:
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.inner_valid_days = inner_valid_days
        self.log_target = log_target
        self.booster_: xgb.Booster | None = None
        self.feature_names_: list[str] = []
        self.best_rounds_: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pl.DataFrame,
        feature_names: list[str],
        target_col: str = "Sales",
        date_col: str = "Date",
    ) -> "SalesModel":
        """Train on `train`, and count the trees on an inner window.

        Parameters
        ----------
        train:
            The training rows, already passed through `FeatureBuilder`.
        feature_names:
            The columns that form the model matrix.
        target_col, date_col:
            The names of the target column and of the date column.
        """
        self.feature_names_ = list(feature_names)

        cut = train[date_col].max() - timedelta(days=self.inner_valid_days)
        inner_train = train.filter(pl.col(date_col) <= cut)
        inner_valid = train.filter(pl.col(date_col) > cut)
        if inner_valid.height == 0 or inner_train.height == 0:
            raise ValueError(
                "The inner window is empty. The training set is too short for "
                f"inner_valid_days={self.inner_valid_days}."
            )

        # Step one: how many trees do we need?
        watch = xgb.train(
            self.params,
            self._matrix(inner_train, target_col),
            num_boost_round=self.num_boost_round,
            evals=[(self._matrix(inner_valid, target_col), "inner_valid")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_rounds_ = watch.best_iteration + 1

        # Step two: train the real model on everything, with that count.
        self.booster_ = xgb.train(
            self.params,
            self._matrix(train, target_col),
            num_boost_round=self.best_rounds_,
            verbose_eval=False,
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Return the predicted sales, back on the normal scale.

        When the model works in log space, we undo the logarithm here. Sales
        can never be negative, so we also cut the result at zero.
        """
        if self.booster_ is None:
            raise RuntimeError("SalesModel must be fitted before predict().")
        raw = self.booster_.predict(self._matrix(df))
        pred = np.expm1(raw) if self.log_target else raw
        return np.clip(pred, 0, None)

    def importance(self, kind: str = "gain") -> pl.DataFrame:
        """Return the features, ordered by how much the model uses them."""
        if self.booster_ is None:
            raise RuntimeError("SalesModel must be fitted before importance().")
        scores = self.booster_.get_score(importance_type=kind)
        return pl.DataFrame(
            {
                "feature": self.feature_names_,
                kind: [float(scores.get(name, 0.0)) for name in self.feature_names_],
            }
        ).sort(kind, descending=True)

    # -------------------------------------------------------------- internal
    def _matrix(self, df: pl.DataFrame, target_col: str | None = None) -> xgb.DMatrix:
        """Build the XGBoost matrix, with the target in log space if asked."""
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise ValueError(f"These feature columns are missing: {missing}")

        x = df.select(self.feature_names_).to_numpy()
        label = None
        if target_col is not None:
            label = df[target_col].to_numpy().astype(float)
            if self.log_target:
                label = np.log1p(label)
        return xgb.DMatrix(x, label=label, feature_names=self.feature_names_)
