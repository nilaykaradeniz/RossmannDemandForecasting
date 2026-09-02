import numpy as np
import pytest

from src.metrics import rmspe


def test_perfect_prediction_scores_zero():
    assert rmspe([100, 200, 300], [100, 200, 300]) == 0.0


def test_known_value():
    # Every prediction is ten percent too high, so the error is exactly 0.1.
    assert rmspe([100, 200, 300], [110, 220, 330]) == pytest.approx(0.1)


def test_zero_sales_rows_are_left_out():
    # The zero row would divide by zero. It must not change the result.
    with_zero = rmspe([100, 0, 200], [110, 5000, 220])
    without = rmspe([100, 200], [110, 220])
    assert with_zero == pytest.approx(without)


def test_all_zero_sales_is_an_error():
    with pytest.raises(ValueError):
        rmspe([0, 0], [1, 2])


def test_missing_prediction_is_an_error():
    with pytest.raises(ValueError):
        rmspe([100, 200], [110, np.nan])


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        rmspe([100, 200], [110])
