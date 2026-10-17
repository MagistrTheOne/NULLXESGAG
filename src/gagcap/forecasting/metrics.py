"""Forecast metrics with strict finite and aligned-input checks."""

from __future__ import annotations

import numpy as np


def _arrays(actual: list[float], predicted: list[float]) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("actual and predicted must be non-empty aligned 1D sequences")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("metrics inputs must be finite")
    return y, p


def mae(actual: list[float], predicted: list[float]) -> float:
    y, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(y - p)))


def rmse(actual: list[float], predicted: list[float]) -> float:
    y, p = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def directional_accuracy(actual: list[float], predicted: list[float], *, origin: float) -> float:
    y, p = _arrays(actual, predicted)
    if not np.isfinite(origin):
        raise ValueError("origin must be finite")
    return float(np.mean(np.sign(y - origin) == np.sign(p - origin)))


def pinball_loss(actual: list[float], predicted_quantile: list[float], *, quantile: float) -> float:
    y, q = _arrays(actual, predicted_quantile)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    error = y - q
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(actual: list[float], lower: list[float], upper: list[float]) -> float:
    y, lo = _arrays(actual, lower)
    y2, hi = _arrays(actual, upper)
    if len(y2) != len(y) or not np.array_equal(y, y2) or np.any(lo > hi):
        raise ValueError("Bounds must align and lower must not exceed upper")
    return float(np.mean((y >= lo) & (y <= hi)))
