"""Return and realized-volatility calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    if prices.empty or prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise ValueError("prices must be non-empty, positive and complete")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices must have an increasing, unique time index")
    return np.log(prices.astype(float) / prices.astype(float).shift(1)).dropna()


def realized_volatility(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    if window < 2 or periods_per_year <= 0:
        raise ValueError("window must be >= 2 and annualization positive")
    if returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must not contain missing values")
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)
