"""Historical and EWMA conditional volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_volatility(returns: pd.Series, *, window: int, sessions_per_year: int = 252) -> pd.Series:
    if window < 2 or returns.empty or returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Complete returns and a window >= 2 are required")
    if sessions_per_year < 1:
        raise ValueError("sessions_per_year must be positive")
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(sessions_per_year)


def ewma_volatility(returns: pd.Series, *, decay: float = 0.94,
                    sessions_per_year: int = 252) -> pd.Series:
    if returns.empty or returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all() or not 0 < decay < 1:
        raise ValueError("Complete returns and 0 < decay < 1 are required")
    variance = returns.astype(float).pow(2).ewm(alpha=1 - decay, adjust=False).mean()
    return np.sqrt(variance * sessions_per_year)
