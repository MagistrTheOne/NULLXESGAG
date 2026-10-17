"""CAPM beta and required-return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    if (not asset_returns.index.is_unique or not market_returns.index.is_unique
            or not asset_returns.index.is_monotonic_increasing
            or not market_returns.index.is_monotonic_increasing):
        raise ValueError("Return indexes must be ordered and unique")
    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        raise ValueError("At least three aligned return observations are required")
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Aligned returns must be finite")
    market_variance = float(aligned.iloc[:, 1].var(ddof=1))
    if market_variance <= 0:
        raise ValueError("Market return variance must be positive")
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1], ddof=1) / market_variance)


def expected_return(risk_free: float, asset_beta: float, market_expected: float) -> float:
    values = np.asarray([risk_free, asset_beta, market_expected], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CAPM inputs must be finite")
    return float(risk_free + asset_beta * (market_expected - risk_free))


def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, *, window: int) -> pd.Series:
    if window < 3 or not asset_returns.index.equals(market_returns.index):
        raise ValueError("Aligned return series and window >= 3 are required")
    if not asset_returns.index.is_monotonic_increasing or not asset_returns.index.is_unique:
        raise ValueError("Return index must be ordered and unique")
    if (asset_returns.isna().any() or market_returns.isna().any()
            or not np.isfinite(asset_returns.to_numpy(dtype=float)).all()
            or not np.isfinite(market_returns.to_numpy(dtype=float)).all()):
        raise ValueError("CAPM returns must be complete and finite")
    covariance = asset_returns.rolling(window, min_periods=window).cov(market_returns)
    variance = market_returns.rolling(window, min_periods=window).var(ddof=1)
    return covariance / variance.where(variance > 0)
