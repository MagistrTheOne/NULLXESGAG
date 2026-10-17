"""Naive forecasts used as a transparent reference, not as evidence of skill."""

from __future__ import annotations

import pandas as pd


def random_walk(last_value: float, horizon: int) -> list[float]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return [float(last_value)] * horizon


def historical_mean(values: pd.Series, horizon: int) -> list[float]:
    if horizon < 1 or values.empty or values.isna().any():
        raise ValueError("horizon must be positive and values complete")
    estimate = float(values.mean())
    return [estimate] * horizon


def seasonal_naive(values: pd.Series, horizon: int, *, season_length: int) -> list[float]:
    if values.empty or values.isna().any() or horizon < 1 or season_length < 1:
        raise ValueError("Complete values, positive horizon and season_length are required")
    if len(values) < season_length:
        raise ValueError("History is shorter than the requested seasonal period")
    season = values.iloc[-season_length:].astype(float).tolist()
    return [float(season[index % season_length]) for index in range(horizon)]
