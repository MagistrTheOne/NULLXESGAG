"""Multi-horizon point-in-time walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from datetime import timedelta

import pandas as pd
import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    origin: pd.Timestamp
    target: pd.Timestamp
    horizon_steps: int
    actual: float
    predicted: float
    origin_value: float
    training_observations: int
    window_type: str


def point_in_time_vintage(observations: pd.DataFrame, forecast_origin: pd.Timestamp) -> pd.DataFrame:
    """Select the last vintage actually available by origin for each series/date."""
    required = {"source", "series_id", "observation_date", "value", "available_at", "vintage_date"}
    if not required.issubset(observations.columns) or observations.empty:
        raise ValueError(f"Non-empty vintage data must include columns: {sorted(required)}")
    if forecast_origin.tzinfo is None:
        raise ValueError("forecast_origin must be timezone-aware")
    frame = observations.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    origin = forecast_origin.tz_convert("UTC")
    if frame["available_at"].isna().any():
        raise ValueError("available_at cannot contain missing or invalid timestamps")
    frame = frame[frame["available_at"] <= origin]
    if frame.empty:
        return frame
    return (frame.sort_values(["available_at", "vintage_date"], na_position="first")
            .drop_duplicates(["source", "series_id", "observation_date"], keep="last")
            .sort_values(["source", "series_id", "observation_date"]))


def walk_forward(series: pd.Series, *, horizon: int, min_train: int,
                 predictor: Callable[[pd.Series, int], list[float]],
                 available_at: pd.Series | None = None,
                 window: Literal["expanding", "rolling"] = "expanding",
                 rolling_window: int | None = None) -> list[ForecastRecord]:
    """At each origin use only observations published no later than that origin."""
    if (horizon < 1 or min_train < 2 or series.empty or series.isna().any()
            or not np.isfinite(series.to_numpy(dtype=float)).all()):
        raise ValueError("Require non-empty complete series, horizon >= 1 and min_train >= 2")
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series index must be an increasing DatetimeIndex")
    if not series.index.is_unique:
        raise ValueError("series index must be unique")
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be expanding or rolling")
    if window == "rolling" and (rolling_window is None or rolling_window < min_train):
        raise ValueError("rolling_window >= min_train is required for rolling mode")
    if available_at is not None:
        if not available_at.index.equals(series.index) or available_at.isna().any():
            raise ValueError("available_at must be complete and exactly aligned")
        if not isinstance(available_at.dtype, pd.DatetimeTZDtype):
            raise ValueError("available_at values must be timezone-aware timestamps")
        if series.index.tz is None or str(series.index.tz) != str(available_at.dtype.tz):
            raise ValueError("series index and available_at must use the same explicit timezone")
    records: list[ForecastRecord] = []
    for origin_pos in range(min_train - 1, len(series) - horizon):
        origin_time = pd.Timestamp(series.index[origin_pos])
        visible = series.iloc[:origin_pos + 1]
        if available_at is not None:
            visible = visible[available_at.iloc[:origin_pos + 1] <= origin_time]
        if window == "rolling" and rolling_window is not None:
            visible = visible.iloc[-rolling_window:]
        if len(visible) < min_train:
            continue
        prediction = predictor(visible.copy(), horizon)
        if len(prediction) != horizon:
            raise ValueError("predictor must return exactly horizon values")
        if not np.isfinite(np.asarray(prediction, dtype=float)).all():
            raise ValueError("predictor must return finite values")
        for step, predicted in enumerate(prediction, start=1):
            target_pos = origin_pos + step
            records.append(ForecastRecord(origin_time, pd.Timestamp(series.index[target_pos]), step,
                                          float(series.iloc[target_pos]), float(predicted), float(visible.iloc[-1]),
                                          len(visible), window))
    return records


def walk_forward_with_vintages(
    target: pd.Series,
    vintages: pd.DataFrame,
    *,
    horizon: int,
    min_train: int,
    predictor: Callable[[pd.Series, pd.DataFrame, int], list[float]],
    window: Literal["expanding", "rolling"] = "expanding",
    rolling_window: int | None = None,
) -> list[ForecastRecord]:
    """Re-select macro vintages at each origin before fitting a joint forecast."""
    if (not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None
            or not target.index.is_monotonic_increasing
            or not target.index.is_unique or target.isna().any()
            or not np.isfinite(target.to_numpy(dtype=float)).all()):
        raise ValueError("Target must be complete, finite and indexed by ordered unique timezone-aware timestamps")
    if horizon < 1 or min_train < 2:
        raise ValueError("horizon >= 1 and min_train >= 2 are required")
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be expanding or rolling")
    if window == "rolling" and (rolling_window is None or rolling_window < min_train):
        raise ValueError("rolling_window >= min_train is required")
    if "available_at" not in vintages or vintages["available_at"].isna().any():
        raise ValueError("Every macro vintage needs an availability timestamp")
    records: list[ForecastRecord] = []
    for origin_pos in range(min_train - 1, len(target) - horizon):
        origin = pd.Timestamp(target.index[origin_pos])
        pit = point_in_time_vintage(vintages, origin)
        train = target.iloc[:origin_pos + 1]
        if window == "rolling" and rolling_window is not None:
            train = train.iloc[-rolling_window:]
        if len(train) < min_train:
            continue
        predictions = predictor(train.copy(), pit.copy(), horizon)
        if len(predictions) != horizon:
            raise ValueError("predictor must return exactly horizon values")
        if not np.isfinite(np.asarray(predictions, dtype=float)).all():
            raise ValueError("predictor must return finite values")
        for step, prediction in enumerate(predictions, start=1):
            target_pos = origin_pos + step
            records.append(ForecastRecord(origin, pd.Timestamp(target.index[target_pos]), step,
                                          float(target.iloc[target_pos]), float(prediction),
                                          float(train.iloc[-1]), len(train), window))
    return records


def resolve_calendar_horizon(session_calendar: pd.DatetimeIndex, forecast_origin: pd.Timestamp,
                             calendar_days: int) -> tuple[int, pd.Timestamp]:
    """Resolve 7/30/90 calendar days to the first supplied exchange session on/after target date.

    The calendar must be a real provider/exchange session index; no 252/365 approximation is used.
    """
    if calendar_days not in {7, 30, 90}:
        raise ValueError("calendar_days must be 7, 30 or 90")
    if (not isinstance(session_calendar, pd.DatetimeIndex) or session_calendar.empty
            or not session_calendar.is_monotonic_increasing or not session_calendar.is_unique):
        raise ValueError("A non-empty ordered unique exchange session calendar is required")
    if forecast_origin.tzinfo is None:
        raise ValueError("forecast_origin must be timezone-aware")
    if session_calendar.tz is None or str(session_calendar.tz) != str(forecast_origin.tz):
        raise ValueError("Session calendar and forecast origin must have the same timezone")
    origin_pos = int(session_calendar.searchsorted(forecast_origin, side="left"))
    if origin_pos >= len(session_calendar) or session_calendar[origin_pos] != forecast_origin:
        raise ValueError("Forecast origin must match an actual session in the supplied calendar")
    target_date = forecast_origin + timedelta(days=calendar_days)
    target_pos = int(session_calendar.searchsorted(target_date, side="left"))
    if target_pos >= len(session_calendar):
        raise ValueError("Session calendar does not extend through requested calendar horizon")
    return target_pos - origin_pos, pd.Timestamp(session_calendar[target_pos])


def next_policy_decision(decision_calendar: pd.DatetimeIndex, forecast_origin: pd.Timestamp) -> pd.Timestamp:
    """Return the next known central-bank decision date from a supplied official calendar."""
    if (not isinstance(decision_calendar, pd.DatetimeIndex) or decision_calendar.empty
            or not decision_calendar.is_monotonic_increasing or not decision_calendar.is_unique
            or decision_calendar.tz is None or forecast_origin.tzinfo is None
            or str(decision_calendar.tz) != str(forecast_origin.tz)):
        raise ValueError("An ordered, timezone-aware official decision calendar is required")
    position = int(decision_calendar.searchsorted(forecast_origin, side="right"))
    if position >= len(decision_calendar):
        raise ValueError("Decision calendar has no future policy date")
    return pd.Timestamp(decision_calendar[position])


def validate_regular_frequency(index: pd.DatetimeIndex, expected_frequency: str) -> None:
    """Validate an asserted regular macro frequency; exchange-session calendars are not regularized."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3 or not index.is_monotonic_increasing:
        raise ValueError("At least three ordered timestamps are required for frequency validation")
    if not index.is_unique or pd.infer_freq(index) != expected_frequency:
        raise ValueError(f"Time index does not match declared frequency {expected_frequency}")


def walk_forward_calendar_horizon(series: pd.Series, *, calendar_days: int, min_train: int,
                                  predictor: Callable[[pd.Series, int], list[float]],
                                  available_at: pd.Series | None = None,
                                  window: Literal["expanding", "rolling"] = "expanding",
                                  rolling_window: int | None = None) -> list[ForecastRecord]:
    """Walk-forward terminal-horizon forecasts with origin-specific actual session mapping."""
    if calendar_days not in {7, 30, 90}:
        raise ValueError("calendar_days must be 7, 30 or 90")
    if (series.empty or series.isna().any() or not isinstance(series.index, pd.DatetimeIndex)
            or not series.index.is_monotonic_increasing or not series.index.is_unique
            or series.index.tz is None or not np.isfinite(series.to_numpy(dtype=float)).all()):
        raise ValueError("Series must be complete, finite and indexed by ordered unique timezone-aware sessions")
    if min_train < 2 or window not in {"expanding", "rolling"}:
        raise ValueError("min_train >= 2 and valid window mode are required")
    if window == "rolling" and (rolling_window is None or rolling_window < min_train):
        raise ValueError("rolling_window >= min_train is required")
    if available_at is not None:
        if (not available_at.index.equals(series.index) or available_at.isna().any()
                or not isinstance(available_at.dtype, pd.DatetimeTZDtype)):
            raise ValueError("available_at must be complete, timezone-aware and aligned with session index")
        if str(available_at.dtype.tz) != str(series.index.tz):
            raise ValueError("available_at and session index must use the same timezone")
    records: list[ForecastRecord] = []
    for origin_pos in range(min_train - 1, len(series)):
        origin = pd.Timestamp(series.index[origin_pos])
        target_limit = origin + timedelta(days=calendar_days)
        target_pos = int(series.index.searchsorted(target_limit, side="left"))
        if target_pos >= len(series):
            continue
        horizon_steps = target_pos - origin_pos
        if horizon_steps < 1:
            raise ValueError("Resolved target session must follow forecast origin")
        visible = series.iloc[:origin_pos + 1]
        if available_at is not None:
            visible = visible[available_at.iloc[:origin_pos + 1] <= origin]
        if window == "rolling" and rolling_window is not None:
            visible = visible.iloc[-rolling_window:]
        if len(visible) < min_train:
            continue
        predictions = predictor(visible.copy(), horizon_steps)
        if len(predictions) != horizon_steps or not np.isfinite(np.asarray(predictions, dtype=float)).all():
            raise ValueError("Predictor must return exactly horizon_steps finite forecasts")
        records.append(ForecastRecord(origin, pd.Timestamp(series.index[target_pos]), horizon_steps,
                                      float(series.iloc[target_pos]), float(predictions[-1]),
                                      float(visible.iloc[-1]), len(visible), window))
    return records


def persist_experiment(records: list[ForecastRecord], path: str | Path) -> None:
    if not records:
        raise ValueError("Cannot persist an empty experiment")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_parquet(output, index=False)


def strategy_net_return(gross_return: float, *, turnover: float, commission_bps: float,
                        slippage_bps: float) -> float:
    values = [gross_return, turnover, commission_bps, slippage_bps]
    if (not all(pd.notna(value) and np.isfinite(value) for value in values)
            or turnover < 0 or commission_bps < 0 or slippage_bps < 0):
        raise ValueError("Returns and non-negative costs/turnover must be supplied")
    return float(gross_return - turnover * (commission_bps + slippage_bps) / 10_000)
