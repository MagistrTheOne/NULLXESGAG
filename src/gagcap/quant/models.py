"""Forecast models with explicit units and auditable result metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Literal


@dataclass(frozen=True, slots=True)
class Forecast:
    model_id: str
    forecast_origin: date
    target_date: date
    horizon_calendar_days: int
    horizon_sessions: int
    point_estimate: float
    interval_lower: float | None
    interval_upper: float | None
    confidence_level: float | None
    unit: str
    input_data_version: str
    assumptions: str

    def __post_init__(self) -> None:
        if self.horizon_calendar_days <= 0 or self.horizon_sessions <= 0:
            raise ValueError("Forecast horizons must be positive")
        if self.target_date <= self.forecast_origin:
            raise ValueError("target_date must follow forecast_origin")
        numeric = [self.point_estimate]
        if self.interval_lower is not None:
            numeric.extend([self.interval_lower, self.interval_upper])
        if self.confidence_level is not None:
            numeric.append(self.confidence_level)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Forecast values must be finite")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("Both interval bounds must be set or both omitted")
        if self.interval_lower is not None and self.interval_lower > self.interval_upper:
            raise ValueError("interval_lower must not exceed interval_upper")
        if self.confidence_level is not None and not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if not self.unit or not self.model_id or not self.input_data_version:
            raise ValueError("model_id, unit and input_data_version are required")

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Horizon:
    calendar_days: int
    sessions: int
    mapping: Literal["explicit_calendar"]


def calendar_horizon(calendar_days: int, *, session_calendar=None, forecast_origin=None) -> Horizon:
    """Map calendar horizon using actual ordered exchange sessions, never an annualized estimate."""
    if calendar_days not in {7, 30, 90}:
        raise ValueError("Supported calendar horizons are 7, 30 and 90 days")
    if session_calendar is None or forecast_origin is None:
        raise ValueError("An actual exchange session calendar and forecast origin are required")
    from gagcap.forecasting.backtesting import resolve_calendar_horizon

    sessions, _ = resolve_calendar_horizon(session_calendar, forecast_origin, calendar_days)
    return Horizon(calendar_days, sessions, "explicit_calendar")
