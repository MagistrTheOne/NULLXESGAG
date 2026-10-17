"""Typed records shared by ingestion, storage and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class Observation:
    source: str
    series_id: str
    observation_date: date
    value: float | None
    unit: str
    source_url: str
    retrieved_at: datetime
    available_at: datetime | None
    quality: str
    vintage_date: date | None = None
    input_data_version: str | None = None
    revision_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.series_id or not self.unit or not self.source_url:
            raise ValueError("source, series_id, unit and source_url are required")
        if self.quality not in {"provider_reported", "provider_missing_flag", "provider_revision"}:
            raise ValueError(f"Unsupported quality state: {self.quality}")
        if self.value is None and self.quality != "provider_missing_flag":
            raise ValueError("A missing value must preserve the provider_missing_flag quality state")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("Observation value must be finite")
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.available_at is not None and self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware when provided")

    @classmethod
    def now_utc(cls) -> datetime:
        return datetime.now(timezone.utc)

    def as_record(self) -> dict[str, Any]:
        return asdict(self)
