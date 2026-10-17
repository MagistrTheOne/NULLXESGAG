"""Geopolitical event contracts, event studies and non-causal stress analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd


EventType = Literal["sanctions", "conflict", "trade_restriction", "export_rule",
                    "energy_crisis", "central_bank", "trade_relations", "other"]
EventStatus = Literal["CONFIRMED_EVENT", "UNVERIFIED_REPORT", "ANALYST_ASSUMPTION"]


@dataclass(frozen=True, slots=True)
class GeoEvent:
    event_id: str
    event_type: EventType
    source_url: str
    published_at: date
    event_date: date
    countries: tuple[str, ...]
    sectors: tuple[str, ...]
    assets: tuple[str, ...]
    evidence: str
    extraction_model: str
    status: EventStatus = "UNVERIFIED_REPORT"
    human_reviewed: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.source_url.startswith("https://") or not self.evidence:
            raise ValueError("Event id, HTTPS source and evidence are required")
        if not (self.countries or self.sectors or self.assets):
            raise ValueError("At least one affected country, sector or asset must be specified")
        if self.status == "CONFIRMED_EVENT" and not self.human_reviewed:
            raise ValueError("CONFIRMED_EVENT requires documented human review")


def events_available_at_origin(events: list[GeoEvent], forecast_origin: datetime) -> list[GeoEvent]:
    """Exclude events without a timezone-aware publication timestamp or published after origin."""
    if forecast_origin.tzinfo is None:
        raise ValueError("forecast_origin must be timezone-aware")
    available: list[GeoEvent] = []
    for event in events:
        published = event.published_at
        if isinstance(published, datetime):
            if published.tzinfo is None:
                raise ValueError(f"Event {event.event_id} publication timestamp must be timezone-aware")
            if published <= forecast_origin:
                available.append(event)
        # Date-only publication values cannot establish intraday point-in-time availability.
    return available


def event_study(asset_returns: pd.Series, event_date: date, *, estimation_window: int = 120,
                event_window_before: int = 1, event_window_after: int = 3) -> dict[str, float | int | str]:
    """Descriptive market-adjusted study using a constant-mean expected-return baseline."""
    if not isinstance(asset_returns.index, pd.DatetimeIndex) or not asset_returns.index.is_monotonic_increasing:
        raise ValueError("Returns require an increasing DatetimeIndex")
    if asset_returns.isna().any() or estimation_window < 20 or event_window_before < 0 or event_window_after < 0:
        raise ValueError("Complete returns and valid event/estimation windows are required")
    event_ts = pd.Timestamp(event_date)
    event_pos = int(asset_returns.index.searchsorted(event_ts))
    start, stop = event_pos - event_window_before, event_pos + event_window_after + 1
    train_start = start - estimation_window
    if train_start < 0 or stop > len(asset_returns):
        raise ValueError("Insufficient observations around event date")
    baseline = float(asset_returns.iloc[train_start:start].mean())
    event_returns = asset_returns.iloc[start:stop]
    abnormal = event_returns - baseline
    applied_event_date = pd.Timestamp(event_returns.index[0]).date()
    return {"method": "constant_mean_descriptive_event_study", "observations": int(len(abnormal)),
            "cumulative_abnormal_return": float(abnormal.sum()),
            "event_return_sum": float(event_returns.sum()),
            "requested_event_date": event_date.isoformat(),
            "first_session_in_window": applied_event_date.isoformat(),
            "causal_claim": "none; association only"}


def factor_regression(returns: pd.Series, factors: pd.DataFrame) -> dict[str, float | int | str]:
    """OLS exposure estimates; does not identify causal effects."""
    aligned = pd.concat([returns.rename("asset"), factors], axis=1, join="inner").dropna()
    if len(aligned) <= aligned.shape[1] + 2 or aligned.shape[1] < 1:
        raise ValueError("Insufficient complete observations for factor regression")
    import statsmodels.api as sm

    fit = sm.OLS(aligned.pop("asset"), sm.add_constant(aligned)).fit()
    return {"method": "ols_association_only", "observations": int(fit.nobs),
            "r_squared": float(fit.rsquared),
            **{f"coef_{name}": float(value) for name, value in fit.params.items()},
            "causal_claim": "none"}


def stress_test(returns: pd.Series, *, shocks: dict[str, float]) -> dict[str, float | str]:
    if returns.empty or returns.isna().any() or not shocks:
        raise ValueError("Complete historical returns and explicit shocks are required")
    baseline = float(returns.mean())
    output: dict[str, float | str] = {"baseline_mean_return": baseline,
                                     "method": "additive_hypothetical_shock"}
    for name, shock in shocks.items():
        if not name or not np.isfinite(shock):
            raise ValueError("Shock names and values must be valid")
        output[name] = baseline + float(shock)
    return output
