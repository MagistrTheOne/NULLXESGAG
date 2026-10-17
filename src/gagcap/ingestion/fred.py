"""FRED observations and ALFRED vintage-aware adapter."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timezone

from gagcap.ingestion.http import ProviderError, get_json
from gagcap.models import Observation

_URL = "https://api.stlouisfed.org/fred/series/observations"
_VINTAGES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"


def observations(series_id: str, start: date, end: date, *, vintage_as_of: date | None = None,
                 unit: str, transform: str = "lin") -> list[Observation]:
    """Fetch FRED/ALFRED data. API key is read from FRED_API_KEY; no key means unavailable."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ProviderError("FRED unavailable: set FRED_API_KEY; no data substituted")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", series_id):
        raise ValueError("series_id contains unsupported characters")
    if end < start:
        raise ValueError("end must be on or after start")
    if not unit.strip():
        raise ValueError("A documented measurement unit is required for this FRED series")
    if transform not in {"lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"}:
        raise ValueError("Unsupported FRED units transformation")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": start.isoformat(), "observation_end": end.isoformat(),
              "units": transform}
    if vintage_as_of:
        params["realtime_start"] = vintage_as_of.isoformat()
        params["realtime_end"] = vintage_as_of.isoformat()
    result: list[dict] = []
    offset = 0
    page_size = 100_000
    expected_count: int | None = None
    while True:
        payload = get_json(_URL, {**params, "limit": str(page_size), "offset": str(offset)})
        if not isinstance(payload, dict) or not isinstance(payload.get("observations"), list):
            raise ProviderError("FRED response has no observations list", category="incomplete_response")
        if not isinstance(payload.get("count"), int) or not isinstance(payload.get("offset"), int):
            raise ProviderError("FRED response omitted valid count/offset pagination metadata",
                                category="incomplete_response")
        if expected_count is None:
            expected_count = payload["count"]
        if payload["offset"] != offset or payload["count"] != expected_count:
            raise ProviderError("FRED pagination metadata changed between pages", category="inconsistent_pages")
        page = payload["observations"]
        result.extend(page)
        if not page or len(result) >= expected_count:
            break
        offset += len(page)
        if len(page) < page_size:
            raise ProviderError("FRED returned a short page before declared count", category="incomplete_response")
    if expected_count is not None and len(result) != expected_count:
        raise ProviderError(f"FRED pagination incomplete: received {len(result)} of {expected_count}",
                            category="incomplete_response")
    retrieved = datetime.now(timezone.utc)
    rows: list[Observation] = []
    seen: set[date] = set()
    for item in result:
        try:
            obs_date = date.fromisoformat(item["date"])
            raw = item["value"]
            available = date.fromisoformat(item["realtime_start"])
            vintage = date.fromisoformat(item["realtime_end"]) if vintage_as_of else None
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(f"Malformed FRED observation: {item}") from exc
        if obs_date in seen:
            raise ProviderError(f"FRED repeated observation date {obs_date}", category="duplicate_observations")
        seen.add(obs_date)
        if raw == ".":
            value = None
            quality = "provider_missing_flag"
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ProviderError(f"Malformed FRED value: {item}") from exc
            quality = "provider_revision" if vintage_as_of else "provider_reported"
        effective_availability = vintage if vintage_as_of else available
        rows.append(Observation("fred", series_id, obs_date, value, unit, _URL, retrieved,
                                datetime.combine(effective_availability, time.max, timezone.utc),
                                quality, vintage, input_data_version=f"fred:{series_id}:{available}:{vintage}"))
    if not rows:
        raise ProviderError(f"FRED returned no numeric observations for {series_id} {start}..{end}")
    return rows


def historical_vintages(series_id: str, start: date, end: date, *, unit: str,
                        transform: str = "lin", max_requests: int = 100,
                        max_vintages: int = 50, minimum_interval_seconds: float = 0.25) -> list[Observation]:
    """Download bounded ALFRED snapshots; each vintage is queried as a dated realtime period.

    The explicit request budget includes vintage-date pages and observation pages. No retry loop
    hides provider throttling; HTTP 429 remains a structured ProviderError.
    """
    import time as time_module

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ProviderError("ALFRED unavailable: set FRED_API_KEY; no data substituted")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", series_id) or end < start or not unit.strip():
        raise ValueError("Valid series_id, ordered date range and documented unit are required")
    if max_requests < 2 or max_vintages < 1 or minimum_interval_seconds < 0:
        raise ValueError("Request/vintage budgets and interval must be positive")
    if transform not in {"lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"}:
        raise ValueError("Unsupported FRED units transformation")
    requests_used = 0
    last_request = 0.0

    def bounded_request(url: str, params: dict[str, str]) -> object:
        nonlocal requests_used, last_request
        if requests_used >= max_requests:
            raise ProviderError("ALFRED request budget exhausted", category="request_budget_exhausted")
        elapsed = time_module.monotonic() - last_request
        if requests_used and elapsed < minimum_interval_seconds:
            time_module.sleep(minimum_interval_seconds - elapsed)
        payload = get_json(url, params)
        requests_used += 1
        last_request = time_module.monotonic()
        return payload

    common = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    dates: list[date] = []
    offset, expected_count = 0, None
    while True:
        payload = bounded_request(_VINTAGES_URL, {**common, "limit": "10000", "offset": str(offset),
                                                   "sort_order": "asc"})
        if not isinstance(payload, dict) or not isinstance(payload.get("vintage_dates"), list):
            raise ProviderError("ALFRED response omitted vintage_dates", category="incomplete_response")
        count = payload.get("count")
        if not isinstance(count, int) or payload.get("offset") != offset:
            raise ProviderError("ALFRED vintage pagination metadata invalid", category="incomplete_response")
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise ProviderError("ALFRED vintage count changed between pages", category="inconsistent_pages")
        page = payload["vintage_dates"]
        for item in page:
            try:
                vintage = date.fromisoformat(item if isinstance(item, str) else item["date"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderError(f"Malformed ALFRED vintage date: {item}") from exc
            if start <= vintage <= end:
                dates.append(vintage)
        if len(dates) >= max_vintages:
            dates = dates[:max_vintages]
            break
        if not page or offset + len(page) >= expected_count:
            break
        offset += len(page)
    output: list[Observation] = []
    for vintage in dates:
        vintage_rows: list[dict] = []
        page_offset, snapshot_count = 0, None
        while True:
            page = bounded_request(_URL, {**common, "observation_start": start.isoformat(),
                                          "observation_end": end.isoformat(), "units": transform,
                                          "realtime_start": vintage.isoformat(),
                                          "realtime_end": vintage.isoformat(), "limit": "100000",
                                          "offset": str(page_offset)})
            if not isinstance(page, dict) or not isinstance(page.get("observations"), list):
                raise ProviderError("ALFRED snapshot omitted observations", category="incomplete_response")
            count = page.get("count")
            if not isinstance(count, int) or page.get("offset") != page_offset:
                raise ProviderError("ALFRED snapshot pagination metadata invalid", category="incomplete_response")
            if snapshot_count is None:
                snapshot_count = count
            elif count != snapshot_count:
                raise ProviderError("ALFRED snapshot count changed between pages", category="inconsistent_pages")
            rows = page["observations"]
            vintage_rows.extend(rows)
            if not rows or len(vintage_rows) >= snapshot_count:
                break
            if len(rows) < 100000:
                raise ProviderError("Short ALFRED snapshot page before declared count",
                                    category="incomplete_response")
            page_offset += len(rows)
        if len(vintage_rows) != snapshot_count:
            raise ProviderError("ALFRED snapshot is incomplete", category="incomplete_response")
        retrieved = datetime.now(timezone.utc)
        seen_dates: set[date] = set()
        for item in vintage_rows:
            try:
                observation_date = date.fromisoformat(item["date"])
                raw_value = item["value"]
                realtime_start = date.fromisoformat(item["realtime_start"])
                realtime_end = date.fromisoformat(item["realtime_end"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderError(f"Malformed ALFRED snapshot observation: {item}") from exc
            if observation_date in seen_dates:
                raise ProviderError(f"ALFRED repeated observation date {observation_date}",
                                    category="duplicate_observations")
            seen_dates.add(observation_date)
            value = None if raw_value == "." else float(raw_value)
            output.append(Observation(
                "fred", series_id, observation_date, value, unit, _URL, retrieved,
                datetime.combine(vintage, time.max, timezone.utc),
                "provider_missing_flag" if value is None else "provider_revision", vintage,
                f"alfred:{series_id}:{vintage}:{realtime_start}:{realtime_end}"))
    return output
