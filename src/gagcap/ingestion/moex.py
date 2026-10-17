"""MOEX ISS candle adapter with pagination and explicit board selection."""

from __future__ import annotations

from datetime import date, datetime, timezone
import re

import httpx

from gagcap.ingestion.http import ProviderError
from gagcap.models import Observation

_BASE = "https://iss.moex.com/iss"
_INTERVALS = {1, 10, 60, 24, 7, 31, 4}
_UNITS = {"RUB_per_share", "index_points", "provider_close_price"}
_PAGE_SIZE = 100


def candles(secid: str, start: date, end: date, *, engine: str = "stock",
            market: str = "shares", board: str = "TQBR", interval: int = 24,
            unit: str) -> list[Observation]:
    """Fetch daily MOEX candles; this adapter does not imply a data-use license."""
    if end < start:
        raise ValueError("end must be on or after start")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (secid, engine, market, board)):
        raise ValueError("secid, engine, market and board must be simple provider identifiers")
    if interval not in _INTERVALS:
        raise ValueError("Unsupported MOEX candle interval")
    if unit not in _UNITS:
        raise ValueError(f"Unsupported or unspecified MOEX unit: {unit}; expected one of {sorted(_UNITS)}")
    base_url = f"{_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"
    offset, result, previous_pages, page_count = 0, [], set(), 0
    retrieved = datetime.now(timezone.utc)
    while True:
        params = {"from": start.isoformat(), "till": end.isoformat(), "interval": str(interval),
                  "start": str(offset), "limit": str(_PAGE_SIZE)}
        try:
            response = httpx.get(base_url, params=params, timeout=30,
                                 headers={"User-Agent": "NULLXES-GAG-CAP-VC/0.1 research"})
            if response.status_code == 429:
                raise ProviderError("MOEX ISS rate limit (HTTP 429)", category="rate_limited",
                                    status_code=429, retry_after=response.headers.get("Retry-After"))
            if response.status_code >= 400:
                raise ProviderError(f"MOEX ISS HTTP error {response.status_code}",
                                    category="http_error", status_code=response.status_code,
                                    retry_after=response.headers.get("Retry-After"))
            payload = response.json()
        except ProviderError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            category = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport_or_payload"
            raise ProviderError(f"MOEX ISS request failed: {base_url}: {exc}", category=category) from exc
        if not isinstance(payload, dict):
            raise ProviderError("MOEX response root is not an object")
        block = payload.get("candles")
        if not isinstance(block, dict) or not isinstance(block.get("columns"), list) or not isinstance(block.get("data", []), list):
            raise ProviderError("MOEX response has no candles block")
        columns, data = block["columns"], block.get("data", [])
        try:
            positions = {name: columns.index(name) for name in ("begin", "close") if name in columns}
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"MOEX candle columns are malformed: {columns}") from exc
        if set(positions) != {"begin", "close"}:
            raise ProviderError(f"MOEX candle schema missing begin/close: {columns}")
        page_count += 1
        if page_count > 5000:
            raise ProviderError("MOEX pagination exceeded the 5,000-page safety ceiling")
        page_signature = tuple(tuple(row[:]) for row in data)
        if page_signature and page_signature in previous_pages:
            raise ProviderError("MOEX repeated a candle page while paginating")
        previous_pages.add(page_signature)
        for row in data:
            try:
                obs_date = date.fromisoformat(row[positions["begin"]][:10])
                close = row[positions["close"]]
            except (IndexError, TypeError, ValueError) as exc:
                raise ProviderError(f"Malformed MOEX candle row: {row}") from exc
            try:
                if not start <= obs_date <= end:
                    raise ProviderError(f"MOEX returned out-of-range session date {obs_date}", category="invalid_date")
                result.append(Observation("moex", f"{secid}:{engine}:{market}:{board}", obs_date,
                                          float(close) if close is not None else None, unit, str(response.url), retrieved,
                                          None, "provider_reported" if close is not None else "provider_missing_flag"))
            except (TypeError, ValueError) as exc:
                raise ProviderError(f"Invalid MOEX close value in candle: {row}") from exc
        cursor = payload.get("candles.cursor", {})
        total = cursor.get("total") if isinstance(cursor, dict) else None
        if not data:
            break
        if total is not None:
            try:
                if int(cursor.get("offset", offset)) != offset or int(total) < offset + len(data):
                    raise ProviderError("MOEX cursor metadata is inconsistent", category="inconsistent_pages")
            except (TypeError, ValueError) as exc:
                raise ProviderError("MOEX cursor total/offset is malformed", category="incomplete_response") from exc
            if offset + len(data) >= int(total):
                break
        elif len(data) < _PAGE_SIZE:
            break
        offset += len(data)
    if not result:
        raise ProviderError(f"MOEX returned no candles for {secid} {start}..{end}")
    dates = [row.observation_date for row in result]
    if len(dates) != len(set(dates)):
        raise ProviderError("MOEX returned duplicate candle dates", category="duplicate_observations")
    if dates != sorted(dates):
        raise ProviderError("MOEX candle pages are not in ascending date order", category="inconsistent_pages")
    return result
