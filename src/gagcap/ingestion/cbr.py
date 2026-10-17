"""Bank of Russia public XML time-series adapter."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from xml.etree import ElementTree

import httpx

from gagcap.ingestion.http import ProviderError
from gagcap.models import Observation

_BASE = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"


def key_rate(start: date, end: date) -> list[Observation]:
    """Fetch CBR key-rate history using the official KeyRate XML endpoint."""
    if end < start:
        raise ValueError("end must be on or after start")
    url = _BASE
    envelope = (f'<?xml version="1.0" encoding="utf-8"?>'
                f'<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                f'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
                f'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
                f'<soap:Body><KeyRateXML xmlns="http://web.cbr.ru/">'
                f'<fromDate>{start.isoformat()}T00:00:00</fromDate>'
                f'<ToDate>{end.isoformat()}T23:59:59</ToDate>'
                f'</KeyRateXML></soap:Body></soap:Envelope>')
    try:
        response = httpx.post(
            url, content=envelope.encode("utf-8"),
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": "NULLXES-GAG-CAP-VC/0.1 (research; ceo@nullxes.com)",
                     "Content-Type": "text/xml; charset=utf-8",
                     "SOAPAction": "\"http://web.cbr.ru/KeyRateXML\""},
        )
        if response.status_code == 429:
            raise ProviderError("CBR rate limit (HTTP 429)", category="rate_limited", status_code=429,
                                retry_after=response.headers.get("Retry-After"))
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        result_node = next((node for node in root.iter()
                            if node.tag.rsplit("}", 1)[-1] == "KeyRateXMLResult"), None)
        result_text = result_node.text if result_node is not None else None
        if not result_text and result_node is not None and list(result_node):
            result_text = "".join(ElementTree.tostring(child, encoding="unicode")
                                  for child in list(result_node))
        if not result_text:
            raise ProviderError("CBR SOAP response omitted KeyRateXMLResult", category="incomplete_response")
        root = ElementTree.fromstring(result_text)
    except ProviderError:
        raise
    except (httpx.HTTPError, ElementTree.ParseError) as exc:
        category = "timeout" if isinstance(exc, httpx.TimeoutException) else "http_or_xml_error"
        raise ProviderError(f"CBR KeyRateXML unavailable: {exc}", category=category) from exc
    retrieved = Observation.now_utc()
    rows: list[Observation] = []
    for row in root.iter():
        fields = {child.tag.rsplit("}", 1)[-1]: child.text for child in row}
        raw_date = fields.get("Date", fields.get("date", fields.get("DT")))
        raw_rate = fields.get("Rate", fields.get("rate", fields.get("Value")))
        if not raw_date or raw_rate is None:
            continue
        try:
            normalized_date = raw_date[:10]
            try:
                obs_date = date.fromisoformat(normalized_date)
            except ValueError:
                obs_date = datetime.strptime(normalized_date, "%d.%m.%Y").date()
            value = float(str(raw_rate).replace(",", "."))
        except ValueError as exc:
            raise ProviderError(f"Malformed CBR key-rate row: {fields}") from exc
        if start <= obs_date <= end:
            if not 0 <= value <= 100:
                raise ProviderError(f"CBR key rate outside percent range: {value}", category="invalid_unit_value")
            rows.append(Observation("cbr", "key_rate", obs_date, value, "percent", str(response.url),
                                    retrieved, datetime.combine(obs_date, time.max, timezone.utc),
                                    "provider_reported"))
    if len({row.observation_date for row in rows}) != len(rows):
        raise ProviderError("CBR returned duplicate key-rate dates", category="duplicate_observations")
    if not rows:
        raise ProviderError(f"CBR returned no key-rate observations for {start}..{end}")
    return rows
