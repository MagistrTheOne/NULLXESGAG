"""Shared bounded HTTP client for official public APIs."""

from __future__ import annotations

import httpx
from urllib.parse import quote, quote_plus


class ProviderError(RuntimeError):
    """An official provider failed or returned data outside the adapter contract."""

    def __init__(self, message: str, *, category: str = "provider_error",
                 status_code: int | None = None, retry_after: str | None = None):
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retry_after = retry_after


def get_json(url: str, params: dict[str, str] | None = None) -> object:
    try:
        response = httpx.get(
            url,
            params=params,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": "NULLXES-GAG-CAP-VC/0.1 (research; ceo@nullxes.com)"},
        )
        if response.status_code == 429:
            raise ProviderError(f"Provider rate limit (HTTP 429): {url}", category="rate_limited",
                                status_code=429, retry_after=response.headers.get("Retry-After"))
        if response.status_code >= 400:
            raise ProviderError(f"Provider HTTP error {response.status_code}: {url}",
                                category="http_error", status_code=response.status_code,
                                retry_after=response.headers.get("Retry-After"))
        return response.json()
    except ProviderError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        detail = str(exc)
        if params and params.get("api_key"):
            secret = params["api_key"]
            detail = detail.replace(secret, "[REDACTED]")
            detail = detail.replace(quote(secret), "[REDACTED]")
            detail = detail.replace(quote_plus(secret), "[REDACTED]")
        category = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport_or_payload"
        raise ProviderError(f"Official API request failed: {url}: {detail}", category=category) from None
