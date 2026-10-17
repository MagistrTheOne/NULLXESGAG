"""Opt-in provider checks. Never run automatically or outside Colab."""

import os
from datetime import date

import pytest

from gagcap.ingestion.cbr import key_rate
from gagcap.ingestion.fred import observations
from gagcap.ingestion.moex import candles


pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("GAGCAP_RUN_NETWORK_INTEGRATION") != "1",
    reason="Set GAGCAP_RUN_NETWORK_INTEGRATION=1 explicitly in Colab")]


def test_cbr_real_key_rate_series():
    rows = key_rate(date(2025, 1, 1), date(2025, 1, 31))
    assert rows and all(row.source == "cbr" for row in rows)


def test_moex_real_sber_candles():
    rows = candles("SBER", date(2025, 1, 1), date(2025, 1, 31), unit="RUB_per_share")
    assert rows and all(row.series_id.startswith("SBER:") for row in rows)


@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY is optional; configure Colab Secret")
def test_fred_real_vintage_series():
    rows = observations("CPIAUCSL", date(2024, 1, 1), date(2024, 12, 31),
                        vintage_as_of=date(2025, 1, 15), unit="index_1982_84_100")
    assert rows and all(row.quality == "provider_revision" for row in rows)
