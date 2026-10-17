from datetime import date, datetime, timezone

import pytest

from gagcap.models import Observation
from gagcap.storage.lake import ObservationLake


def test_observation_rejects_naive_timestamps():
    with pytest.raises(ValueError):
        Observation("cbr", "key_rate", date(2025, 1, 1), 21.0, "percent", "https://cbr.ru",
                    datetime(2025, 1, 1), None, "provider_reported")


def test_empty_lake_write_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ObservationLake(tmp_path).write([])


def test_missing_observation_requires_provider_quality_flag():
    with pytest.raises(ValueError):
        Observation("fred", "CPI", date(2025, 1, 1), None, "index", "https://fred.stlouisfed.org",
                    datetime.now(timezone.utc), datetime.now(timezone.utc), "provider_reported")


def test_storage_read_query_policy_rejects_mutation(tmp_path):
    lake = ObservationLake(tmp_path)
    with pytest.raises(ValueError):
        lake.query("DELETE FROM observations")
