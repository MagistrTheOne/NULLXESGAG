"""Append-only Parquet persistence for forecast outputs and experiment metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from gagcap.quant.models import Forecast


def write_forecasts(forecasts: Iterable[Forecast], path: str | Path) -> Path:
    rows = [forecast.as_record() for forecast in forecasts]
    if not rows:
        raise ValueError("Refusing to persist an empty forecast batch")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    if output.exists():
        frame = pd.concat([pd.read_parquet(output), frame], ignore_index=True)
    keys = ["model_id", "forecast_origin", "target_date", "horizon_calendar_days", "input_data_version"]
    frame = frame.drop_duplicates(subset=keys, keep="last")
    temporary = output.with_name(output.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output
