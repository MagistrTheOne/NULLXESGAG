"""Append/merge Parquet observation store with DuckDB query support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from gagcap.models import Observation


class ObservationLake:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, observations: Iterable[Observation]) -> Path:
        records = [item.as_record() for item in observations]
        if not records:
            raise ValueError("Refusing to write an empty observation batch")
        frame = pd.DataFrame.from_records(records)
        output = self.root / "observations.parquet"
        if output.exists():
            existing = pd.read_parquet(output)
            frame = pd.concat([existing, frame], ignore_index=True)
        keys = ["source", "series_id", "observation_date", "vintage_date"]
        frame = frame.drop_duplicates(subset=keys, keep="last")
        temporary = output.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
        return output

    def query(self, sql: str) -> pd.DataFrame:
        """Run a read-only DuckDB query with `observations` bound to stored Parquet files."""
        normalized = sql.lstrip().lower()
        if not normalized.startswith(("select", "describe", "show")) or ";" in normalized.rstrip("; "):
            raise ValueError("Only a single read-only SELECT/DESCRIBE/SHOW statement is allowed")
        files = list(self.root.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet observations in {self.root}")
        connection = duckdb.connect(database=":memory:", read_only=False)
        try:
            paths = ["'" + str(path).replace("'", "''") + "'" for path in files]
            connection.execute("CREATE TEMP TABLE observations AS SELECT * FROM read_parquet([" + ",".join(paths) + "])")
            connection.execute("SET enable_external_access=false")
            return connection.execute(sql).fetchdf()
        finally:
            connection.close()
