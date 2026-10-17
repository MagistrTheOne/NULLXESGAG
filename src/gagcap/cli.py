"""Small CLI for explicit real-data ingestion into the local Parquet lake."""

from __future__ import annotations

import argparse
from datetime import date

from gagcap.ingestion.cbr import key_rate
from gagcap.ingestion.fred import observations as fred_observations
from gagcap.ingestion.moex import candles
from gagcap.storage.lake import ObservationLake


def main() -> None:
    parser = argparse.ArgumentParser(prog="gagcap")
    sub = parser.add_subparsers(dest="provider", required=True)
    for provider in ("cbr", "fred", "moex"):
        command = sub.add_parser(provider)
        command.add_argument("--start", type=date.fromisoformat, required=True)
        command.add_argument("--end", type=date.fromisoformat, required=True)
        command.add_argument("--store", default="data/lake")
        if provider == "fred":
            command.add_argument("--series", required=True)
            command.add_argument("--unit", required=True)
            command.add_argument("--transform", default="lin")
            command.add_argument("--vintage-as-of", type=date.fromisoformat)
        if provider == "moex":
            command.add_argument("--secid", required=True)
            command.add_argument("--engine", default="stock")
            command.add_argument("--market", default="shares")
            command.add_argument("--board", default="TQBR")
            command.add_argument("--unit", required=True)
    args = parser.parse_args()
    if args.provider == "cbr":
        rows = key_rate(args.start, args.end)
    elif args.provider == "fred":
        rows = fred_observations(args.series, args.start, args.end,
                                 vintage_as_of=args.vintage_as_of, unit=args.unit,
                                 transform=args.transform)
    else:
        rows = candles(args.secid, args.start, args.end, engine=args.engine,
                       market=args.market, board=args.board, unit=args.unit)
    path = ObservationLake(args.store).write(rows)
    print(f"Stored {len(rows)} provider observations at {path}")


if __name__ == "__main__":
    main()
