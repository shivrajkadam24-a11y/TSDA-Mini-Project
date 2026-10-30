"""Download a reproducible station-level hourly ridership series from NY Open Data.

The source dataset contains one row for each combination of fare category and
payment method. This module requests a server-side aggregation so the saved
CSV contains exactly one ridership observation per station complex per hour.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://data.ny.gov/resource/5wq4-mkjj.json"
DEFAULT_STATION_ID = "611"
DEFAULT_START = "2025-01-01T00:00:00"


def build_query(station_id: str, start: str, end: str | None) -> dict[str, str]:
    """Return a SoQL query that aggregates source rows at hourly resolution."""
    conditions = [
        "transit_mode = 'subway'",
        f"station_complex_id = '{station_id}'",
        f"transit_timestamp >= '{start}'",
    ]
    if end:
        conditions.append(f"transit_timestamp <= '{end}'")

    return {
        "$select": (
            "transit_timestamp,station_complex_id,station_complex,borough,"
            "sum(ridership) as ridership,sum(transfers) as transfers"
        ),
        "$where": " AND ".join(conditions),
        "$group": "transit_timestamp,station_complex_id,station_complex,borough",
        "$order": "transit_timestamp ASC",
        "$limit": "50000",
    }


def fetch_hourly_series(station_id: str, start: str, end: str | None) -> list[dict[str, str]]:
    """Request the aggregated series and fail clearly on incomplete API responses."""
    query = build_query(station_id, start, end)
    request = Request(
        f"{BASE_URL}?{urlencode(query)}",
        headers={"User-Agent": "MetroFlow academic project/1.0"},
    )
    with urlopen(request, timeout=90) as response:
        rows = json.loads(response.read().decode("utf-8"))
    if not rows:
        raise ValueError("The API returned no rows. Check station ID and date range.")
    if len(rows) >= 50000:
        raise ValueError("API result reached the row limit. Narrow the date range.")
    return rows


def write_snapshot(rows: list[dict[str, str]], output_path: Path, metadata_path: Path, query: dict[str, str]) -> None:
    """Save raw aggregate data plus retrieval metadata for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transit_timestamp",
        "station_complex_id",
        "station_complex",
        "borough",
        "ridership",
        "transfers",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_name": "NY Open Data - MTA Subway Hourly Ridership: Beginning 2025",
        "source_dataset_id": "5wq4-mkjj",
        "source_api_url": BASE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "record_count": len(rows),
        "output_file": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mta_hourly_station_611_2025_onward.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/raw/mta_hourly_station_611_2025_onward.metadata.json"),
    )
    args = parser.parse_args()

    query = build_query(args.station_id, args.start, args.end)
    rows = fetch_hourly_series(args.station_id, args.start, args.end)
    write_snapshot(rows, args.output, args.metadata_output, query)
    print(f"Saved {len(rows):,} hourly records to {args.output}")
    print(f"Saved retrieval metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()

