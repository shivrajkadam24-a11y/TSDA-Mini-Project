"""Validate and prepare MetroFlow's hourly station-ridership modelling table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def median_for_matching_hours(rows: list[dict[str, object]], timestamp: datetime) -> float:
    """Estimate an expected daylight-saving gap from comparable recent hours."""
    comparable = [
        float(row["ridership"])
        for row in rows
        if isinstance(row["timestamp"], datetime)
        and row["timestamp"].hour == timestamp.hour
        and row["timestamp"].weekday() == timestamp.weekday()
        and timestamp - timedelta(weeks=12) <= row["timestamp"] < timestamp
    ]
    if not comparable:
        raise ValueError(f"Cannot impute {timestamp}: no comparable prior observations.")
    return float(median(comparable))


def load_and_validate(input_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    with input_path.open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))

    required_columns = {
        "transit_timestamp",
        "station_complex_id",
        "station_complex",
        "borough",
        "ridership",
    }
    if not source_rows or not required_columns.issubset(source_rows[0]):
        raise ValueError("Input does not match the expected MetroFlow raw-data contract.")

    rows: list[dict[str, object]] = []
    for source_row in source_rows:
        ridership = float(source_row["ridership"])
        if ridership < 0:
            raise ValueError("Negative ridership is not valid for this application.")
        rows.append(
            {
                "timestamp": parse_timestamp(source_row["transit_timestamp"]),
                "station_complex_id": source_row["station_complex_id"],
                "station_complex": source_row["station_complex"],
                "borough": source_row["borough"],
                "ridership": ridership,
                "was_imputed": False,
            }
        )

    rows.sort(key=lambda row: row["timestamp"])
    timestamps = [row["timestamp"] for row in rows]
    duplicate_count = len(timestamps) - len(set(timestamps))
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate hourly timestamps.")

    observed = {timestamp for timestamp in timestamps if isinstance(timestamp, datetime)}
    expected = []
    current = timestamps[0]
    assert isinstance(current, datetime)
    final_timestamp = timestamps[-1]
    assert isinstance(final_timestamp, datetime)
    while current <= final_timestamp:
        if current not in observed:
            expected.append(current)
        current += timedelta(hours=1)

    report: dict[str, object] = {
        "input_record_count": len(source_rows),
        "start_timestamp_local": timestamps[0].isoformat(),
        "end_timestamp_local": timestamps[-1].isoformat(),
        "duplicate_timestamp_count": duplicate_count,
        "negative_ridership_count": 0,
        "missing_timestamps_before_repair": [timestamp.isoformat() for timestamp in expected],
        "imputation_policy": "Median of the preceding 12 matching weekday-and-hour observations; only for expected DST gap.",
    }
    if expected:
        # This project snapshot has the spring daylight-saving skipped hour. Any other
        # missing timestamp should stop the pipeline and be investigated manually.
        invalid_gaps = [
            timestamp
            for timestamp in expected
            if not (timestamp.month == 3 and timestamp.hour == 2 and 8 <= timestamp.day <= 14)
        ]
        if invalid_gaps:
            raise ValueError(f"Unexpected missing timestamps: {invalid_gaps}")
        for timestamp in expected:
            template = rows[0]
            rows.append(
                {
                    "timestamp": timestamp,
                    "station_complex_id": template["station_complex_id"],
                    "station_complex": template["station_complex"],
                    "borough": template["borough"],
                    "ridership": median_for_matching_hours(rows, timestamp),
                    "was_imputed": True,
                }
            )
        rows.sort(key=lambda row: row["timestamp"])

    report["output_record_count"] = len(rows)
    report["imputed_record_count"] = sum(bool(row["was_imputed"]) for row in rows)
    report["ridership_min"] = min(float(row["ridership"]) for row in rows)
    report["ridership_max"] = max(float(row["ridership"]) for row in rows)
    report["ridership_mean"] = round(sum(float(row["ridership"]) for row in rows) / len(rows), 3)
    return rows, report


def write_outputs(rows: list[dict[str, object]], report: dict[str, object], output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "station_complex_id",
        "station_complex",
        "borough",
        "ridership",
        "hour",
        "day_of_week",
        "is_weekend",
        "was_imputed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            timestamp = row["timestamp"]
            assert isinstance(timestamp, datetime)
            writer.writerow(
                {
                    "timestamp": timestamp.isoformat(),
                    "station_complex_id": row["station_complex_id"],
                    "station_complex": row["station_complex"],
                    "borough": row["borough"],
                    "ridership": round(float(row["ridership"]), 3),
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.weekday(),
                    "is_weekend": timestamp.weekday() >= 5,
                    "was_imputed": row["was_imputed"],
                }
            )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["output_file"] = str(output_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/mta_hourly_station_611_2025_onward.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mta_hourly_station_611_clean.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/data_quality_report.json"),
    )
    args = parser.parse_args()
    rows, report = load_and_validate(args.input)
    write_outputs(rows, report, args.output, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

