"""Persistent local alert feedback for the MetroFlow operations workflow."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path("data/app/metroflow_alerts.sqlite3")


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("""CREATE TABLE IF NOT EXISTS alert_feedback (
        alert_id TEXT PRIMARY KEY, status TEXT NOT NULL, note TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    )""")
    return connection


def save_feedback(alert_id: str, status: str, note: str) -> dict[str, str]:
    updated_at_utc = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            "INSERT INTO alert_feedback(alert_id, status, note, updated_at_utc) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(alert_id) DO UPDATE SET status=excluded.status, note=excluded.note, updated_at_utc=excluded.updated_at_utc",
            (alert_id, status, note.strip(), updated_at_utc),
        )
    return {"alert_id": alert_id, "status": status, "note": note.strip(), "updated_at_utc": updated_at_utc}


def all_feedback() -> dict[str, dict[str, str]]:
    with _connection() as connection:
        rows = connection.execute("SELECT alert_id, status, note, updated_at_utc FROM alert_feedback").fetchall()
    return {row[0]: {"status": row[1], "note": row[2], "updated_at_utc": row[3]} for row in rows}
