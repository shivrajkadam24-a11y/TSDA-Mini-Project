"""MetroFlow REST API."""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from metroflow.services.forecast import SUPPORTED_STATION_ID, get_service


app = FastAPI(title="MetroFlow API", version="0.1.0", description="Station footfall forecasts and peak-crowding decision support.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_methods=["GET"], allow_headers=["*"])


def verify_station(station_id: str) -> None:
    if station_id != SUPPORTED_STATION_ID:
        raise HTTPException(status_code=404, detail="Only station complex 611 is available in this release.")


@app.get("/health")
def health() -> dict[str, object]:
    service = get_service()
    return {"status": "ok", "data_end": service.data.index.max().isoformat(), "model": "hist_gradient_boosting_calendar"}


@app.get("/stations")
def stations() -> list[dict[str, str]]:
    return [get_service().station()]


@app.get("/forecast")
def forecast(station_id: str = SUPPORTED_STATION_ID, horizon_hours: int = Query(default=24, ge=1, le=168)) -> dict[str, object]:
    verify_station(station_id)
    rows = get_service().forecast(horizon_hours)
    return {"station": get_service().station(), "horizon_hours": horizon_hours, "forecast": rows, "alert_count": sum(row["high_crowding_expected"] for row in rows)}


@app.get("/history")
def history(station_id: str = SUPPORTED_STATION_ID, start: datetime | None = None, end: datetime | None = None) -> dict[str, object]:
    verify_station(station_id)
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must be before end.")
    rows = get_service().history(start, end)
    return {"station": get_service().station(), "history": rows}


@app.get("/metrics")
def metrics(station_id: str = SUPPORTED_STATION_ID) -> dict[str, object]:
    verify_station(station_id)
    return get_service().metrics()


@app.get("/insights")
def insights(station_id: str = SUPPORTED_STATION_ID) -> dict[str, object]:
    verify_station(station_id)
    return get_service().insights()
