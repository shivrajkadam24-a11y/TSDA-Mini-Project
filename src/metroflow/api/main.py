"""MetroFlow REST API."""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from metroflow.services.forecast import SUPPORTED_STATION_ID, get_service
from metroflow.services.alerts import all_feedback, save_feedback


app = FastAPI(title="MetroFlow API", version="0.1.0", description="Station footfall forecasts and peak-crowding decision support.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_methods=["GET", "POST"], allow_headers=["*"])


class EventScenarioRequest(BaseModel):
    event_name: str = Field(min_length=2, max_length=100)
    event_start: datetime
    expected_attendance: int = Field(ge=0, le=500_000)
    subway_mode_share: float = Field(ge=0, le=1)
    station_share: float = Field(ge=0, le=1)
    staff_throughput: int = Field(ge=1, le=10_000)
    baseline_staff: int = Field(ge=0, le=500)


class AlertFeedbackRequest(BaseModel):
    status: str = Field(pattern="^(acknowledged|resolved)$")
    note: str = Field(default="", max_length=500)


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


@app.get("/anomalies")
def anomalies(station_id: str = SUPPORTED_STATION_ID, days: int = Query(default=28, ge=7, le=90)) -> dict[str, object]:
    verify_station(station_id)
    return get_service().anomalies(days)


@app.post("/scenario/event")
def event_scenario(payload: EventScenarioRequest, station_id: str = SUPPORTED_STATION_ID) -> dict[str, object]:
    verify_station(station_id)
    try:
        return get_service().simulate_event(**payload.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/alerts/feedback")
def alert_feedback() -> dict[str, object]:
    return {"feedback": all_feedback()}


@app.post("/alerts/{alert_id}/feedback")
def update_alert_feedback(alert_id: str, payload: AlertFeedbackRequest) -> dict[str, str]:
    return save_feedback(alert_id, payload.status, payload.note)
