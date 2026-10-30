"""MetroFlow interactive decision-support dashboard."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.getenv("METROFLOW_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="MetroFlow", page_icon="🚇", layout="wide")


@st.cache_data(ttl=300)
def api_get(path: str, **params: object) -> dict | list:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.title("MetroFlow")
    st.caption("NYC Subway footfall forecasting and peak-crowding decision support")
    try:
        health = api_get("/health")
        station = api_get("/stations")[0]
    except requests.RequestException:
        st.error("The MetroFlow API is unavailable. Start it first with `uvicorn metroflow.api.main:app --reload`.")
        st.stop()

    with st.sidebar:
        st.header("Forecast settings")
        horizon = st.selectbox("Forecast horizon", options=[24, 48, 72, 168], format_func=lambda hours: f"Next {hours} hours")
        st.caption(f"Data through {pd.Timestamp(health['data_end']).strftime('%d %b %Y %H:%M')}")
        st.caption(f"Station: {station['station_name']}")

    forecast_response = api_get("/forecast", station_id=station["station_id"], horizon_hours=horizon)
    forecast = pd.DataFrame(forecast_response["forecast"])
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    alerts = forecast[forecast["high_crowding_expected"]]

    left, middle, right = st.columns(3)
    left.metric("Forecast hours", horizon)
    middle.metric("Predicted peak entries", f"{forecast.predicted_ridership.max():,.0f}")
    right.metric("High-crowding alerts", len(alerts))

    chart = go.Figure()
    chart.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.predicted_ridership, mode="lines+markers", name="Forecast"))
    chart.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.threshold, mode="lines", name="90th-percentile threshold", line={"dash": "dash"}))
    chart.add_trace(go.Scatter(x=alerts.timestamp, y=alerts.predicted_ridership, mode="markers", name="Alert", marker={"color": "#d62728", "size": 9}))
    chart.update_layout(title="Forecast footfall and high-crowding threshold", xaxis_title="Time", yaxis_title="Estimated entries", legend_title="")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Operational alerts")
    if alerts.empty:
        st.success("No high-crowding period is predicted for this horizon.")
    else:
        display = alerts[["timestamp", "predicted_ridership", "threshold", "recommendation"]].copy()
        display.columns = ["Forecast time", "Predicted entries", "Alert threshold", "Recommendation"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Recent footfall")
    start = (pd.Timestamp(health["data_end"]) - pd.Timedelta(days=14)).isoformat()
    history_response = api_get("/history", station_id=station["station_id"], start=start)
    history = pd.DataFrame(history_response["history"])
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    st.plotly_chart(px.line(history, x="timestamp", y="ridership", title="Observed hourly entries: last 14 days", labels={"ridership": "Entries", "timestamp": "Time"}), use_container_width=True)

    with st.expander("Model performance and limits"):
        metrics = api_get("/metrics", station_id=station["station_id"])
        if "models" in metrics:
            st.dataframe(pd.DataFrame(metrics["models"]).T.rename_axis("Model"), use_container_width=True)
            st.write(f"Current selected model: `{metrics['selected_by_lowest_wape']}`")
        st.info("Alerts are planning guidance based on historical station demand. They do not represent real-time crowd density or safety certification.")


if __name__ == "__main__":
    main()
