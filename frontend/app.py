"""MetroFlow operations-command dashboard."""
from __future__ import annotations

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("METROFLOW_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="MetroFlow | Operations command", page_icon="🚇", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@keyframes railMove { from { transform: translateX(-8%); } to { transform: translateX(108%); } }
@keyframes glow { 0%,100% { box-shadow: 0 0 8px rgba(25,211,169,.22); } 50% { box-shadow: 0 0 28px rgba(25,211,169,.48); } }
@keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.stApp { background: radial-gradient(circle at 88% 8%, #15365b 0, #091728 34%, #050b14 100%); color:#edf6ff; }
[data-testid="stSidebar"] { background:#081320; border-right:1px solid rgba(126,185,220,.16); }
.hero { padding:1.6rem 1.8rem; border:1px solid rgba(75,199,255,.22); border-radius:18px; background:linear-gradient(112deg, rgba(16,45,73,.88), rgba(5,18,31,.62)); animation:rise .55s ease-out; overflow:hidden; }
.hero h1 { margin:0; font-size:2.2rem; letter-spacing:-.04em; }.hero p { color:#a9c8db; margin:.45rem 0 0; }
.rail { height:3px; background:linear-gradient(90deg,transparent,#4bc7ff,transparent); margin-top:1.1rem; position:relative; }.train { position:absolute; top:-12px; animation:railMove 5s linear infinite; color:#19d3a9; font-size:1.35rem; }
.status { display:inline-block; margin-top:.9rem; color:#19d3a9; font-size:.86rem; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stMetric"] { border:1px solid rgba(126,185,220,.16); background:rgba(10,30,49,.68); padding:.75rem; border-radius:13px; animation:glow 4s ease-in-out infinite; }
.section-label { color:#6fd6ff; font-weight:600; letter-spacing:.1em; font-size:.78rem; text-transform:uppercase; margin-top:1.7rem; }
div[data-testid="stDataFrame"] { border:1px solid rgba(126,185,220,.16); border-radius:12px; overflow:hidden; }
@media (prefers-reduced-motion: reduce) { .train, [data-testid="stMetric"] { animation:none; } }
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def api_get(path: str, **params: object) -> dict | list:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()

def format_hour(hour: int) -> str:
    return f"{hour:02d}:00"

def main() -> None:
    try:
        health = api_get("/health")
        station = api_get("/stations")[0]
        insights = api_get("/insights", station_id=station["station_id"])
    except requests.RequestException:
        st.error("MetroFlow API is offline. Start the FastAPI service, then refresh this dashboard.")
        st.stop()

    with st.sidebar:
        st.markdown("## MetroFlow")
        st.caption("Operations command center")
        horizon = st.radio("Forecast horizon", [24, 48, 72, 168], index=0, format_func=lambda value: f"Next {value} hours")
        st.divider(); st.write("**Station complex**"); st.caption(station["station_name"]); st.caption(station["borough"])
        st.divider(); st.caption("Alert rule: forecast ≥ historical 90th percentile for the comparable hour and day type.")

    st.markdown(f"""<div class="hero"><h1>MetroFlow <span style="color:#4bc7ff">/</span> station intelligence</h1><p>Forecasting passenger footfall, peak risk, and staffing actions for {station['station_name']}.</p><div class="rail"><span class="train">🚇</span></div><span class="status">● System ready · data through {pd.Timestamp(health['data_end']).strftime('%d %b %Y %H:%M')}</span></div>""", unsafe_allow_html=True)
    forecast_payload = api_get("/forecast", station_id=station["station_id"], horizon_hours=horizon)
    forecast = pd.DataFrame(forecast_payload["forecast"]); forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    alerts = forecast.loc[forecast.high_crowding_expected].copy(); summary = insights["historical_summary"]
    st.markdown('<div class="section-label">Live planning summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast peak", f"{forecast.predicted_ridership.max():,.0f}", "entries/hour")
    c2.metric("Peak-risk hours", len(alerts), f"of {horizon} forecast hours")
    c3.metric("Historical daily mean", f"{summary['mean_daily_entries']:,.0f}", "entries/day")
    c4.metric("Latest day vs prior week", f"{summary['change_vs_same_weekday_percent']:+.1f}%", "same weekday")

    overview, seasonal, model = st.tabs(["Forecast command", "Demand intelligence", "Model evidence"])
    with overview:
        left, right = st.columns([2.1, 1])
        with left:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.predicted_ridership, mode="lines+markers", name="Predicted entries", line={"color":"#4bc7ff","width":3}, marker={"size":5}))
            chart.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.threshold, mode="lines", name="Alert threshold", line={"color":"#f5b942","dash":"dash"}))
            chart.add_trace(go.Scatter(x=alerts.timestamp, y=alerts.predicted_ridership, mode="markers", name="High-risk hour", marker={"color":"#ff5d73","size":11,"symbol":"diamond"}))
            chart.update_layout(template="plotly_dark", height=440, title="Footfall forecast against peak-risk threshold", xaxis_title="Forecast time", yaxis_title="Estimated entries", legend={"orientation":"h","y":1.12}, margin={"l":0,"r":0,"t":58,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,25,42,.45)")
            st.plotly_chart(chart, use_container_width=True)
        with right:
            st.markdown("#### Operational action queue")
            if alerts.empty: st.success("No exceptional demand is predicted in this horizon.")
            else:
                for _, row in alerts.head(5).iterrows():
                    st.markdown(f"**{row.timestamp.strftime('%a · %H:%M')}**  \n{row.predicted_ridership:,.0f} forecast entries  \n`{row.risk_ratio:.2f}× threshold`  \n{row.recommendation}"); st.divider()
                if len(alerts) > 5: st.caption(f"+ {len(alerts)-5} more high-risk hours in this horizon")
        st.markdown("#### Alert schedule")
        table = forecast[["timestamp", "predicted_ridership", "threshold", "risk_ratio", "alert_level", "recommendation"]].copy()
        table.columns = ["Forecast time", "Predicted entries", "Threshold", "Risk ratio", "Status", "Recommended action"]
        st.dataframe(table, use_container_width=True, hide_index=True, height=330)
    with seasonal:
        hist_start = (pd.Timestamp(health["data_end"]) - pd.Timedelta(days=28)).isoformat()
        history = pd.DataFrame(api_get("/history", station_id=station["station_id"], start=hist_start)["history"])
        history["timestamp"] = pd.to_datetime(history["timestamp"]); history["date"] = history.timestamp.dt.date; history["hour"] = history.timestamp.dt.hour; history["day"] = history.timestamp.dt.day_name()
        l, r = st.columns(2)
        with l:
            daily = history.groupby("date", as_index=False).ridership.sum()
            fig = px.area(daily, x="date", y="ridership", title="Observed daily demand · last 28 days", color_discrete_sequence=["#4bc7ff"])
            fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,25,42,.45)", margin={"l":0,"r":0,"t":42,"b":0}, yaxis_title="Entries")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat = history.pivot_table(index="day", columns="hour", values="ridership", aggfunc="mean").reindex(order)
            fig = px.imshow(heat, aspect="auto", color_continuous_scale="Teal", title="Hourly demand heatmap · last 28 days", labels={"x":"Hour", "y":"Day", "color":"Entries"})
            fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", margin={"l":0,"r":0,"t":42,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Typical peak windows")
        peaks = pd.DataFrame(insights["peak_windows"]); peaks["Peak hour"] = peaks.peak_hour.map(format_hour); peaks = peaks[["day", "Peak hour", "mean_entries"]].rename(columns={"day":"Day", "mean_entries":"Mean entries"})
        st.dataframe(peaks, use_container_width=True, hide_index=True)
    with model:
        metrics = api_get("/metrics", station_id=station["station_id"])
        st.markdown("#### Chronological holdout evidence"); st.caption(metrics.get("evaluation_design", ""))
        metric_table = pd.DataFrame(metrics.get("models", {})).T.rename_axis("Model").reset_index(); st.dataframe(metric_table, use_container_width=True, hide_index=True)
        if not metric_table.empty:
            fig = px.bar(metric_table, x="Model", y="wape_percent", text_auto=".2f", color="Model", title="Final 28-day WAPE comparison", color_discrete_sequence=["#f5b942", "#7b8da8", "#19d3a9"])
            fig.update_layout(template="plotly_dark", height=360, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,25,42,.45)", yaxis_title="WAPE (%)", margin={"l":0,"r":0,"t":42,"b":0}); st.plotly_chart(fig, use_container_width=True)
        st.success(f"Production forecast model: {metrics.get('selected_by_lowest_wape', 'not available')}")
        season = insights["seasonality"]
        st.info(f"Demand repeats strongly: 24-hour autocorrelation = {season['autocorrelation_24h']:.3f}; weekly autocorrelation = {season['autocorrelation_168h']:.3f}. This validates seasonal baseline comparison and calendar-aware forecasting.")
        st.warning("Operational limit: forecasts estimate station entries, not platform occupancy or real-time safety conditions.")

if __name__ == "__main__": main()
