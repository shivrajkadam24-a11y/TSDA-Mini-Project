"""Run MetroFlow EDA and save reproducible figures plus diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    source = Path("data/processed/mta_hourly_station_611_clean.csv")
    out = Path("reports/generated")
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(source, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp").asfreq("h")
    if df.ridership.isna().any():
        raise ValueError("Prepared EDA input contains missing target values.")
    daily = df.ridership.resample("D").sum()
    fig, ax = plt.subplots(figsize=(14, 4)); daily.plot(ax=ax, lw=.7, label="Daily entries"); daily.rolling(28, center=True).mean().plot(ax=ax, lw=2, label="28-day mean"); ax.set(title="Daily entries", ylabel="Entries"); ax.legend(); fig.tight_layout(); fig.savefig(out / "daily_trend.png", dpi=150); plt.close(fig)
    heat = df.pivot_table(index="day_of_week", columns="hour", values="ridership", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(15, 4)); sns.heatmap(heat, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean entries"}); ax.set(title="Typical hourly footfall", xlabel="Hour", ylabel="Day (0=Mon)"); fig.tight_layout(); fig.savefig(out / "hour_day_heatmap.png", dpi=150); plt.close(fig)
    # A dependency-free ACF screen keeps EDA runnable on the available local runtime.
    # Full STL, ADF and PACF remain in the committed notebook and run once statsmodels is installed.
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 120, 144, 168, 336]
    acf = {str(lag): round(float(df.ridership.autocorr(lag=lag)), 4) for lag in lags}
    fig, ax = plt.subplots(figsize=(10, 4)); ax.stem(lags, [acf[str(lag)] for lag in lags]); ax.set(title="Selected hourly autocorrelations", xlabel="Lag (hours)", ylabel="Autocorrelation"); fig.tight_layout(); fig.savefig(out / "selected_acf.png", dpi=150); plt.close(fig)
    profile = df.assign(hour=df.index.hour, day_type=np.where(df.index.dayofweek >= 5, "Weekend", "Weekday")).groupby(["day_type", "hour"]).ridership.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(12, 4)); profile.plot(ax=ax, marker="o", linewidth=2); ax.set(title="Typical hourly demand profile", xlabel="Hour of day", ylabel="Mean entries"); fig.tight_layout(); fig.savefig(out / "hourly_profiles.png", dpi=150); plt.close(fig)
    quality = df.assign(hour=df.index.hour, weekend=df.index.dayofweek >= 5)
    bounds = quality.groupby(["weekend", "hour"]).ridership.quantile([0.05, 0.95]).unstack()
    lower = [bounds.loc[(row.weekend, row.hour), 0.05] for row in quality.itertuples()]
    upper = [bounds.loc[(row.weekend, row.hour), 0.95] for row in quality.itertuples()]
    anomaly = (quality.ridership < lower) | (quality.ridership > upper)
    recent = quality.tail(24 * 28).copy(); recent["anomaly"] = anomaly.tail(24 * 28).to_numpy()
    fig, ax = plt.subplots(figsize=(14, 4)); ax.plot(recent.index, recent.ridership, linewidth=.7, color="#3478bf"); ax.scatter(recent.index[recent.anomaly], recent.loc[recent.anomaly, "ridership"], color="#d1495b", s=12, label="Comparable-hour anomaly"); ax.set(title="Recent observed entries and review signals", ylabel="Entries"); ax.legend(); fig.tight_layout(); fig.savefig(out / "recent_anomalies.png", dpi=150); plt.close(fig)
    peak_hours = {str(day): int(heat.loc[day].idxmax()) for day in heat.index}
    report = {"records": len(df), "start": df.index.min().isoformat(), "end": df.index.max().isoformat(), "imputed_hours": int(df.was_imputed.sum()), "daily_mean": round(float(daily.mean()), 2), "hourly_mean": round(float(df.ridership.mean()), 2), "hourly_median": round(float(df.ridership.median()), 2), "hourly_p90": round(float(df.ridership.quantile(.9)), 2), "minimum_hourly_entries": round(float(df.ridership.min()), 2), "maximum_hourly_entries": round(float(df.ridership.max()), 2), "comparable_hour_anomaly_count": int(anomaly.sum()), "selected_autocorrelation": acf, "peak_hour_by_day": peak_hours, "pending_diagnostics": ["STL decomposition", "ADF stationarity test", "PACF plot"]}
    (out / "eda_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
