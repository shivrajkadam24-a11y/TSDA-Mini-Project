"""Load MetroFlow data, forecast future demand, and create crowding alerts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from metroflow.models.experiment import calendar_features


SUPPORTED_STATION_ID = "611"
DATA_PATH = Path("data/processed/mta_hourly_station_611_clean.csv")
METRICS_PATH = Path("reports/generated/model_metrics.json")


@dataclass
class ForecastService:
    data: pd.DataFrame
    model: HistGradientBoostingRegressor
    origin: pd.Timestamp
    thresholds: pd.Series

    @classmethod
    def build(cls) -> "ForecastService":
        data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp").asfreq("h")
        if data.ridership.isna().any():
            raise ValueError("Prepared data contains missing ridership values.")
        origin = data.index.min()
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(calendar_features(data.index, origin), data.ridership)
        grouped = data.assign(is_weekend=data.index.dayofweek >= 5, hour=data.index.hour).groupby(["is_weekend", "hour"]).ridership.quantile(0.90)
        return cls(data=data, model=model, origin=origin, thresholds=grouped)

    def station(self) -> dict[str, str]:
        first = self.data.iloc[0]
        return {"station_id": str(first.station_complex_id), "station_name": str(first.station_complex), "borough": str(first.borough)}

    def history(self, start: datetime | None, end: datetime | None) -> list[dict[str, object]]:
        subset = self.data
        if start:
            subset = subset.loc[subset.index >= pd.Timestamp(start)]
        if end:
            subset = subset.loc[subset.index <= pd.Timestamp(end)]
        return [{"timestamp": timestamp.isoformat(), "ridership": round(float(row.ridership), 2), "was_imputed": bool(row.was_imputed)} for timestamp, row in subset.iterrows()]

    def forecast(self, horizon_hours: int) -> list[dict[str, object]]:
        future = pd.date_range(self.data.index.max() + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        predictions = np.clip(self.model.predict(calendar_features(future, self.origin)), 0, None)
        results = []
        for timestamp, prediction in zip(future, predictions):
            key = (timestamp.dayofweek >= 5, timestamp.hour)
            threshold = float(self.thresholds.loc[key])
            high = bool(prediction >= threshold)
            results.append({
                "timestamp": timestamp.isoformat(),
                "predicted_ridership": round(float(prediction), 2),
                "high_crowding_expected": high,
                "threshold": round(threshold, 2),
                "recommendation": "Deploy additional staff and issue a peak-period advisory." if high else "Normal staffing level is expected.",
            })
        return results

    def metrics(self) -> dict[str, object]:
        if METRICS_PATH.exists():
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return {"note": "Run metroflow.models.experiment to generate chronological model metrics."}


@lru_cache(maxsize=1)
def get_service() -> ForecastService:
    return ForecastService.build()
