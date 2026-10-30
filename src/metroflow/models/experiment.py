"""Chronologically evaluate three direct multi-step MetroFlow forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


TEST_HOURS = 24 * 28
SEASONAL_LAG = 24 * 7


def calendar_features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """Features known at forecast creation time, so no future leakage occurs."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    elapsed_days = (index - origin).total_seconds().to_numpy() / 86_400
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": (dow >= 5).astype(int),
            "hour": hour,
            "day_of_week": dow,
            "hour_x_dow": hour * (dow + 1),
            "trend_days": elapsed_days,
        },
        index=index,
    )


def scores(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    error = actual.to_numpy() - predicted
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "wape_percent": round(float(np.abs(error).sum() / actual.sum() * 100), 2),
    }


def main() -> None:
    data = pd.read_csv("data/processed/mta_hourly_station_611_clean.csv", parse_dates=["timestamp"]).set_index("timestamp").asfreq("h")
    y = data["ridership"].astype(float)
    train, test = y.iloc[:-TEST_HOURS], y.iloc[-TEST_HOURS:]
    origin = train.index.min()
    x_train, x_test = calendar_features(train.index, origin), calendar_features(test.index, origin)

    # 1. Transparent operational baseline: same station/hour one week ago.
    seasonal_naive = y.shift(SEASONAL_LAG).loc[test.index].to_numpy()

    # 2. Interpretable regression model with calendar/Fourier features.
    ridge = Ridge(alpha=100.0).fit(x_train, train)
    ridge_pred = np.clip(ridge.predict(x_test), 0, None)

    # 3. Non-linear ML model. Features are calendar-only and therefore usable for a full 7-day forecast.
    hgb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    hgb.fit(x_train, train)
    hgb_pred = np.clip(hgb.predict(x_test), 0, None)

    predictions = pd.DataFrame({"actual": test, "seasonal_naive": seasonal_naive, "ridge_calendar": ridge_pred, "hist_gradient_boosting": hgb_pred}, index=test.index)
    result = {name: scores(test, predictions[name].to_numpy()) for name in predictions.columns if name != "actual"}
    winner = min(result, key=lambda name: result[name]["wape_percent"])
    report = {
        "evaluation_design": "Final 28-day chronological holdout. All ML features are known at forecast origin; no future target values are used.",
        "train_start": train.index.min().isoformat(),
        "train_end": train.index.max().isoformat(),
        "test_start": test.index.min().isoformat(),
        "test_end": test.index.max().isoformat(),
        "test_hours": len(test),
        "models": result,
        "selected_by_lowest_wape": winner,
        "note": "SARIMAX and ETS are documented planned extensions when statsmodels becomes available; this run remains a valid three-model TSDA/ML comparison.",
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "holdout_predictions.csv")
    (output_dir / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
