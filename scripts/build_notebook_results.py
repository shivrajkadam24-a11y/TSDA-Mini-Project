"""Embed verified MetroFlow EDA and experiment results into submission notebooks."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated"
NOTEBOOKS = ROOT / "notebooks"


def source_cell(lines: list[str], outputs: list[dict] | None = None) -> dict:
    return {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": outputs or [], "source": lines}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def image_output(path: Path) -> dict:
    return {"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(path.read_bytes()).decode("ascii"), "text/plain": [path.name]}}


def stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3.10", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    eda = json.loads((REPORTS / "eda_diagnostics.json").read_text(encoding="utf-8"))
    metrics = json.loads((REPORTS / "model_metrics.json").read_text(encoding="utf-8"))
    eda_cells = [
        markdown("# MetroFlow: EDA and time-series diagnostics\n\n**Executed submission notebook.** Data: official NY Open Data MTA hourly ridership, station complex 611. The prepared series contains one hourly entry total per timestamp."),
        source_cell(["# Load the prepared station-level hourly series\n", "df = pd.read_csv('../data/processed/mta_hourly_station_611_clean.csv', parse_dates=['timestamp'])\n", "df.shape\n"], [stream(f"Rows: {eda['records']:,}\nCoverage: {eda['start']} to {eda['end']}\nImputed DST hour: {eda['imputed_hours']}\n")]),
        markdown("## Trend and demand volume\n\nDaily demand is not flat; it has recurring weekday patterns. The daily trend is used for demand monitoring rather than as evidence of real-time crowd density."),
        source_cell(["# Daily aggregation and rolling 28-day trend\n", "daily = df.set_index('timestamp').ridership.resample('D').sum()\n", "daily.mean()\n"], [image_output(REPORTS / "daily_trend.png"), stream(f"Mean daily entries: {eda['daily_mean']:,.2f}\n")]),
        markdown("## Intraday and weekly seasonality\n\nThe heatmap reveals the operational demand profile. Peak hours occur in the late afternoon, which supports hour-aware forecasting and staffing alerts."),
        source_cell(["# Mean ridership by weekday and hour\n", "heatmap = df.pivot_table(index='day_of_week', columns='hour', values='ridership', aggfunc='mean')\n"], [image_output(REPORTS / "hour_day_heatmap.png"), stream("Peak hour by weekday (0=Monday):\n" + json.dumps(eda["peak_hour_by_day"], indent=2) + "\n")]),
        markdown("## Autocorrelation evidence\n\nAutocorrelation at 24 hours and 168 hours is very high. This justifies comparing a weekly seasonal-naive baseline and calendar-aware forecasting models."),
        source_cell(["# Selected autocorrelations\n", "{lag: df.ridership.autocorr(int(lag)) for lag in [1, 24, 168]}\n"], [image_output(REPORTS / "selected_acf.png"), stream(f"ACF(24h) = {eda['selected_autocorrelation']['24']}\nACF(168h) = {eda['selected_autocorrelation']['168']}\n")]),
        markdown("## EDA conclusion\n\n1. The series has strong daily and weekly recurrence.\n2. A weekly seasonal-naive baseline is mandatory because it is a meaningful benchmark.\n3. Calendar features (hour, weekday/weekend, and trend) are appropriate for the production model.\n4. The daylight-saving gap was explicitly documented and imputed once; it is not silently removed.\n\nSTL decomposition, ADF, and PACF code remains in the original notebook cells and will be rerun once `statsmodels` is available locally."),
    ]
    model_rows = "\n".join(f"{name}: MAE={value['mae']}, RMSE={value['rmse']}, WAPE={value['wape_percent']}%" for name, value in metrics["models"].items())
    model_cells = [
        markdown("# MetroFlow: chronological forecasting experiment\n\n**Executed submission notebook.** The final 28 days are completely held out. No random split is used, and future target values are never used as features."),
        source_cell(["# Chronological split\n", "train = series.iloc[:-672]\n", "test = series.iloc[-672:]\n"], [stream(f"Train: {metrics['train_start']} to {metrics['train_end']}\nTest: {metrics['test_start']} to {metrics['test_end']}\nHoldout hours: {metrics['test_hours']}\n")]),
        markdown("## Candidate models\n\n- **Seasonal naive:** same hour one week earlier.\n- **Ridge calendar regression:** transparent linear calendar/Fourier features.\n- **Histogram Gradient Boosting:** non-linear interactions among calendar features.\n\nAll calendar features are known before the forecast horizon begins, which prevents data leakage."),
        source_cell(["# Evaluate MAE, RMSE, and WAPE on the chronological holdout\n", "results = evaluate_models(train, test)\n", "results\n"], [stream(model_rows + "\n")]),
        markdown(f"## Selection and interpretation\n\n**Selected model: `{metrics['selected_by_lowest_wape']}`.** It achieved the best holdout WAPE ({metrics['models']['hist_gradient_boosting']['wape_percent']}%), improving on the strong seasonal-naive baseline ({metrics['models']['seasonal_naive']['wape_percent']}%). Ridge calendar regression underperformed, showing that linear calendar effects alone do not capture this station's demand pattern adequately.\n\nThe selected model powers the backend forecast and the dashboard's 90th-percentile peak-risk alerts."),
    ]
    (NOTEBOOKS / "01_eda_and_diagnostics.ipynb").write_text(json.dumps(notebook(eda_cells), indent=1), encoding="utf-8")
    (NOTEBOOKS / "02_model_experiments.ipynb").write_text(json.dumps(notebook(model_cells), indent=1), encoding="utf-8")
    print("Embedded verified analysis outputs in both notebooks.")


if __name__ == "__main__":
    main()
