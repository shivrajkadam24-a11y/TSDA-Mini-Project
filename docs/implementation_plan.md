# MetroFlow implementation plan

## 1. Fixed project scope

**Problem:** Station managers need advance warning of unusually high passenger demand to plan staff and passenger-flow interventions.

**Primary user:** A subway station operations planner.

**First forecasting target:** hourly ridership at Times Sq-42 St / Port Authority Bus Terminal, station-complex ID `611`.

**Forecast horizons:** next 24 hours and next 7 days.

**Decision rule:** flag an hour as `high crowding expected` when its forecast exceeds the 90th percentile of comparable historical hours (same station, hour of day, and weekday/weekend category). The alert is demand planning guidance, not a real-time safety certification.

## 2. Dataset and data contract

| Item | Decision |
|---|---|
| Primary source | Official NY Open Data MTA Subway Hourly Ridership API, dataset `5wq4-mkjj` |
| Historical extension | Official archive `wujg-7c2s` (2020-2024) |
| Granularity after preparation | One record per station complex per hour |
| Target | Sum of `ridership` across payment methods and fare classes |
| Minimum modelling window | Most recent 12 complete months after cleaning |
| Optional exogenous features | Hour, day of week, weekend flag, public-holiday flag |

### Required quality checks

- Parse timestamps in a declared timezone and confirm hourly continuity.
- Filter to `transit_mode = subway` and the selected `station_complex_id`.
- Aggregate payment/fare categories exactly once per hour.
- Detect duplicates, missing hours, negative values, and extreme observations.
- Treat outages and special-event spikes as observations to explain, not values to silently delete.
- Create a versioned data-quality summary with row counts and date range.

## 3. ML / TSDA pipeline

### Notebook 1: EDA and diagnostics

Outputs:

- hourly and daily trend charts;
- hour-by-day heatmap;
- weekly seasonality and STL decomposition;
- ADF stationarity test, ACF/PACF plots;
- missingness/outlier summary;
- clear observations that justify the candidate models.

### Notebook 2: model experiment

Use chronological splits only. Reserve the final 28 days for final testing and use expanding-window validation before it.

| Model | Role |
|---|---|
| Seasonal naive | Required transparent baseline: same hour from the previous week |
| Holt-Winters / ETS | Interpretable trend + multiple-seasonal-cycle candidate |
| SARIMAX | Main statistical model with daily/weekly seasonality and calendar regressors |

Evaluate MAE, RMSE, and WAPE. Select the production model based on validation performance, forecast stability, and interpretability. A deep-learning model is explicitly out of scope unless these three models fail for a documented reason.

### Reproducible scripts

- `src/metroflow/data/fetch_mta.py`: fetches aggregated API data.
- `src/metroflow/data/prepare.py`: validates and prepares the modelling table.
- `src/metroflow/models/`: train, evaluate, and serialize models.
- `src/metroflow/services/`: shared forecast and alert logic for the API.

## 4. Application architecture

```text
NY Open Data API -> ingestion/preprocessing -> trained model + thresholds
                                              -> FastAPI backend -> Streamlit dashboard
```

### Backend endpoints

- `GET /health`
- `GET /stations`
- `GET /forecast?station_id=611&horizon_hours=24`
- `GET /history?station_id=611&start=...&end=...`
- `GET /metrics?station_id=611`

### Dashboard views

- station and forecast-horizon selectors;
- key forecast metrics and alert count;
- historical series and seasonal heatmap;
- actual-versus-predicted / model-comparison chart;
- forecast table with high-crowding alerts and operational recommendation;
- methodology, data source, limits, and model metric panel.

## 5. Definition of done

The final submission will include a working frontend and backend, not a notebook alone. It will demonstrate a complete user flow: select a station -> request forecast -> view peak alerts -> interpret staffing/advisory recommendation. It will also include source documentation, model comparison, reproducible setup instructions, report, screenshots, and viva-ready explanation of every pipeline stage.

## 6. Build milestones

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| M1 | Project scaffold and data fetcher | Official hourly data retrieved reproducibly |
| M2 | Clean modelling table + data-quality report | One valid hourly series with documented repairs |
| M3 | EDA notebook | Trends, seasonality, ACF/PACF, decomposition interpreted |
| M4 | Model notebook | Three models compared using chronological evaluation |
| M5 | Backend | API serves saved forecast, history, metrics, and alerts |
| M6 | Frontend | Interactive end-to-end dashboard works locally |
| M7 | Submission package | Report, README, screenshots, tests, and viva notes complete |

## 7. Immediate next actions

1. Download the station-level hourly aggregate for 2025 onward.
2. Validate its timestamp coverage and quality.
3. Create the first EDA notebook from that frozen snapshot.

