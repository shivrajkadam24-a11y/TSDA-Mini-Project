# MetroFlow report outline

## 1. Abstract

State the objective: forecast hourly passenger entries at Times Sq-42 St / Port Authority Bus Terminal and convert forecasts into peak-crowding planning alerts.

## 2. Problem statement and users

Explain how advance hourly demand estimates help station operations planners prepare staffing, queue-management measures, and passenger advisories. Clarify that the alert is planning support, not real-time crowd-density measurement.

## 3. Dataset and preparation

Document the official NY Open Data MTA hourly ridership source, station complex ID 611, source attributes, hourly frequency, coverage period, aggregation across payment/fare categories, missing-hour daylight-saving treatment, and data-quality checks.

## 4. EDA and TSDA diagnostics

Include the daily trend, hour-by-day heatmap, selected ACF, and the STL/ADF/PACF results after the statistical dependency is available. Interpret observed daily and weekly seasonality rather than only displaying plots.

## 5. Methodology

Describe the chronological holdout design and the three compared models: seasonal naive, ridge calendar regression, and histogram gradient boosting. Explain why calendar-only features avoid future leakage in multi-step forecasts.

## 6. Evaluation and result interpretation

Report MAE, RMSE, and WAPE on the final 28-day holdout. State that gradient boosting achieved the lowest WAPE (5.82%), compared with seasonal naive (7.29%). Discuss why this improvement is meaningful and why the ridge model performed poorly.

## 7. Application implementation

Show the frontend/backend architecture, API endpoints, dashboard views, 90th-percentile alert rule, and one end-to-end use case.

## 8. Limitations and future work

Discuss estimated ridership rather than real-time occupancy, one initial station complex, no event/weather/service-disruption inputs, and the need to refresh data/model regularly. State that ETS/SARIMAX will be added when the statistical dependency is installed.

## 9. Conclusion

Summarize the operational value and evidence supporting the selected model.
