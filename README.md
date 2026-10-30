# MetroFlow

**NYC Subway Station-Complex Footfall Forecasting and Peak-Crowding Alert System**

MetroFlow forecasts hourly subway entries for selected NYC station complexes and turns the forecast into operational alerts for station managers. The first supported station is **Times Sq-42 St / Port Authority Bus Terminal** (station complex ID `611`).

## Project objective

Help transit operations teams anticipate high-demand periods so they can plan staff deployment, queue management, and passenger advisories. This is a time-series decision-support application, not only a forecasting notebook.

## Data source

The project retrieves data programmatically from the official NY Open Data MTA Subway Hourly Ridership API:

- Current dataset: `5wq4-mkjj` (Beginning 2025)
- Historical dataset: `wujg-7c2s` (2020-2024), available for later robustness analysis

The ingestion pipeline aggregates individual fare/payment records into one hourly `ridership` value per station complex. Raw downloaded data remains local and is excluded from Git.

## Development sequence

1. Acquire and validate the official data.
2. Build a reproducible preprocessing pipeline and data-quality report.
3. Perform EDA and time-series diagnostics in a notebook.
4. Train and compare seasonal-naive, Holt-Winters/ETS, and SARIMAX models.
5. Package the selected model as a backend API.
6. Build an interactive dashboard with forecasts, comparisons, and peak-crowding alerts.
7. Test the complete workflow and prepare the report, README, screenshots, and viva material.

The detailed scope and acceptance criteria are in [docs/implementation_plan.md](docs/implementation_plan.md).

