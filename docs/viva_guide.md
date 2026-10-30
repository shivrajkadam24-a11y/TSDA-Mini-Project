# MetroFlow viva guide

## One-minute explanation

MetroFlow forecasts hourly entries for Times Sq-42 St / Port Authority Bus Terminal. It uses official MTA hourly ridership data, cleans it into one station-hour series, compares three forecasting approaches using a chronological 28-day test, and gives peak-crowding planning alerts when predicted footfall exceeds the comparable historical 90th percentile.

## Likely questions

**Why is this a time-series project?** The target is ordered hourly ridership with strong daily and weekly dependence. We preserve time order in EDA, validation, and forecasting.

**Why no random train/test split?** Random splitting would leak future behaviour into training and give unrealistic accuracy. The final 28 days are unseen chronologically.

**Why seasonal naive as a baseline?** Reusing the same hour from the previous week is a strong, understandable baseline for transit demand. A useful model must beat it.

**Why did Gradient Boosting win?** It captured non-linear interactions between hour, weekday, weekend status, and gradual trend. Its WAPE was 5.82%, better than the 7.29% seasonal-naive baseline.

**How is an alert generated?** An alert appears when the forecast is at least the 90th percentile of historical demand for the same hour and weekday/weekend category.

**What are the limitations?** The data measures estimated entries, not platform crowd density. The first release supports one station, does not include events/weather/service disruptions, and requires periodic refresh/retraining.

## Demonstration flow

1. Start API and dashboard.
2. Select a 24- or 168-hour horizon.
3. Show the forecast line against its alert threshold.
4. Open an alert row and explain the operational recommendation.
5. Show model metrics and explain the chronological evaluation.
