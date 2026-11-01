"""Standard-library tests for MetroFlow's forecast decision service."""
from __future__ import annotations

import unittest

from metroflow.services.forecast import SUPPORTED_STATION_ID, get_service


class ForecastServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = get_service()

    def test_station_metadata_matches_supported_station(self) -> None:
        station = self.service.station()
        self.assertEqual(station["station_id"], SUPPORTED_STATION_ID)
        self.assertTrue(station["station_name"])

    def test_forecast_has_requested_horizon_and_valid_alert_fields(self) -> None:
        forecast = self.service.forecast(24)
        self.assertEqual(len(forecast), 24)
        for row in forecast:
            self.assertGreaterEqual(row["predicted_ridership"], 0)
            self.assertGreater(row["threshold"], 0)
            self.assertLessEqual(row["p10_ridership"], row["p50_ridership"])
            self.assertLessEqual(row["p50_ridership"], row["p90_ridership"])
            self.assertIsInstance(row["high_crowding_expected"], bool)
            self.assertTrue(row["recommendation"])

    def test_history_is_chronological(self) -> None:
        history = self.service.history(None, None)
        self.assertGreater(len(history), 1000)
        self.assertLess(history[0]["timestamp"], history[-1]["timestamp"])

    def test_insights_include_evidence_for_operational_views(self) -> None:
        insights = self.service.insights()
        self.assertEqual(len(insights["peak_windows"]), 7)
        self.assertGreater(insights["seasonality"]["autocorrelation_24h"], 0.8)
        self.assertGreater(insights["historical_summary"]["p90_hourly_entries"], 0)

    def test_event_scenario_uses_declared_assumptions_and_produces_staff_plan(self) -> None:
        event_start = self.service.forecast(24)[-1]["timestamp"]
        scenario = self.service.simulate_event("Test event", event_start, 10_000, 0.3, 0.2, 900, 6)
        self.assertEqual(scenario["expected_station_entries"], 600.0)
        self.assertTrue(scenario["impacted_hours"])
        self.assertIn("staff_required", scenario["impacted_hours"][-1])


if __name__ == "__main__":
    unittest.main()
