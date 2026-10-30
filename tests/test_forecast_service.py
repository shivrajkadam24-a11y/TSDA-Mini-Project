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
            self.assertIsInstance(row["high_crowding_expected"], bool)
            self.assertTrue(row["recommendation"])

    def test_history_is_chronological(self) -> None:
        history = self.service.history(None, None)
        self.assertGreater(len(history), 1000)
        self.assertLess(history[0]["timestamp"], history[-1]["timestamp"])


if __name__ == "__main__":
    unittest.main()
