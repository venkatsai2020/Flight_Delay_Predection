import unittest
from datetime import datetime
import sys
import os

# Add the parent directory (ABC) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the module from the Model package
import calcDelay as Delay

class TestDelay(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.scheduled_arrival = ["10:00 AM", "13:30", "08:45 PM"]
        self.actual_arrival = ["10:05 AM", "13:40", "09:00 PM"]
        self.delay_instance = Delay(self.scheduled_arrival, self.actual_arrival)

    def test_time_to_minutes_valid_formats(self):
        self.assertEqual(self.delay_instance.time_to_minutes("10:00 AM"), 600)
        self.assertEqual(self.delay_instance.time_to_minutes("13:30"), 810)
        self.assertEqual(self.delay_instance.time_to_minutes("08:45 PM"), 1245)

    def test_time_to_minutes_invalid_formats(self):
        self.assertIsNone(self.delay_instance.time_to_minutes("10:00"))
        self.assertIsNone(self.delay_instance.time_to_minutes("3:30 PM"))
        self.assertIsNone(self.delay_instance.time_to_minutes("25:00"))

    def test_calculate_delays(self):
        self.delay_instance.calculate_delays()
        expected_delays = [5, 10, 15]
        self.assertEqual(self.delay_instance.get_delays(), expected_delays)

    def test_calculate_delays_invalid_times(self):
        # Test with invalid times
        delay_instance = Delay(["10:00 AM", "invalid", "08:45 PM"], ["10:05 AM", "13:40", "09:00 PM"])
        delay_instance.calculate_delays()
        expected_delays = [5, None, 15]
        self.assertEqual(delay_instance.get_delays(), expected_delays)

if __name__ == '__main__':
    unittest.main()
