import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nullvalueHandler import nullvalueHandler as NullValueHandler

class TestNullValueHandler(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        data = {
            'Feature1': ['100', '200', 'N/A', '300'],
            'Feature2': ['ABC', '500', '600', np.nan],
            'Target': [1, 2, np.nan, 4]
        }
        self.df = pd.DataFrame(data)
        self.features_with_null = ['Feature1', 'Feature2']

    def test_clean_and_convert(self):
        handler = NullValueHandler()

        # Call clean_and_convert method
        cleaned_data = handler.clean_and_convert(self.df.copy(), self.features_with_null)

        # Assertions
        self.assertFalse(cleaned_data['Feature1'].isnull().any())  # Ensure no nulls in Feature1
        self.assertFalse(cleaned_data['Feature2'].isnull().any())  # Ensure no nulls in Feature2

        # Check conversion correctness
        self.assertAlmostEqual(cleaned_data.at[0, 'Feature1'], 100.0)
        self.assertAlmostEqual(cleaned_data.at[1, 'Feature2'], 500.0)

    def test_clean_target(self):
        handler = NullValueHandler()

        # Create sample target series
        target = pd.Series([1, 2, np.nan, 4])

        # Call clean_target method
        cleaned_target = handler.clean_target(target)

        # Assertions
        self.assertFalse(cleaned_target.isnull().any())  # Ensure no nulls in cleaned target
        self.assertEqual(cleaned_target.median(), 2.0)  # Ensure median value is correctly calculated

    # Add more test cases as needed for edge cases, additional features, etc.

if __name__ == '__main__':
    unittest.main()
