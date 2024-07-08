import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataSplitter import dataSplitter as DataSplitter

class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Delay': [0, 1, 0, 1, 1]  # Example target column
        }
        self.df = pd.DataFrame(data)

    def test_split_data(self):
        # Initialize DataSplitter with the sample DataFrame
        splitter = DataSplitter(self.df)

        # Call split_data method
        X_train, X_test, y_train, y_test = splitter.split_data(target_column='Delay', test_size=0.2, random_state=42)

        # Assertions
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

        self.assertEqual(len(X_train) + len(X_test), len(self.df))  # Ensure all rows are accounted for
        self.assertEqual(len(y_train) + len(y_test), len(self.df))  # Ensure all target values are accounted for

        # Add more assertions based on your specific requirements

    def test_split_data_invalid_target_column(self):
        # Initialize DataSplitter with the sample DataFrame
        splitter = DataSplitter(self.df)

        # Call split_data with an invalid target column name
        with self.assertRaises(KeyError):
            splitter.split_data(target_column='InvalidColumn', test_size=0.2, random_state=42)

    def test_split_data_invalid_test_size(self):
        # Initialize DataSplitter with the sample DataFrame
        splitter = DataSplitter(self.df)

        # Call split_data with an invalid test_size
        with self.assertRaises(ValueError):
            splitter.split_data(target_column='Delay', test_size=1.5, random_state=42)

    # Add more test cases as needed for edge cases, random_state variation, etc.

if __name__ == '__main__':
    unittest.main()
