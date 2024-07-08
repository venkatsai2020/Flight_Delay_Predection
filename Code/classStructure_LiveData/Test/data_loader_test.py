import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import dataLoader as DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary test Excel file
        self.test_file_path = 'test_data.xlsx'
        self.create_test_excel()

    def tearDown(self):
        # Clean up: Delete the temporary test Excel file if it exists
        import os
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def create_test_excel(self):
        # Create a sample DataFrame and save it as Excel for testing
        data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)
        df.to_excel(self.test_file_path, index=False)

    def test_load_excel_data(self):
        # Initialize DataLoader with the test file path
        loader = DataLoader(self.test_file_path)

        # Attempt to load Excel data
        try:
            loaded_data = loader.load_excel_data()
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), 3)  # Assuming the sample data has 3 rows
            # Add more assertions as needed based on your data content
        except Exception as e:
            self.fail(f"Loading Excel data failed: {e}")

    def test_load_excel_data_invalid_file(self):
        # Initialize DataLoader with a non-existent file path
        invalid_loader = DataLoader('invalid_path.xlsx')

        # Expecting an exception to be raised
        with self.assertRaises(Exception):
            invalid_loader.load_excel_data()

if __name__ == '__main__':
    unittest.main()
