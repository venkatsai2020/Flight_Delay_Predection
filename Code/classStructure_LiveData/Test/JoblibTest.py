import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Joblib import Joblib

class TestJoblib(unittest.TestCase):

    def setUp(self):
        self.joblib_instance = Joblib()

    def tearDown(self):
        # Clean up any created files after each test
        folder_path = '../../JoblibModels/'
        for filename in os.listdir(folder_path):
            if filename.endswith('.joblib'):
                os.remove(os.path.join(folder_path, filename))

    def test_save_and_load_to_joblib_folder(self):
        model_name = 'test_model'
        model_data = {'param1': 1, 'param2': 'example'}

        # Save model
        self.joblib_instance.save_model_to_joblib_folder(model_data, model_name)

        # Load model
        loaded_model = self.joblib_instance.load_model_from_joblib_folder(model_name)

        # Check if loaded model matches the saved model
        self.assertEqual(loaded_model, model_data)

    def test_save_and_load_to_specified_path(self):
        model_name = 'test_model'
        model_data = {'param1': 1, 'param2': 'example'}
        specified_path = '../../CustomPath/'

        # Save model to specified path
        self.joblib_instance.save_model_to_specified_path(model_data, specified_path, model_name)

        # Load model from specified path
        loaded_model = self.joblib_instance.load_model_from_specified_path(specified_path, model_name)

        # Check if loaded model matches the saved model
        self.assertEqual(loaded_model, model_data)

if __name__ == '__main__':
    unittest.main()
