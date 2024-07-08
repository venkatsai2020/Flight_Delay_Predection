import unittest
import sys
import shutil
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Pickle import Pickle

class TestPickle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pickle_instance = Pickle()
        cls.model = {"key": "value"}  # Example model to pickle
        cls.default_path = '../../PickledModels'
        cls.custom_path = './custom_models'
        os.makedirs(cls.default_path, exist_ok=True)
        os.makedirs(cls.custom_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.default_path)
        shutil.rmtree(cls.custom_path)

    def test_save_model_to_pickle_folder(self):
        self.pickle_instance.save_model_to_pickle_folder(self.model, 'test_model')
        file_path = os.path.join(self.default_path, 'test_model.pk1')
        self.assertTrue(os.path.exists(file_path))

    def test_load_model_from_pickle_folder(self):
        self.pickle_instance.save_model_to_pickle_folder(self.model, 'test_model')
        loaded_model = self.pickle_instance.load_model_from_pickle_folder('test_model')
        self.assertEqual(loaded_model, self.model)

    def test_save_model_to_specified_path(self):
        self.pickle_instance.save_model_to_specified_path(self.model, self.custom_path, 'test_model')
        file_path = os.path.join(self.custom_path, 'test_model.pk1')
        self.assertTrue(os.path.exists(file_path))

    def test_load_model_from_specified_path(self):
        self.pickle_instance.save_model_to_specified_path(self.model, self.custom_path, 'test_model')
        loaded_model = self.pickle_instance.load_model_from_specified_path(self.custom_path, 'test_model')
        self.assertEqual(loaded_model, self.model)

    def test_nonexistent_load_model_from_pickle_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.pickle_instance.load_model_from_pickle_folder('nonexistent_model')

    def test_nonexistent_load_model_from_specified_path(self):
        with self.assertRaises(FileNotFoundError):
            self.pickle_instance.load_model_from_specified_path(self.custom_path, 'nonexistent_model')

if __name__ == '__main__':
    unittest.main()
