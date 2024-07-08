import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model import Model

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup classification dataset
        cls.X_classification, cls.y_classification = make_classification(n_samples=100, n_features=20, random_state=42)
        cls.X_train_class, cls.X_test_class, cls.y_train_class, cls.y_test_class = train_test_split(cls.X_classification, cls.y_classification, test_size=0.2, random_state=42)

        # Setup regression dataset
        cls.X_regression, cls.y_regression = make_regression(n_samples=100, n_features=20, random_state=42)
        cls.X_train_reg, cls.X_test_reg, cls.y_train_reg, cls.y_test_reg = train_test_split(cls.X_regression, cls.y_regression, test_size=0.2, random_state=42)
        
        # Initialize Model class
        cls.model = Model()

    def test_train_linear_regression(self):
        model = self.model.train_linear_regression(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg))
        self.assertIsNotNone(model)

    def test_train_lasso(self):
        model = self.model.train_lasso(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), alpha=0.1)
        self.assertIsNotNone(model)

    def test_train_ridge(self):
        model = self.model.train_ridge(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), alpha=0.1)
        self.assertIsNotNone(model)

    def test_train_decision_tree_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_decision_tree(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)

    # def test_train_decision_tree_regression(self):
    #     model, mae, mse, rmse, r2 = self.model.train_decision_tree(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
    #     self.assertIsNotNone(model)
    #     self.assertGreaterEqual(r2, 0)

    def test_train_random_forest_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_random_forest_classification(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)

    def test_train_random_forest_regression(self):
        model, mae, mse, rmse, r2 = self.model.train_random_forest_classification(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(r2, 0)

    def test_train_xgboost_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_xgboost(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)

    def test_train_xgboost_regression(self):
        model, mae, mse, rmse, r2 = self.model.train_xgboost(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(r2, 0)

    def test_train_gradient_boosting_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_gradient_boosting_classification(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)

    # def test_train_gradient_boosting_regression(self):
    #     model, mae, mse, rmse, r2 = self.model.train_gradient_boosting_classification(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
    #     self.assertIsNotNone(model)
    #     self.assertGreaterEqual(r2, 0)

    def test_train_svm_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_svm_classification(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)

    # def test_train_svm_regression(self):
    #     model, mae, mse, rmse, r2 = self.model.train_svm_classification(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
    #     self.assertIsNotNone(model)
    #     self.assertGreaterEqual(r2, 0)

    def test_evaluate_model_classification(self):
        model, accuracy, precision, confusion_mat, classification_rpt = self.model.train_random_forest_classification(pd.DataFrame(self.X_train_class), pd.Series(self.y_train_class), classification=True)
        evaluation_metrics, y_pred = self.model.evaluate_model(model, pd.DataFrame(self.X_test_class), pd.Series(self.y_test_class), classification=True)
        self.assertIsNotNone(evaluation_metrics)
        self.assertGreaterEqual(evaluation_metrics['accuracy_test'], 0)
        self.assertGreaterEqual(evaluation_metrics['precision_test'], 0)

    def test_evaluate_model_regression(self):
        model, mae, mse, rmse, r2 = self.model.train_random_forest_classification(pd.DataFrame(self.X_train_reg), pd.Series(self.y_train_reg), classification=False)
        evaluation_metrics, y_pred = self.model.evaluate_model(model, pd.DataFrame(self.X_test_reg), pd.Series(self.y_test_reg), classification=False)
        self.assertIsNotNone(evaluation_metrics)
        self.assertGreaterEqual(evaluation_metrics['r2_test'], 0)


    # Edge cases

    def test_train_on_small_dataset(self):
        X_small, y_small = make_classification(n_samples=5, n_features=20, random_state=42)
        model = self.model.train_decision_tree(pd.DataFrame(X_small), pd.Series(y_small), classification=True)
        self.assertIsNotNone(model[0])

    def test_train_on_large_dataset(self):
        X_large, y_large = make_classification(n_samples=100000, n_features=20, random_state=42)
        model = self.model.train_decision_tree(pd.DataFrame(X_large), pd.Series(y_large), classification=True)
        self.assertIsNotNone(model[0])

    def test_train_on_dataset_with_missing_values(self):
        X_missing = self.X_train_class.copy()
        X_missing[0, 0] = np.nan  # introduce missing value
        model = self.model.train_decision_tree(pd.DataFrame(X_missing), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model[0])

    def test_train_on_dataset_with_one_feature(self):
        X_one_feature = self.X_train_class[:, :1]
        model = self.model.train_decision_tree(pd.DataFrame(X_one_feature), pd.Series(self.y_train_class), classification=True)
        self.assertIsNotNone(model[0])

    def test_train_on_dataset_with_one_class(self):
        y_one_class = np.zeros(len(self.y_train_class))  # all samples belong to one class
        model = self.model.train_decision_tree(pd.DataFrame(self.X_train_class), pd.Series(y_one_class), classification=True)
        self.assertIsNotNone(model[0])

    def test_train_on_highly_imbalanced_dataset(self):
        y_imbalanced = np.where(self.y_train_class == 0, 0, 1)  # highly imbalanced classes
        model = self.model.train_decision_tree(pd.DataFrame(self.X_train_class), pd.Series(y_imbalanced), classification=True)
        self.assertIsNotNone(model[0])


    def test_scalability_large_dataset(self):
        # Test with a large dataset
        X_large, y_large = make_classification(n_samples=1000000, n_features=20, random_state=42)
        X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.2, random_state=42)
        
        start_time = time.time()
        model = self.model.train_decision_tree(pd.DataFrame(X_train_large), pd.Series(y_train_large), classification=True)
        end_time = time.time()
        
        print(f"Training time for large dataset: {end_time - start_time} seconds")
        self.assertIsNotNone(model[0])


if __name__ == '__main__':
    unittest.main()
