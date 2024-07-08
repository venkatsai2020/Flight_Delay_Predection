import pandas as pd
import numpy as np

class nullvalueHandler:
    def __init__(self):
        pass

    def clean_and_convert(self, X_train, features_with_null):
        """
        Cleans and converts specified features in X_train by:
        1. Extracting numeric values from each feature, converting them to float.
        2. Filling null values in each feature with its median value.

        Args:
        - X_train (DataFrame): The training data containing features to clean and convert.
        - features_with_null (list): List of feature names in X_train that contain null values.

        Returns:
        - X_train (DataFrame): The cleaned and converted training data with null values filled.

        """
        for feature in features_with_null:
            # Remove non-numeric characters and convert to numeric
            #X_train[feature] = X_train[feature].astype(str).str.extract( '(\d+)',expand=False).astype(float)
            X_train[feature] = X_train[feature].astype(str).str.extract(r'(\d+)', expand=False).astype(float)

            # Calculate median for the column
            median_value = X_train[feature].median()

            # Fill null values with median
            #X_train[feature].fillna(median_value, inplace=True)
            X_train[feature] = X_train[feature].fillna(median_value)

        return X_train

    def clean_target(self, y_train):
        median_value = y_train.median()
        print("y_train median_value:",median_value)
        y_train_cleaned = y_train.fillna(median_value)
        return y_train_cleaned
