import pandas as pd
import numpy as np

class nullvalueHandler:
    def __init__(self):
        pass

    def clean_and_convert(self, X_train, features_with_null):
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
        y_train = y_train.fillna(median_value)
        return y_train
