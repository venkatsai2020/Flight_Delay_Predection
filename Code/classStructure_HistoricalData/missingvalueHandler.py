import pandas as pd
import numpy as np
class MissingValueHandler:
    def __init__(self, x,y, features_with_null):
        self.x = x
        self.y=y
        self.features_with_null = features_with_null

    def print_missing_values(self):
        print("\nMissing values in X_TRAIN:")
        print(self.x.isnull().sum())
        print("\nMissing values in Y_TRAIN:")

        print(self.y.isnull().sum())
    def check_null_values(self, X, y):
        null_sum_X = np.sum(X.isnull().values)
        null_sum_y = np.sum(np.isnan(y))
        print("Sum of null values in X:", null_sum_X)
        print("Sum of null values in y:", null_sum_y)
        print(X.shape)
        return null_sum_X, null_sum_y

    def remove_null_values(self, X, y):
        # Remove null values from X
        X_cleaned = X.dropna()

        # Remove corresponding rows from y
        y_cleaned = y[X_cleaned.index]

        # Reset index for y if needed
        y_cleaned = y_cleaned.reset_index(drop=True)

        print("Null values removed from X:", X_cleaned.isnull().sum().sum())
        print("Null values removed from y:", y_cleaned.isnull().sum().sum())
        return X_cleaned, y_cleaned

    def print_cleaned_missing_values(self, X):
        print("\nMissing values:")
        print(X.isnull().sum())

