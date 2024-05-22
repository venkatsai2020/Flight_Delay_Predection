import numpy as np
import pandas as pd


class MissingValueHandler:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def print_missing_values(self):
        print("\nMissing values before handling:")
        print(self.X_train.isnull().sum())

    def handle_missing_values(self):
        # Sum of null values in X_train
        null_sum_X_train = self.X_train.isnull().sum().sum()

        # Sum of null values in y_train
        null_sum_y_train = self.y_train.isnull().sum()

        print("Sum of null values in X_train:", null_sum_X_train)
        print("Sum of null values in y_train:", null_sum_y_train)

        # Remove null values from X_train
        self.X_train = self.X_train.dropna()

        # Remove corresponding rows from y_train
        self.y_train = self.y_train[self.X_train.index]

        # Reset index for y_train if needed
        self.y_train = self.y_train.reset_index(drop=True)

        # Check if null values are removed
        print("Null values removed from X_train:", self.X_train.isnull().sum().sum())
        print("Null values removed from y_train:", self.y_train.isnull().sum().sum())
        print("\nMissing values after handling:")
        print(self.X_train.isnull().sum())

    def get_cleaned_data(self):
        return self.X_train, self.y_train
