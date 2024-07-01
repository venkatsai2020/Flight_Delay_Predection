import numpy as np
import pandas as pd


class outlierHandler:
    def __init__(self, data):
        self.data = data

    def display_outliers(self):
        numeric_data = self.data.select_dtypes(include=['int', 'float'])
        for col in numeric_data.columns:
            Q1 = np.percentile(self.data[col], 25)
            Q3 = np.percentile(self.data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)][col]
            if not outliers.empty:
                print(f"Outliers in column '{col}':")
                print(outliers)

    def display_outliers_y(self, y):
        numeric_data = y[y.apply(lambda x: isinstance(x, (int, float)))]
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = y[(y < lower_bound) | (y > upper_bound)]
        if not outliers.empty:
            print("Outliers in y_train:")
            print(outliers)

    def remove_outliers(self):
        numeric_data = self.data.select_dtypes(include=['int', 'float'])
        cleaned_data = self.data.copy()
        for col in numeric_data.columns:
            Q1 = np.percentile(self.data[col], 25)
            Q3 = np.percentile(self.data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        return cleaned_data
