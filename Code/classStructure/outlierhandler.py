import numpy as np
import pandas as pd


class OutlierHandler:
    def __init__(self):
        pass

    def display_outliers(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        numeric_data = data.select_dtypes(include=['int', 'float'])
        for col in numeric_data.columns:
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            if not outliers.empty:
                print("Outliers in column '{}':".format(col))
                print(outliers)

    def count_outliers(self, data):
        numeric_data = data.select_dtypes(include=['int', 'float'])
        for col in numeric_data.columns:
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outliers_count = outliers.shape[0]
            print(f"Outliers in column '{col}': {outliers_count}")

    def remove_outliers(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame()

        numeric_data = data.select_dtypes(include=['int', 'float'])
        cleaned_data = data.copy()
        for col in numeric_data.columns:
            Q1 = np.percentile(numeric_data[col], 25)
            Q3 = np.percentile(numeric_data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        return cleaned_data