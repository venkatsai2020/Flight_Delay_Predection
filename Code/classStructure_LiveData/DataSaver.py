import pandas as pd

class DataSaver:
    def __init__(self, X_train_cleaned, y_train, X_test_cleaned, y_test):
        self.X_train_cleaned = X_train_cleaned
        self.y_train = y_train
        self.X_test_cleaned = X_test_cleaned
        self.y_test = y_test

    def save_to_excel(self, filename='cleaned_data.xlsx'):
        with pd.ExcelWriter(filename) as writer:
            self.X_train_cleaned.to_excel(writer, sheet_name='X_train_cleaned')
            self.y_train.to_excel(writer, sheet_name='y_train_cleaned')
            self.X_test_cleaned.to_excel(writer, sheet_name='X_test_cleaned')
            self.y_test.to_excel(writer, sheet_name='y_test_cleaned')
