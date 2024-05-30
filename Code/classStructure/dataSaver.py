import pandas as pd


class DataSaver:
    def __init__(self, X_train_cleaned, y_train_cleaned):
        self.X_train_cleaned = X_train_cleaned
        self.y_train_cleaned = y_train_cleaned
        self.output_file = "cleaned_data.xlsx"

    def save_combined_data_as_excel(self):
        combined_data = pd.concat([self.X_train_cleaned, self.y_train_cleaned], axis=1)
        combined_data.to_excel(self.output_file, index=False)
        print(f"Combined data saved to '{self.output_file}'")