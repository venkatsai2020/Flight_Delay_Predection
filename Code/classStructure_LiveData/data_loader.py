import pandas as pd

class dataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_excel_data(self):
        try:
            return pd.read_excel(self.file_path)
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
