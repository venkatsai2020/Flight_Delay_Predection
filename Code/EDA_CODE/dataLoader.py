# my_ml_project/data/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data


    def preprocess_data(self):
        print("Implement preprocessing steps (e.g., handling missing values, encoding)")
        return 1

    def split_data(self):
       print("code for splitting data into train and test")
       return 1
