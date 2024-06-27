import pandas as pd
from sklearn.model_selection import train_test_split

class dataSplitter:
    def __init__(self, df):
        self.df = df

    def split_data(self, target_column='Delay', test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
