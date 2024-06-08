import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, df, target_column, test_size=0.2, random_state=42):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        print("the value of X:")
        print(X)
        print("The value of Y:")
        print(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


    def print_shapes(self):
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)
        #print(self.y_train)


