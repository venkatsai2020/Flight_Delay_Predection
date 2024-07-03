import pandas as pd

class columnDropper:
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def drop_columns(self, df):
        df.drop(columns=self.columns_to_drop, inplace=True)
        return df
