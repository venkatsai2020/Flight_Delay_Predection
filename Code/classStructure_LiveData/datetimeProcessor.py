import pandas as pd

class datetimeProcessor:
    def __init__(self):
        pass

    def process_time(self, df, time_column='TIME'):
        df[time_column] = pd.to_datetime(df[time_column], format='%I:%M %p')
        df['Hour_of_Departure'] = df[time_column].dt.hour
        return df

    def process_date(self, df, date_column='DATE'):
        df[date_column] = pd.to_datetime(df[date_column], format='%A, %b %d')
        df['Day_of_Week'] = df[date_column].dt.dayofweek  # Monday=0, Sunday=6
        df['Month'] = df[date_column].dt.month
        return df

    def drop_columns(self, df, columns_to_drop):
        df.drop(columns=columns_to_drop, inplace=True)
        return df

    def process_dataframe(self, df, time_column='TIME', date_column='DATE', columns_to_drop=None):
        df = self.process_time(df, time_column)
        df = self.process_date(df, date_column)
        if columns_to_drop is not None:
            df = self.drop_columns(df, columns_to_drop)
        return df
