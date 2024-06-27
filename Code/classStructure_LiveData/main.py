from data_loader import dataLoader
from calcDelay import Delay
from dataSplitter import dataSplitter
from nullvalueHandler import nullvalueHandler
import numpy as np
def main():
    # Load data
    file_path = r'..\..\Data\liveData.xlsx'
    data_loader = dataLoader(file_path)
    data = data_loader.load_excel_data()
    print("\n\n DATA SET:\n\n")
    print("________________________________________________________\n\n")
    print(data)
    print(data.shape)
    print(data.dtypes)
    data.drop(columns=['Unnamed: 5'], inplace=True)
    print("\n\n CALCULATE DELAY FROM DATASET\n\n")
    print("\n\n________________________________\n\n")
    Scheduled_arrival = data['Scheduled_arrival']
    Actual_arrival = data['Actual_arrival']
    calculator = Delay(Scheduled_arrival, Actual_arrival)
    calculator.calculate_delays()

    # Get and print delays
    delays = calculator.get_delays()
    print("Delays (minutes):", delays)
    data['Delay'] = delays
    print("\n\n\n\n\nSPLITTING DATA:\n\n")
    print("________________________________________________________\n\n")
    splitter = dataSplitter(data)
    X_train, X_test, y_train, y_test = splitter.split_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("\n\n\n\n\nHANDLING NULL VALUES:\n\n")
    print("________________________________________________________\n\n")
    print("Null values in the dataset:",data.isnull().sum())
    print("Sum of null values in X_train:", np.sum(X_train.isnull().values))
    print("Sum of null values in y_train:", np.sum(np.isnan(y_train)))
    # List of features with null values
    features_with_null = ['WEATHER_ARRIVAL', 'WIND_ARRIVAL', 'WEATHER_DEPARTURE', 'DIRECTION_ARRIVAL',
                          'WIND_DEPARTURE', 'Scheduled_departures', 'Actual_departures', 'DIRECTION_DEPARTURE',
                          'Scheduled_arrival', 'Actual_arrival']

    # Initialize the DataProcessor instance
    nullHandler = nullvalueHandler()

    # Clean X_train based on features_with_null
    X_train_cleaned = nullHandler.clean_and_convert(X_train, features_with_null)

    y_train_cleaned = nullHandler.clean_target(y_train)
    # Print cleaned DataFrame
    print("X_train after filling null values with median:")
    print(X_train_cleaned)
    print("Y_train after filling null values with median:")
    print(y_train_cleaned)
    print("Sum of null values in X_train after handling:", np.sum(X_train.isnull().values))
    print("Sum of null values in y_train after handling:", np.sum(y_train.isnull().sum()))

if __name__ == "__main__":
    main()
