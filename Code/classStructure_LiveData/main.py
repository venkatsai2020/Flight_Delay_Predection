import alert

from data_loader import dataLoader
from calcDelay import Delay
from dataSplitter import dataSplitter
from nullvalueHandler import nullvalueHandler
import numpy as np
from outlierHandler import outlierHandler
from  columnDropper import columnDropper
from datetimeProcessor import datetimeProcessor
from numCorrMatrix import numericalMatrix
from DataSaver import DataSaver
from onehotEncoder import DataEncoder
from FeatureNormalizer import  FeatureNormalizer
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
    median_value=nullHandler.median_value

    # Clean X_train based on features_with_null
    X_train_cleaned = nullHandler.clean_and_convert(X_train, features_with_null)

    y_train_cleaned = nullHandler.clean_target(y_train)
    # Print cleaned DataFrame
    print("X_train after filling null values with median:")
    print(X_train_cleaned)
    print("Y_train after filling null values with median:")
    print(y_train_cleaned)
    print("Sum of null values in X_train after handling:", np.sum(X_train_cleaned.isnull().values))
    print("Sum of null values in y_train after handling:", np.sum(y_train_cleaned.isnull().values))
    print("\n\n\n Outlier Handling...\n\n\n")
    print("\n\n\n_____________________________________\n\n\n")
    X_train_cleaned.reset_index(drop=True, inplace=True)
    y_train_cleaned.reset_index(drop=True, inplace=True)

    # Create an instance of OutlierHandler
    outlier_handler = outlierHandler(X_train_cleaned)

    # Display outliers in X_train
    outlier_handler.display_outliers()

    # Display outliers in y_train
    outlier_handler.display_outliers_y(y_train_cleaned)

    # Remove outliers from X_train
    X_train_outliersRemoved = outlier_handler.remove_outliers()

    # Get the index of cleaned rows
    cleaned_indices = X_train_outliersRemoved.index


    # Remove corresponding rows from y_train
    y_train_outliersRemoved = y_train_cleaned.iloc[cleaned_indices]
    print("Shape of X_train:", X_train_outliersRemoved.shape)
    print("Shape of y_train:", y_train_outliersRemoved.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    print("\n\n column Dropping \n\n")
    print("\n\n\n_____________________________________\n\n\n")

    columns_to_drop = ['FLIGHT', 'AIRCRAFT', 'STATUS', 'ARRIVAL_AIRPORT_NAME', 'ARRIVAL_AIRPORT_CODE']
    cleaner = columnDropper(columns_to_drop)

    # Clean the data
    X_train_cleaned1= cleaner.drop_columns(X_train_cleaned)
    X_test1 = cleaner.drop_columns(X_test)

    print(X_train_cleaned1)
    print(X_test1)
    print("\n\nDate Time processor\n\n")
    print("\n\n\n_____________________________________\n\n\n")
    columns_to_drop = ['TIME', 'DATE']

    # Process the dataframes
    dateProcessor = datetimeProcessor()
    X_train_cleaned2 = dateProcessor.process_dataframe(X_train_cleaned1, time_column='TIME', date_column='DATE',
                                                  columns_to_drop=columns_to_drop)
    X_test_cleaned2 = dateProcessor.process_dataframe(X_test1, time_column='TIME', date_column='DATE',
                                                 columns_to_drop=columns_to_drop)
    print("\n\nX_train_cleaned2\n\n")
    print(X_train_cleaned2)
    print("\n\nX_test_cleaned2\n\n")
    print(X_test_cleaned2)

    print("\n\n Correlation Matrix with numerical values only\n\n")
    print("\n\n________________________________________________\n\n")
    numCorr = numericalMatrix(X_train_cleaned2, X_test_cleaned2)
    numCorr.find_numerical_features()
    numCorr.plot_correlation_matrix()
    high_corr_pairs = numCorr.find_highly_correlated_pairs()
    if high_corr_pairs:
        correlated_features = [pair[0] for pair in high_corr_pairs]
        dataset= numCorr.drop_highly_correlated_features(correlated_features)
        X_train_cleaned2,X_test_cleaned2 =dataset
    print("\n\nDownload the cleaned dataset\n\n")
    data_saver = DataSaver(X_train_cleaned2, y_train, X_test_cleaned2, y_test)

    # Save the data to an Excel file
    data_saver.save_to_excel()

    print("\n\n One-hot Encoder\n\n")
    print("\n\n___________________\n\n")
    categorical_features = ['FROM', 'AIRLINE']
    data_encoder = DataEncoder(categorical_features)

    # Fit and transform the data
    data_encoder.fit_transform(X_train_cleaned2, X_test_cleaned2)

    # Get the encoded data
    X_train_encoded, X_test_encoded = data_encoder.get_encoded_data()

    # You can now use X_train_encoded and X_test_encoded as needed
    print(X_train_encoded.head())
    print(X_test_encoded.head())
    print("Shape of X_train:", X_train_encoded.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test_encoded.shape)
    print("Shape of y_test:", y_test.shape)
    #X_test_encoded.isnull().sum()
    print("median_value:",median_value)
    X_test_filled=X_test_encoded.fillna(median_value)
    y_test.isnull().sum()
    y_test_filled = y_test.fillna(median_value)
    y_test = y_test_filled
    X_test = X_test_filled

    print("\n\n Feature Normalizer\n\n")
    print("\n\n____________________\n\n")
    numerical_features = numCorr.numerical_features
    normalizer = FeatureNormalizer(numerical_features)

    # Normalize the features
    X_train_encoded_normalized = normalizer.normalize_features(X_train_encoded)

    print("\n\n Normalized dataFrame:")
    print(X_train_encoded_normalized)



if __name__ == "__main__":
    main()
