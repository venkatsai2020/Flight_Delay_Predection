from data_loader import DataLoader
from data_processor import DataProcessor
from data_splitter import DataSplitter
from missingvalueHandler import MissingValueHandler
from outlierhandler import OutlierHandler
from dataSaver import DataSaver
from correlation_plotter import CorrelationMatrixPlotter
import pandas as pd
from correlation_analyzer import CorrelationAnalyzer
from feature_remover import FeatureRemover

from sklearn.model_selection import train_test_split

from ml_model import MLModel

from sklearn.linear_model import LinearRegression


def main():
    # Load data
    file_path = r'..\..\Data\Airline_Delay_Cause.csv'
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()
    print("\n\n DATA SET:\n\n")
    print("________________________________________________________\n\n")
    print(data)
    print(data.shape)
    data_processor = DataProcessor()

    # print(data_processor.printHead(data))
    # print("\ndata types:",data_processor.printDatatypes(data))
    # print("\nSummary statistics:",data_processor.printSummary(data))
    print("\n\n\n\n\nSPLITTING DATA:\n\n")
    print("________________________________________________________\n\n")
    data_splitter = DataSplitter(data, target_column='arr_delay')
    X_train, X_test, y_train, y_test = data_splitter.split_data()
    data_splitter.print_shapes()
    # handling missing values
    print("\n\n\n\n\nHANDLING MISSING VALUES(or NULL VALUES):\n\n")
    print("________________________________________________________\n\n")
    features_with_null = [
        'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct',
        'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
        'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay',
        'late_aircraft_delay'
    ]
    handler = MissingValueHandler(X_train, y_train, features_with_null)
    handler.print_missing_values()
    # X = data.drop(columns='arr_delay')
    # y = data['arr_delay']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check null values in training data
    handler.check_null_values(X_train, y_train)

    # Remove null values from training data
    X_train_cleaned, y_train_cleaned = handler.remove_null_values(X_train, y_train)

    # Print cleaned missing values
    handler.print_cleaned_missing_values(X_train_cleaned)

    # Print cleaned data shapes
    print("X_train_clean shape:", X_train_cleaned.shape)
    print("y_train_clean shape:", y_train_cleaned.shape)

    print("\n\n\n\n\nHANDLING OUTLIERS:\n\n")
    print("________________________________________________________\n\n")
    # Create an instance of the OutlierHandler class
    outlier_handler = OutlierHandler()
    # Initialize the OutlierHandler
    outlier_handler = OutlierHandler()

    # Display outliers in X_train
    print("Outliers in X_train:")
    outlier_handler.display_outliers(X_train)

    # Remove outliers from X_train
    X_train_cleaned = outlier_handler.remove_outliers(X_train)

    # Get the index of cleaned rows
    cleaned_indices = X_train_cleaned.index

    # Remove corresponding rows from y_train
    y_train_cleaned = y_train.loc[cleaned_indices]

    # Display outliers in y_train
    print("Outliers in y_train:")
    outlier_handler.display_outliers(y_train_cleaned)

    # Check the shape of cleaned datasets
    print("Shape of X_train after removing outliers:", X_train_cleaned.shape)
    print("Shape of y_train after removing outliers:", y_train_cleaned.shape)

    print("\n\n\n\n\nDOWNLOADING THE CLEANED DATA:\n\n")
    print("________________________________________________________\n\n")
    data_saver = DataSaver(X_train_cleaned, y_train_cleaned)

    # Call the save_combined_data_as_excel method to save the combined data as an Excel file
    data_saver.save_combined_data_as_excel()
    print("\n\n\n\n\nCORRELATION HEATMAP:\n\n")
    print("________________________________________________________\n\n")
    df = pd.DataFrame(data)

    # Create an instance of the CorrelationMatrixPlotter
    plotter = CorrelationMatrixPlotter(df)

    # Plot the heatmap
    plotter.plot_heatmap()
    print("\n\n\n\n\nCORRELATION ANALYZER:\n\n")
    print("________________________________________________________\n\n")
    # Define target feature and threshold
    target_feature = 'arr_delay'
    threshold = 0.89
    # Create an instance of the CorrelationAnalyzer
    analyzer = CorrelationAnalyzer(df, target_feature, threshold)

    # Print features with high correlation
    analyzer.print_high_correlation_features()
    print("\n\n\n\n\nFEATURE REMOVER:\n\n")
    print("________________________________________________________\n\n")
    features_to_remove_high = ['arr_del15', 'carrier_ct', 'late_aircraft_delay']
    features_to_remove_low = ['weather_delay', 'nas_delay']

    # Create an instance of the FeatureRemover
    remover = FeatureRemover(X_train_cleaned)

    # Remove high and low correlation features
    X_train_modified = remover.remove_high_and_low_correlation_features(features_to_remove_high, features_to_remove_low)

    # Print the modified DataFrame
    print("Modified DataFrame:")
    print(X_train_modified)
    print(X_train_modified.shape)


if __name__ == "__main__":
    main()
