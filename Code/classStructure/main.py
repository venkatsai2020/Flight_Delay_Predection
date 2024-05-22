from data_loader import DataLoader
from data_processor import DataProcessor
from data_splitter import DataSplitter
from missingvalueHandler import MissingValueHandler
from outlierhandler import OutlierHandler

from ml_model import MLModel

from sklearn.linear_model import LinearRegression


def main():
    # Load data
    file_path = r'..\..\Data\Airline_Delay_Cause.csv'
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()
    print(data)
    data_processor = DataProcessor()

   # print(data_processor.printHead(data))
   #print("\ndata types:",data_processor.printDatatypes(data))
   # print("\nSummary statistics:",data_processor.printSummary(data))

    data_splitter = DataSplitter(data, target_column='arr_delay')
    data_splitter.split_data()
    data_splitter.print_shapes()
    # handling missing values
    missing_value_handler = MissingValueHandler(data_splitter.X_train, data_splitter.y_train)
    missing_value_handler.print_missing_values()
    X_train_clean, y_train_clean = missing_value_handler.get_cleaned_data()
    # Print cleaned data shapes
    print("X_train_clean shape:", X_train_clean.shape)
    print("y_train_clean shape:", y_train_clean.shape)
#detecting and removing outliers
    # Instantiate the OutlierHandler
    outlier_handler = OutlierHandler()

    # Display outliers in X_train
    print("Outliers in X_train:")
    outlier_handler.display_outliers(X_train_clean)

    # Display outliers in y_train
    print("\nOutliers in y_train:")
    outlier_handler.display_outliers(X_train_clean)

    # Remove outliers from X_train
    X_train_cleaned = outlier_handler.remove_outliers(X_train_clean)

    # Get the index of cleaned rows
    cleaned_indices = X_train_cleaned.index

    # Remove corresponding rows from y_train
    y_train_cleaned = X_train_clean.loc[cleaned_indices]

    # Check the shape of cleaned datasets
    print("\nShape of X_train after removing outliers:", X_train_cleaned.shape)
    print("Shape of y_train after removing outliers:", y_train_cleaned.shape)
   # data_processor.plotHist(data)
    #data_processor.plotBox(data)
    #data_processor.chartBar(data)
    #data_processor.plotHeatmap(data)



    ''' processor = DataProcessor()
    X, y = processor.preprocess(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = processor.split_data(X, y)

    # Initialize model
    model = LinearRegression()  # Example model, replace with your choice

    # Initialize ML model wrapper
    ml_model = MLModel(model)

    # Train the model
    ml_model.train(X_train, y_train)

    # Evaluate the model
    mse = ml_model.evaluate(X_test, y_test)
    print("Mean Squared Error:", mse)'''


if __name__ == "__main__":
    main()
