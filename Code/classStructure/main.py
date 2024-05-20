from data_loader import DataLoader
from data_processor import DataProcessor
from ml_model import MLModel

from sklearn.linear_model import LinearRegression


def main():
    # Load data
    file_path = r'..\..\Data\Airline_Delay_Cause.csv'
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()
    print(data)

    data_processor = DataProcessor()

    print(data_processor.printHead(data))
    print("\ndata types:",data_processor.printDatatypes(data))
    print("\nSummary statistics:",data_processor.printSummary(data))
    print("\nMissing values:",data_processor.printMissingvalues(data))
    data_processor.plotHist(data)
    data_processor.plotBox(data)
    data_processor.chartBar(data)
    data_processor.plotHeatmap(data)


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
