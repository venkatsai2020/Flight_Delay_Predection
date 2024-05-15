# my_ml_project/main.py

from dataLoader import DataLoader
from model import MyModel
from helperFunctions import print_evaluation_metrics
import config
def main():
    # Load and preprocess data
    data_loader = DataLoader(config.DATA_PATH)
    data = data_loader.load_data()
    data = DataLoader.preprocess_data(data)
    #X_train, X_test, y_train, y_test = DataLoader.split_data(data)
    X_train=123
    y_train=567
    X_test=987
    y_test=765


    # Initialize and train model
    model = MyModel()
    model.train(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mse = model.evaluate(y_test, predictions)

    # Print evaluation metrics
    print_evaluation_metrics()


if __name__ == '__main__':
    main()
