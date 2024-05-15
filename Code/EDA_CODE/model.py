

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class MyModel:
    def __init__(self):
        self.model = "call_our_modelhere"

    def train(self, X_train, y_train):
        print("train the model")

    def predict(self, X_test):
        print("function to predict" )

    def evaluate(self, y_test, predictions):
        print("function to evaluate the metrices")