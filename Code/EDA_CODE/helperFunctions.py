from sklearn.ensemble import RandomForestClassifier
from flight_data import FlightData  # Import from flight_data.py
from dataLoader import DataLoader
from model import MyModel
import config
class helperFunctions:
  """
  Handles loading, preprocessing, training, and prediction tasks.
  """
  def __init__(self):
    self.data = []
    self.model = None

  def load_data(self, filename):
    """
    Loads flight data from a CSV file and creates FlightData objects.
    """
    data_loader = DataLoader(config.DATA_PATH)
    data = data_loader.load_data()
    self.data.append(data)

  def preprocess_data(self):
    """
    Performs data cleaning and feature engineering (replace with your specific logic).
    """
    # Example: Convert categorical features to numerical using techniques like one-hot encoding
    # Handle missing values (e.g., impute missing arrival times)
    pass

  def train_model(self, model_type):
    """
    Trains a specified machine learning model on the preprocessed data.
    """

    # Preprocess data before training (call self.preprocess_data() if needed)
    X = [data.features for data in self.data]  # Assuming features are in a dictionary
    y = [data.delay for data in self.data]  # Target variable: delay

    # Train the model
    self.model = model_type()  # Replace with your model instance (e.g., RandomForestClassifier())
    self.model.fit(X, y)

  def predict_delay(self, new_flight):
    """
    Takes a new FlightData object and predicts its delay using the trained model.
    """
    if self.model is None:
      raise Exception("Model not trained yet. Please train the model first.")

    # Extract features from the new flight (assuming features are in a dictionary)
    new_features = new_flight.features
    prediction = self.model.predict([new_features])[0]
    return prediction

  def evaluate_model(self, test_data):
    """
    Evaluates the model performance on the provided test data.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    # Make predictions on test data
    predictions = [self.predict_delay(data) for data in test_data]
    actual_delays = [data.delay for data in test_data]

    # Calculate evaluation metrics (replace with desired metrics)
    accuracy = accuracy_score(actual_delays, predictions)
    precision = precision_score(actual_delays, predictions)
    recall = recall_score(actual_delays, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
