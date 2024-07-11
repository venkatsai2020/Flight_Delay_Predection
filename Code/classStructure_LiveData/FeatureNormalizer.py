import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureNormalizer:
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        self.scaler = MinMaxScaler()

    def normalize_features(self, X_train_encoded):

        for feature in self.numerical_features:
            # Reshape the data to fit the scaler
            reshaped_feature = X_train_encoded[feature].values.reshape(-1, 1)
            # Apply the scaler and update the DataFrame
            X_train_encoded[feature] = self.scaler.fit_transform(reshaped_feature)

        return X_train_encoded
