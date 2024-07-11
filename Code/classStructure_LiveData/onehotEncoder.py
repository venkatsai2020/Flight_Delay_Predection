import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataEncoder:
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder(sparse_output = False, drop='first', dtype=int, handle_unknown='ignore')
        self.X_train_encoded = None
        self.X_test_encoded = None

    def fit_transform(self, X_train_cleaned, X_test_cleaned):
        self.encoder.fit(X_train_cleaned[self.categorical_features])

        self.X_train_encoded = pd.DataFrame(
            self.encoder.transform(X_train_cleaned[self.categorical_features]),
            columns=self.encoder.get_feature_names_out(self.categorical_features)
        )

        self.X_test_encoded = pd.DataFrame(
            self.encoder.transform(X_test_cleaned[self.categorical_features]),
            columns=self.encoder.get_feature_names_out(self.categorical_features)
        )

    def get_encoded_data(self):
        return self.X_train_encoded, self.X_test_encoded
