import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class numericalMatrix:
    def __init__(self, X_train_cleaned, X_test, threshold=0.8):
        self.X_train_cleaned = X_train_cleaned
        self.X_test = X_test
        self.threshold = threshold
        self.numerical_features = []

    def find_numerical_features(self):
        for column in self.X_train_cleaned.columns:
            if pd.api.types.is_numeric_dtype(self.X_train_cleaned[column]):
                self.numerical_features.append(column)

    def plot_correlation_matrix(self):
        numerical_df = self.X_train_cleaned[self.numerical_features]
        correlation_matrix = numerical_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

    def find_highly_correlated_pairs(self):
        high_corr_pairs = []
        correlation_matrix = self.X_train_cleaned[self.numerical_features].corr()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > self.threshold:
                    colname_i = correlation_matrix.columns[i]
                    colname_j = correlation_matrix.columns[j]
                    high_corr_pairs.append((colname_i, colname_j, correlation_matrix.iloc[i, j]))

        for pair in high_corr_pairs:
            print(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.2f}")

        return high_corr_pairs

    def drop_highly_correlated_features(self, feature_list):
        self.X_train_cleaned.drop(columns=feature_list, inplace=True)
        self.X_test.drop(columns=feature_list, inplace=True)
        return self.X_train_cleaned, self.X_test
