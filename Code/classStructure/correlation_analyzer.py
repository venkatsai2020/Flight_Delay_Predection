#class to perform feature removal
import pandas as pd


class CorrelationAnalyzer:
    def __init__(self, dataframe, target_feature, threshold=0.89):
       # self.dataframe = dataframe
        self.target_feature = target_feature
        self.threshold = threshold
        self.dataframe = dataframe.select_dtypes(include=[float, int])

    def get_correlation_matrix(self):
        # Create a copy of the correlation matrix to avoid modifying the original data
        correlation_matrix = self.dataframe.corr().copy()

        # Exclude the target feature from the correlation matrix (both rows and columns)
        correlation_matrix = correlation_matrix.drop(self.target_feature, axis=0)
        correlation_matrix = correlation_matrix.drop(self.target_feature, axis=1)

        return correlation_matrix

    def find_high_correlation_features(self):
        correlation_matrix = self.get_correlation_matrix()

        # Find features with correlation above the threshold
        high_correlation_features = []
        for col in correlation_matrix.columns:
            for other_col in correlation_matrix.columns:
                if col != other_col and abs(correlation_matrix.loc[col, other_col]) > self.threshold:
                    high_correlation_features.append(col)

        # Count the occurrences of each feature in the list
        feature_counts = {}
        for feature in high_correlation_features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1

        return feature_counts

    def print_high_correlation_features(self):
        feature_counts = self.find_high_correlation_features()

        # Print the results
        print(f"Features with correlation greater than {self.threshold}, excluding {self.target_feature} (count):")
        for feature, count in feature_counts.items():
            print(f"{feature} : {count}")

