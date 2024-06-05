import pandas as pd

class FeatureRemover:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_features(self, features_to_remove):
        """
        Remove specified features from the DataFrame.

        Parameters:
        features_to_remove (list): List of feature names to be removed.

        Returns:
        pd.DataFrame: DataFrame with specified features removed.
        """
        return self.dataframe.drop(features_to_remove, axis=1)

    def remove_high_and_low_correlation_features(self, high_corr_features, low_corr_features):
        """
        Remove both high and low correlation features from the DataFrame.

        Parameters:
        high_corr_features (list): List of high correlation feature names to be removed.
        low_corr_features (list): List of low correlation feature names to be removed.

        Returns:
        pd.DataFrame: DataFrame with specified features removed.
        """
        df_high_removed = self.remove_features(high_corr_features)
        df_final = df_high_removed.drop(low_corr_features, axis=1)
        return df_final
