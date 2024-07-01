import pandas as pd

class FeatureRemover:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_features(self, features_to_remove):

        return self.dataframe.drop(features_to_remove, axis=1)

    def remove_high_and_low_correlation_features(self, high_corr_features, low_corr_features):

        df_high_removed = self.remove_features(high_corr_features)
        df_final = df_high_removed.drop(low_corr_features, axis=1)
        return df_final
