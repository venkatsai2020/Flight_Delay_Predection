import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationMatrixPlotter:
    def __init__(self, dataframe):
        self.dataframe = dataframe.select_dtypes(include=[float, int])  # Select only numerical columns

    def plot_heatmap(self, figsize=(10, 8), cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 10}):
        correlation_matrix = self.dataframe.corr()
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt, annot_kws=annot_kws)
        plt.title('Correlation Matrix Heatmap')
        plt.show()
