from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def printHead(self, data):
        return data.head()

    def printDatatypes(self, data):
        return data.info()

    def printSummary(self, data):
        return data.describe()

    def printMissingvalues(self, data):
        return data.isnull().sum()
    def plotHist(self,data):
        data.hist()
        print(plt.show())
    def plotBox(self,data):
        data.boxplot()
        print(plt.show())
    def chartBar(self,data):
        data['carrier'].value_counts().plot(kind='bar')
        print(plt.show())
        data['carrier_name'].value_counts().plot(kind='bar')
        print(plt.show())
        data['airport'].value_counts().plot(kind='bar')
        print(plt.show())
        data['airport_name'].value_counts().plot(kind='bar')
        print(plt.show())
    def plotHeatmap(self,data):
        data_encoded = pd.get_dummies(data, columns=['carrier'])
        data_types = data.dtypes

        categorical_columns = data_types[data_types == 'object'].index.tolist()

        if len(categorical_columns) > 0:
            print("Categorical columns found:")
            print(categorical_columns)
        else:
            print("No categorical columns found.")
        correlation_matrix = data.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.show()

        '''

    def preprocess(self, data):
        # Assuming data is a pandas DataFrame
        X = data.drop(columns=['target_column'])  # Adjust target column name
        y = data['target_column']  # Adjust target column name

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def split_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test'''
