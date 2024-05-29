import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'..\..\Data\Airline_Delay_Cause.csv')

print(data)
print(data.head())
print("First few rows:")
print(data.head())

print("\nData types:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())


# Histogram
data.hist()
print(plt.show())

# Box plot
data.boxplot()
print(plt.show())

# Bar chart
data['carrier'].value_counts().plot(kind='bar')
print(plt.show())
data['carrier_name'].value_counts().plot(kind='bar')
print(plt.show())
data['airport'].value_counts().plot(kind='bar')
print(plt.show())
data['airport_name'].value_counts().plot(kind='bar')
print(plt.show())
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
data['in_hours'] = data['late_aircraft_delay'] /60
print(data['in_hours'])

