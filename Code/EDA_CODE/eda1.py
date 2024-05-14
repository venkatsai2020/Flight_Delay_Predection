import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'..\..\Data\Airline_Delay_Cause.csv')# reading the file
print(df)#printing the file
print(df.head())#printing the first 5 dataset
print(df.tail())#printing the last set of data
print(df.shape)#(24518, 21)
print(df.info())#dtypes: float64(15), int64(2), object(4)
print(df.isnull().sum())#count of missing values in each column
print(df.isnull().sum()/df.shape[0]*100)#calculating the percentage of missing values
print(df.duplicated().sum())# 0, means no duplicates
for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())
    print("***"*10)

print(df.describe().T)
print(df.describe(include="object").T)
#ploting boxplot
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.title(f'Boxplot of {i}')
    plt.show()
print(df.select_dtypes(include="number").columns)
#scatter plot
for i in [ 'arr_del15', 'carrier_ct', 'weather_ct',
       'nas_ct', 'security_ct', 'late_aircraft_ct', 'arr_cancelled',
       'arr_diverted', 'carrier_delay', 'weather_delay',
       'nas_delay', 'security_delay', 'late_aircraft_delay']:
    sns.scatterplot(data=df,x=i,y='arr_delay')
    plt.title(f'Scatter plot of {i} vs. arr_delay')
    plt.show()
#plotting heatmap
correlation_matrix = df.select_dtypes(include="number").corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()