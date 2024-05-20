import pandas as pd
data = pd.read_csv(r'..\..\Data\Airline_Delay_Cause.csv')
print(data)
print(data.dtypes)
print(data.columns)
missing_values = data.isnull().sum()
numerical_summary = data.describe()

mode_values = data.mode().iloc[0] 

for column in data.columns:
    if missing_values[column] > 0:
        print(f"Column: {column}")
        print(f"Missing values count: {missing_values[column]}")
        
        if pd.api.types.is_numeric_dtype(data[column]):
            print(f"Mean: {data[column].mean()}")
            print(f"Median: {data[column].median()}")
        else:
            print(f"Mode: {mode_values[column]}")
        
        print("-" * 20)

