import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score
import numpy as np
import pickle as pkl


df = pd.read_excel(r'C:\Users\kiran\Downloads\cleaned_data.xlsx')    
print(df.shape)
ls = []
for i in df.columns.values:
    if(str(i).endswith('.1')):
        ls.append(i)

df.drop(labels=ls, axis=1, inplace=True)

df.sort_values(by=['year', 'month'], ascending=True, inplace=True)


x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['arr_del15', 'carrier', 'carrier_name', 'airport', 'airport_name']), df['arr_del15'], test_size=0.3
                                                            # , random_state=42
                                                            )

model = pkl.load(open(r'PickledModels/flight_delay_linear_regression.pk1' , 'rb'))

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
# Calculate RMSE and R2 scores
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)


test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print the scores
print()
print("Linear Regression:")
print("Training RMSE:", train_rmse)
print("Training R2:", train_r2)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)