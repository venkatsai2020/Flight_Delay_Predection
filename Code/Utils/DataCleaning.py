import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class DataCleaning:
    
    def __init__(self) -> None:
        pass

    pd.set_option('display.max_columns', None)

    df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\Programing\Ml_Project_Venkat\Flight_Delay_Predection\Data\Airline_Delay_Cause.csv')
    print(df.shape)
    # print(df.info(verbose=True))
    # print(df.describe(include='all'))
    df_one_hot = pd.get_dummies(df, prefix=df.select_dtypes(include=['object']).columns, dtype=float)
    print(df_one_hot.shape)
    # print(df_one_hot.info(verbose=True))
    # print(df_one_hot.describe(include='all'))
    # plt.figure("Correalation HeatMap")
    # corr = df.corr(numeric_only=True, method="spearman").round(2)
    # sb.heatmap(corr, annot=True)
    # plt.show()
    print(df_one_hot.isnull().sum())
    print(df_one_hot.columns[df_one_hot.isnull().sum() > 1])
    print(df_one_hot.fillna(0, inplace=True))
    x_train, x_test, y_train, y_test = train_test_split(df_one_hot.drop(columns=['arr_delay']), df_one_hot['arr_delay'], test_size=0.2, random_state=42)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    model = LinearRegression()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    # Calculate RMSE and R2 scores
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Print the scores
    print("Training RMSE:", train_rmse)
    print("Training R2:", train_r2)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)