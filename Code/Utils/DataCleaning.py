import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score
import numpy as np

class DataCleaning:
    
    def __init__(self) -> None:
        pass

    pd.set_option('display.max_columns', None)

    df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\Files\Programing\Ml_Project_Venkat\Flight_Delay_Predection\Data\Airline_Delay_Cause.csv')
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
    x_train, x_test, y_train, y_test = train_test_split(df_one_hot.drop(columns=['arr_delay']), df_one_hot['arr_delay'], test_size=0.2
                                                        # , random_state=42
                                                        )

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    #Linear Regression
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
    print()
    print("Linear Regression:")
    print("Training RMSE:", train_rmse)
    print("Training R2:", train_r2)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)

    #Lasso regression (L1)
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train)
    y_train_pred_lasso = lasso.predict(x_train)
    y_test_pred_lasso = lasso.predict(x_test)

    r2_train_lasso = r2_score(y_train, y_train_pred_lasso)
    r2_test_lasso = r2_score(y_test, y_test_pred_lasso)

    rmse_train_lasso = np.sqrt(mean_squared_error(y_train, y_train_pred_lasso))
    rmse_test_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))

    print()
    print("Lasso Regression:")
    print(f"Train R2: {r2_train_lasso}")
    print(f"Test R2: {r2_test_lasso}")
    print(f"Train RMSE: {rmse_train_lasso}")
    print(f"Test RMSE: {rmse_test_lasso}")

    #Ridge regression (L2)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    y_train_pred_ridge = ridge.predict(x_train)
    y_test_pred_ridge = ridge.predict(x_test)

    r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
    r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

    rmse_train_ridge = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
    rmse_test_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
    print()
    print("Ridge Regression:")
    print(f"Train R2: {r2_train_ridge}")
    print(f"Test R2: {r2_test_ridge}")
    print(f"Train RMSE: {rmse_train_ridge}")
    print(f"Test RMSE: {rmse_test_ridge}")

    #Random Forest
    rf = RandomForestClassifier(
            n_estimators=100,    # More trees (can be tuned using cross-validation)
            max_depth=6,        # Shallower trees
            min_samples_split=4, # More samples required to split a node
            min_samples_leaf=2,  # More samples required at a leaf node
            bootstrap=True,      # Use bootstrapping
            max_features='sqrt'  # Number of features for best split
        )

    rf.fit(x_train, y_train)

    rf_train_pred = rf.predict(x_train)
    rf_y_pred = rf.predict(x_test)
    train_acc = accuracy_score(y_train, rf_train_pred)
    test_acc = accuracy_score(y_test, rf_y_pred)
    print()
    print("Random Forest:")
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)