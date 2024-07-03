from data_loader import dataLoader
from calcDelay import Delay
from dataSplitter import dataSplitter
from nullvalueHandler import nullvalueHandler
from Model import Model
from Pickle import Pickle
from Joblib import Joblib
import numpy as np
from outlierHandler import outlierHandler


def main():
    # Load data
    file_path = '..//..//Data//liveData.xlsx'
    data_loader = dataLoader(file_path)
    data = data_loader.load_excel_data()
    print("\n\n DATA SET:\n\n")
    print("________________________________________________________\n\n")
    print(data)
    print(data.shape)
    print(data.dtypes)
    data.drop(columns=['Unnamed: 5'], inplace=True)
    print("\n\n CALCULATE DELAY FROM DATASET\n\n")
    print("\n\n________________________________\n\n")
    Scheduled_arrival = data['Scheduled_arrival']
    Actual_arrival = data['Actual_arrival']
    calculator = Delay(Scheduled_arrival, Actual_arrival)
    calculator.calculate_delays()

    # Get and print delays
    delays = calculator.get_delays()
    print("Delays (minutes):", delays)
    data['Delay'] = delays
    print("\n\n\n\n\nSPLITTING DATA:\n\n")
    print("________________________________________________________\n\n")
    splitter = dataSplitter(data)
    X_train, X_test, y_train, y_test = splitter.split_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("\n\n\n\n\nHANDLING NULL VALUES:\n\n")
    print("________________________________________________________\n\n")
    print("Null values in the dataset:",data.isnull().sum())
    print("Sum of null values in X_train:", np.sum(X_train.isnull().values))
    print("Sum of null values in y_train:", np.sum(np.isnan(y_train)))
    # List of features with null values
    features_with_null = ['WEATHER_ARRIVAL', 'WIND_ARRIVAL', 'WEATHER_DEPARTURE', 'DIRECTION_ARRIVAL',
                          'WIND_DEPARTURE', 'Scheduled_departures', 'Actual_departures', 'DIRECTION_DEPARTURE',
                          'Scheduled_arrival', 'Actual_arrival']

    # Initialize the DataProcessor instance
    nullHandler = nullvalueHandler()

    # Clean X_train based on features_with_null
    X_train_cleaned = nullHandler.clean_and_convert(X_train, features_with_null)

    y_train_cleaned = nullHandler.clean_target(y_train)
    # Print cleaned DataFrame
    print("X_train after filling null values with median:")
    print(X_train_cleaned)
    print("Y_train after filling null values with median:")
    print(y_train_cleaned)
    print("Sum of null values in X_train after handling:", np.sum(X_train_cleaned.isnull().values))
    print("Sum of null values in y_train after handling:", np.sum(y_train_cleaned.isnull().values))
    print("\n\n\n Outlier Handling...\n\n\n")
    print("\n\n\n_____________________________________\n\n\n")
    X_train_cleaned.reset_index(drop=True, inplace=True)
    y_train_cleaned.reset_index(drop=True, inplace=True)

    # Create an instance of OutlierHandler
    outlier_handler = outlierHandler(X_train_cleaned)

    # Display outliers in X_train
    outlier_handler.display_outliers()

    # Display outliers in y_train
    outlier_handler.display_outliers_y(y_train_cleaned)

    # Remove outliers from X_train
    X_train_outliersRemoved = outlier_handler.remove_outliers()

    # Get the index of cleaned rows
    cleaned_indices = X_train_cleaned.index


    # Remove corresponding rows from y_train
    y_train_outliersRemoved = y_train_cleaned.iloc[cleaned_indices]
    #y_train_cleaned = y_train.iloc[cleaned_indices]

    # Check the shape of cleaned datasets
    print("Shape of X_train after removing outliers:", X_train_outliersRemoved.shape)
    print("Shape of y_train after removing outliers:", y_train_outliersRemoved.shape)

    # ========================================================================================
    # #################################### Modeling ##########################################
    # ========================================================================================

    model = Model()

    X_train_cleaned = X_train_cleaned.select_dtypes(exclude=['object'])

    #===================================== Pickle Models =============================

    pickle = Pickle()
    #Linear Regression
    try :
        lr = pickle.load_model_from_pickle_folder('LinearRegression')
        lr, mae, mse, rmse, r2 = model.train_predefined_model(lr, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(lr, 'LinearRegression')
    except Exception as e:
        lr = model.train_linear_regression(X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(lr, 'LinearRegression')

    #Lasso Regularization
    try :
        lar = pickle.load_model_from_pickle_folder('LassoRegularization')
        lar, mae, mse, rmse, r2 = model.train_predefined_model(lar, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(lar, 'LassoRegularization')
    except Exception as e:
        lar = model.train_lasso(X_train_cleaned, y_train_cleaned, 0.1)
        pickle.save_model_to_pickle_folder(lar, 'LassoRegularization')

    #Ridge Regularization
    try :
        rr = pickle.load_model_from_pickle_folder('RidgeRegularization')
        rr, mae, mse, rmse, r2 = model.train_predefined_model(rr, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(rr, 'RidgeRegularization')
    except Exception as e:
        rr = model.train_ridge(X_train_cleaned, y_train_cleaned, 0.1)
        pickle.save_model_to_pickle_folder(rr, 'RidgeRegularization')

    #Decision Tree
    try :
        dt = pickle.load_model_from_pickle_folder('DecisionTree')
        dt, mae, mse, rmse, r2 = model.train_predefined_model(dt, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(dt, 'DecisionTree')
    except Exception as e:
        dt, mae, mse, rmse, r2 = model.train_decision_tree(X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(dt, 'DecisionTree')
    
    #RandomForest
    try :
        rf = pickle.load_model_from_pickle_folder('RandomForest')
        rf, mae, mse, rmse, r2 = model.train_predefined_model(rf, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(rf, 'RandomForest')
    except Exception as e:
        rf, mae, mse, rmse, r2 = model.train_random_forest_classification(X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(rf, 'RandomForest')

    #Xgboot
    try :
        xgb = pickle.load_model_from_pickle_folder('Xgboot')
        xgb, mae, mse, rmse, r2 = model.train_predefined_model(xgb, X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(xgb, 'Xgboot')
    except Exception as e:
        xgb, mae, mse, rmse, r2 = model.train_xgboost(X_train_cleaned, y_train_cleaned)
        pickle.save_model_to_pickle_folder(xgb, 'Xgboot')

    # =========================== Joblib Models ================================
    joblib = Joblib()
    try :
        lr = joblib.load_model_from_joblib_folder('LinearRegression')
        lr, mae, mse, rmse, r2 = model.train_predefined_model(lr, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(lr, 'LinearRegression')
    except Exception as e:
        lr = model.train_linear_regression(X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(lr, 'LinearRegression')

    #Lasso Regularization
    try :
        lar = joblib.load_model_from_joblib_folder('LassoRegularization')
        lar, mae, mse, rmse, r2 = model.train_predefined_model(lar, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(lar, 'LassoRegularization')
    except Exception as e:
        lar = model.train_lasso(X_train_cleaned, y_train_cleaned, 0.1)
        joblib.save_model_to_joblib_folder(lar, 'LassoRegularization')

    #Ridge Regularization
    try :
        rr = joblib.load_model_from_joblib_folder('RidgeRegularization')
        rr, mae, mse, rmse, r2 = model.train_predefined_model(rr, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(rr, 'RidgeRegularization')
    except Exception as e:
        rr = model.train_ridge(X_train_cleaned, y_train_cleaned, 0.1)
        joblib.save_model_to_joblib_folder(rr, 'RidgeRegularization')

    #Decision Tree
    try :
        dt = joblib.load_model_from_joblib_folder('DecisionTree')
        dt, mae, mse, rmse, r2 = model.train_predefined_model(dt, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(dt, 'DecisionTree')
    except Exception as e:
        dt, mae, mse, rmse, r2 = model.train_decision_tree(X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(dt, 'DecisionTree')
    
    #RandomForest
    try :
        rf = joblib.load_model_from_joblib_folder('RandomForest')
        rf, mae, mse, rmse, r2 = model.train_predefined_model(rf, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(rf, 'RandomForest')
    except Exception as e:
        rf, mae, mse, rmse, r2 = model.train_random_forest_classification(X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(rf, 'RandomForest')


    #Xgboot
    try :
        xgb = joblib.load_model_from_joblib_folder('Xgboot')
        xgb, mae, mse, rmse, r2 = model.train_predefined_model(xgb, X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(xgb, 'Xgboot')
    except Exception as e:
        xgb, mae, mse, rmse, r2 = model.train_xgboost(X_train_cleaned, y_train_cleaned)
        joblib.save_model_to_joblib_folder(xgb, 'Xgboot')


if __name__ == "__main__":
    main()