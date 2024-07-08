from data_loader import dataLoader
from calcDelay import Delay
from dataSplitter import dataSplitter
from nullvalueHandler import nullvalueHandler
from Model import Model
from Pickle import Pickle
from Joblib import Joblib
import numpy as np
from outlierHandler import outlierHandler
from DataCleaning import DataCleaning


def main():
    # ========================================================================================
    # ################################## Data Cleaning #######################################
    # ========================================================================================
    dc = DataCleaning()

    x_train, y_train, x_test, y_test = dc.dataCleaning('..//..//Data//Final_data.xlsx')
    

    # ========================================================================================
    # #################################### Modeling ##########################################
    # ========================================================================================
    model = Model()

    #===================================== Pickle Models =====================================
    pickle = Pickle()
    #Linear Regression
    try :
        lr = pickle.load_model_from_pickle_folder('LinearRegression')
        lr, mae, mse, rmse, r2 = model.train_predefined_model(lr, x_train, y_train)
        pickle.save_model_to_pickle_folder(lr, 'LinearRegression')
    except Exception as e:
        lr = model.train_linear_regression(x_train, y_train)
        pickle.save_model_to_pickle_folder(lr, 'LinearRegression')

    #Lasso Regularization
    try :
        lar = pickle.load_model_from_pickle_folder('LassoRegularization')
        lar, mae, mse, rmse, r2 = model.train_predefined_model(lar, x_train, y_train)
        pickle.save_model_to_pickle_folder(lar, 'LassoRegularization')
    except Exception as e:
        lar = model.train_lasso(x_train, y_train, 0.1)
        pickle.save_model_to_pickle_folder(lar, 'LassoRegularization')

    #Ridge Regularization
    try :
        rr = pickle.load_model_from_pickle_folder('RidgeRegularization')
        rr, mae, mse, rmse, r2 = model.train_predefined_model(rr, x_train, y_train)
        pickle.save_model_to_pickle_folder(rr, 'RidgeRegularization')
    except Exception as e:
        rr = model.train_ridge(x_train, y_train, 0.1)
        pickle.save_model_to_pickle_folder(rr, 'RidgeRegularization')

    #Decision Tree
    try :
        dt = pickle.load_model_from_pickle_folder('DecisionTree')
        dt, mae, mse, rmse, r2 = model.train_predefined_model(dt, x_train, y_train)
        pickle.save_model_to_pickle_folder(dt, 'DecisionTree')
    except Exception as e:
        dt, mae, mse, rmse, r2 = model.train_decision_tree(x_train, y_train)
        pickle.save_model_to_pickle_folder(dt, 'DecisionTree')
    
    #RandomForest
    try :
        rf = pickle.load_model_from_pickle_folder('RandomForest')
        rf, mae, mse, rmse, r2 = model.train_predefined_model(rf, x_train, y_train)
        pickle.save_model_to_pickle_folder(rf, 'RandomForest')
    except Exception as e:
        rf, mae, mse, rmse, r2 = model.train_random_forest_classification(x_train, y_train)
        pickle.save_model_to_pickle_folder(rf, 'RandomForest')

    #Xgboot
    try :
        xgb = pickle.load_model_from_pickle_folder('Xgboot')
        xgb, mae, mse, rmse, r2 = model.train_predefined_model(xgb, x_train, y_train)
        pickle.save_model_to_pickle_folder(xgb, 'Xgboot')
    except Exception as e:
        xgb, mae, mse, rmse, r2 = model.train_xgboost(x_train, y_train)
        pickle.save_model_to_pickle_folder(xgb, 'Xgboot')

    # ================================= Joblib Models ========================================
    joblib = Joblib()
    try :
        lr = joblib.load_model_from_joblib_folder('LinearRegression')
        lr, mae, mse, rmse, r2 = model.train_predefined_model(lr, x_train, y_train)
        joblib.save_model_to_joblib_folder(lr, 'LinearRegression')
    except Exception as e:
        lr = model.train_linear_regression(x_train, y_train)
        joblib.save_model_to_joblib_folder(lr, 'LinearRegression')

    #Lasso Regularization
    try :
        lar = joblib.load_model_from_joblib_folder('LassoRegularization')
        lar, mae, mse, rmse, r2 = model.train_predefined_model(lar, x_train, y_train)
        joblib.save_model_to_joblib_folder(lar, 'LassoRegularization')
    except Exception as e:
        lar = model.train_lasso(x_train, y_train, 0.1)
        joblib.save_model_to_joblib_folder(lar, 'LassoRegularization')

    #Ridge Regularization
    try :
        rr = joblib.load_model_from_joblib_folder('RidgeRegularization')
        rr, mae, mse, rmse, r2 = model.train_predefined_model(rr, x_train, y_train)
        joblib.save_model_to_joblib_folder(rr, 'RidgeRegularization')
    except Exception as e:
        rr = model.train_ridge(x_train, y_train, 0.1)
        joblib.save_model_to_joblib_folder(rr, 'RidgeRegularization')

    #Decision Tree
    try :
        dt = joblib.load_model_from_joblib_folder('DecisionTree')
        dt, mae, mse, rmse, r2 = model.train_predefined_model(dt, x_train, y_train)
        joblib.save_model_to_joblib_folder(dt, 'DecisionTree')
    except Exception as e:
        dt, mae, mse, rmse, r2 = model.train_decision_tree(x_train, y_train)
        joblib.save_model_to_joblib_folder(dt, 'DecisionTree')
    
    #RandomForest
    try :
        rf = joblib.load_model_from_joblib_folder('RandomForest')
        rf, mae, mse, rmse, r2 = model.train_predefined_model(rf, x_train, y_train)
        joblib.save_model_to_joblib_folder(rf, 'RandomForest')
    except Exception as e:
        rf, mae, mse, rmse, r2 = model.train_random_forest_classification(x_train, y_train)
        joblib.save_model_to_joblib_folder(rf, 'RandomForest')


    #Xgboot
    try :
        xgb = joblib.load_model_from_joblib_folder('Xgboot')
        xgb, mae, mse, rmse, r2 = model.train_predefined_model(xgb, x_train, y_train)
        joblib.save_model_to_joblib_folder(xgb, 'Xgboot')
    except Exception as e:
        xgb, mae, mse, rmse, r2 = model.train_xgboost(x_train, y_train)
        joblib.save_model_to_joblib_folder(xgb, 'Xgboot')


if __name__ == "__main__":
    main()