import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, confusion_matrix, classification_report

class Model:

    def __init__(self) -> None:
        pass


    def train_linear_regression(self, x_train : pd.DataFrame, y_train : pd.Series):
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        r2_train = r2_score(y_train, y_train_pred)
        rms_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        print(f'R-squared (train): {r2_train:.4f}')
        print(f'RMSE (train): {rms_train:.4f}')
        return model


    def train_lasso(self, x_train : pd.DataFrame, y_train : pd.Series, alpha : float):
        model = Lasso(alpha = alpha)
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        r2_train = r2_score(y_train, y_train_pred)
        rms_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        print(f'R-squared (train): {r2_train:.4f}')
        print(f'RMSE (train): {rms_train:.4f}')
        return model


    def train_ridge(self, x_train : pd.DataFrame, y_train : pd.Series, alpha : float):
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        r2_train = r2_score(y_train, y_train_pred)
        rms_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        print(f'R-squared (train): {r2_train:.4f}')
        print(f'RMSE (train): {rms_train:.4f}')
        return model


    def train_decision_tree(self, x_train : pd.DataFrame, y_train : pd.Series,
                                         max_depth = None, min_sample_split: float|int = 2, 
                                         min_samples_leaf: float|int  = 1, max_features = None):
        model = DecisionTreeClassifier(
            max_depth= max_depth,
            min_samples_split= min_sample_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train = precision_score(y_train, y_train_pred)
        confusion_mat = confusion_matrix(y_train, y_train_pred)
        classification_rpt = classification_report(y_train, y_train_pred)


        print(f'Accuracy (train): {accuracy_train:.4f}')
        print(f'Precision (train): {precision_train:.4f}')
        print(f'Classification Report (train):\n{classification_rpt}')
        print(f'Confusion Matrix (train):\n{confusion_mat}')
        return model, accuracy_train, precision_train, confusion_mat, classification_rpt
    


    def train_random_forest_classification(self, x_train : pd.DataFrame, y_train : pd.Series,
                                         n_estimators: int = 100,
                                         max_depth = None, min_sample_split: float|int = 2, 
                                         min_samples_leaf: float|int  = 1, max_features = None):
        model = RandomForestClassifier(
            n_estimators= n_estimators,
            max_depth= max_depth,
            min_samples_split= min_sample_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train = precision_score(y_train, y_train_pred)
        confusion_mat = confusion_matrix(y_train, y_train_pred)
        classification_rpt = classification_report(y_train, y_train_pred)
        
        print(f'Accuracy (train): {accuracy_train:.4f}')
        print(f'Precision (train): {precision_train:.4f}')
        print(f'Classification Report (train):\n{classification_rpt}')
        print(f'Confusion Matrix (train):\n{confusion_mat}')
        return model, accuracy_train, precision_train, confusion_mat, classification_rpt
    
    
    
    def train_predefined_model(self, model : any, x_train : pd.DataFrame, y_train: pd.Series, classification=False):
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_train)
        evaluation_metrics = {}
        
        evaluation_metrics['r2_train'] = r2_score(y_train, y_pred)
        evaluation_metrics['rmse_train'] = np.sqrt(mean_squared_error(y_train, y_pred))

        print(f'R-squared (train): {evaluation_metrics['r2_train']:.4f}')
        print(f'RMSE (train): {evaluation_metrics['rmse_train']:.4f}')
        
        if classification:
            accuracy_train = accuracy_score(y_train, y_pred)
            precision_train = precision_score(y_train, y_pred)
            confusion_mat = confusion_matrix(y_train, y_pred)
            classification_rpt = classification_report(y_train, y_pred)
            
            evaluation_metrics['accuracy_train'] = accuracy_train
            evaluation_metrics['precision_train'] = precision_train
            evaluation_metrics['confusion_matrix'] = confusion_mat
            evaluation_metrics['classification_rpt'] = classification_rpt

            print(f'Accuracy (train): {accuracy_train:.4f}')
            print(f'Precision (train): {precision_train:.4f}')
            print(f'Classification Report (train):\n{classification_rpt}')
            print(f'Confusion Matrix (train):\n{confusion_mat}')
        
        return model, evaluation_metrics
    
    

    def evaluate_model(self, model : any, x_test : pd.DataFrame, y_test : pd.Series, classification=False):

        y_pred = model.predict(x_test)
        evaluation_metrics = {}
        
        if classification:
            accuracy_test = accuracy_score(y_test, y_pred)
            precision_test = precision_score(y_test, y_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)
            classification_rpt = classification_report(y_test, y_pred)
            
            evaluation_metrics['accuracy_test'] = accuracy_test
            evaluation_metrics['precision_test'] = precision_test
            evaluation_metrics['confusion_matrix'] = confusion_mat
            evaluation_metrics['classification_report'] = classification_rpt

            print(f'Accuracy (test): {accuracy_test:.4f}')
            print(f'Precision (test): {precision_test:.4f}')
            print(f'Classification Report (test):\n{classification_rpt}')
            print(f'Confusion Matrix (test):\n{confusion_mat}')

        else:
            evaluation_metrics['r2_test'] = r2_score(y_test, y_pred)
            evaluation_metrics['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f'R-squared (test): {evaluation_metrics['r2_test']:.4f}')
            print(f'RMSE (test): {evaluation_metrics['rmse_test']:.4f}')
        
        return evaluation_metrics, y_pred
    

    def predict_new_data_point(self, model : any, x_new : pd.DataFrame):

        y_pred = model.predict(x_new)
        return y_pred