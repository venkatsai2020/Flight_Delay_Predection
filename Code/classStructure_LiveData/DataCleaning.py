import pandas as pd
import openpyxl
import datetime
import pandas as pd
from datetime import datetime, timedelta
from data_loader import dataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class DataCleaning:


    def __init__(self) -> None:
        pass


    def time_to_minutes(self, time_str):
            if isinstance(time_str, str): # Check if time_str is a string
                try:
                    # Attempt to parse the time string as "hh:mm AM/PM"
                    time_obj = datetime.strptime(time_str, "%I:%M %p")
                except ValueError:
                    try:
                        # If ValueError (not in "hh:mm AM/PM" format), attempt to parse as "hh:mm"
                        time_obj = datetime.strptime(time_str, "%H:%M")
                    except ValueError:
                        # If both formats fail, return None or handle as needed
                        return None

                # Calculate total minutes from midnight
                total_minutes = time_obj.hour * 60 + time_obj.minute
                return total_minutes
            else:
                return None # Return None if time_str is not a string
            
    def normalize_features(self, X_train_encoded, numerical_features):
        """
        Normalize the specified numerical features in the DataFrame using MinMaxScaler.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        numerical_features (list): A list of numerical feature names to normalize.

        Returns:
        pd.DataFrame: The DataFrame with normalized numerical features.
        """
        scaler = MinMaxScaler()

        for feature in numerical_features:
            # Reshape the data to fit the scaler
            reshaped_feature = X_train_encoded[feature].values.reshape(-1, 1)
            # Apply the scaler and update the DataFrame
            X_train_encoded[feature] = scaler.fit_transform(reshaped_feature)

        return X_train_encoded
    
    def remove_outliers(self, data):
        numeric_data = data.select_dtypes(include=['int', 'float'])
        cleaned_data = data.copy()
        for col in numeric_data.columns:
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        return cleaned_data
    
    def display_outliers_y(self, data):
        numeric_data = data[data.apply(lambda x: isinstance(x, (int, float)))]
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if not outliers.empty:
            print("Outliers in y_train:")
            print(outliers)


    def display_outliers(self, data):
        numeric_data = data.select_dtypes(include=['int', 'float'])
        for col in numeric_data.columns:
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            if not outliers.empty:
                print("Outliers in column '{}':".format(col))
                print(outliers)

    def dataCleaning(self, file_path):

        data_loader = dataLoader(file_path)
        df = data_loader.load_excel_data()

        df.drop(columns=['Unnamed: 5'], inplace=True)

        # Assuming 'Scheduled_arrival' and 'Actual_arrival' are columns in your DataFrame 'df'
        Scheduled_arrival = df['Scheduled_arrival'] # Extract the 'Scheduled_arrival' column from df
        Actual_arrival = df['Actual_arrival'] # Extract the 'Actual_arrival' column from df

        # Create empty list for delays
        delays = []

        # Calculate delays for each pair of scheduled and actual arrival times
        for scheduled, actual in zip(Scheduled_arrival, Actual_arrival):
            scheduled_minutes = self.time_to_minutes(scheduled)
            actual_minutes = self.time_to_minutes(actual)

            if scheduled_minutes is not None and actual_minutes is not None:
                delay_minutes = actual_minutes - scheduled_minutes
                delays.append(delay_minutes)
            else:
                # Handle invalid time format or other errors
                delays.append(None)

        # Print delays
        print("Delays (minutes):", delays)

        df['Delay'] = delays

        X = df.drop(columns=['Delay'])
        y = df['Delay']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        null_sum_X_train = np.sum(X_train.isnull().values)

        # Sum of null values in y_train
        null_sum_y_train = np.sum(np.isnan(y_train))

        print("Sum of null values in X_train:", null_sum_X_train)
        print("Sum of null values in y_train:", null_sum_y_train)

        features_with_null = ['WEATHER_ARRIVAL', 'WIND_ARRIVAL', 'WEATHER_DEPARTURE','DIRECTION_ARRIVAL',
                            'WIND_DEPARTURE', 'Scheduled_departures', 'Actual_departures','DIRECTION_DEPARTURE',
                            'Scheduled_arrival', 'Actual_arrival']

        # Clean and convert data to numeric where possible
        for feature in features_with_null:
            # Remove non-numeric characters and convert to numeric
            X_train[feature] = X_train[feature].astype(str).str.extract('(\d+)', expand=False).astype(float)

            # Calculate median for the column
            median_value = X_train[feature].median()

            # Fill null values with median
            X_train[feature].fillna(median_value, inplace=True)

        # Print cleaned DataFrame
        print("DataFrame after filling null values with median:")
        print(X_train)

        features_with_null = ['WEATHER_ARRIVAL', 'WIND_ARRIVAL', 'WEATHER_DEPARTURE','DIRECTION_ARRIVAL',
                            'WIND_DEPARTURE', 'Scheduled_departures', 'Actual_departures','DIRECTION_DEPARTURE',
                            'Scheduled_arrival', 'Actual_arrival']

        # Clean and convert data to numeric where possible
        for feature in features_with_null:
            # Remove non-numeric characters and convert to numeric
            X_test[feature] = X_test[feature].astype(str).str.extract('(\d+)', expand=False).astype(float)

        # Print cleaned DataFrame
        print("DataFrame after filling null values with median:")
        print(X_test)

        y_train = y_train.fillna(y_train.median())

        # Assuming X_train is your dataset
        self.display_outliers(X_train)

        # Assuming y_train is your target variable
        self.display_outliers_y(y_train)

        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        # Remove outliers from X_train
        X_train_cleaned = self.remove_outliers(X_train)

        # Get the index of cleaned rows
        cleaned_indices = X_train_cleaned.index

        # Remove corresponding rows from y_train
        y_train_cleaned = y_train.loc[cleaned_indices]

        # Check the shape of cleaned datasets
        print("Shape of X_train after removing outliers:", X_train_cleaned.shape)
        print("Shape of y_train after removing outliers:", y_train_cleaned.shape)

        y_train = y_train_cleaned
        X_train = X_train_cleaned

        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_test:", y_test.shape)

        X_train_cleaned.dtypes

        X_train_cleaned.drop(columns=['FLIGHT','AIRCRAFT','STATUS','ARRIVAL_AIRPORT_NAME','ARRIVAL_AIRPORT_CODE'], inplace=True)
        X_test.drop(columns=['FLIGHT','AIRCRAFT','STATUS','ARRIVAL_AIRPORT_NAME','ARRIVAL_AIRPORT_CODE'], inplace=True)

        X_train_cleaned['TIME'] = pd.to_datetime(X_train_cleaned['TIME'], format='%I:%M %p')
        X_train_cleaned['Hour_of_Departure'] = X_train_cleaned['TIME'].dt.hour

        # Convert 'DATE' to datetime format and extract day of the week and month
        X_train_cleaned['DATE'] = pd.to_datetime(X_train_cleaned['DATE'], format='%A, %b %d')
        X_train_cleaned['Day_of_Week'] = X_train_cleaned['DATE'].dt.dayofweek  # Monday=0, Sunday=6
        X_train_cleaned['Month'] = X_train_cleaned['DATE'].dt.month

        # Drop the original 'TIME' and 'DATE' columns as they are no longer needed
        X_train_cleaned = X_train_cleaned.drop(columns=['TIME', 'DATE'])

        X_test['TIME'] = pd.to_datetime(X_test['TIME'], format='%I:%M %p')
        X_test['Hour_of_Departure'] = X_test['TIME'].dt.hour

        # Convert 'DATE' to datetime format and extract the day of the week and month
        X_test['DATE'] = pd.to_datetime(X_test['DATE'], format='%A, %b %d')
        X_test['Day_of_Week'] = X_test['DATE'].dt.dayofweek  # Monday=0, Sunday=6
        X_test['Month'] = X_test['DATE'].dt.month

        # Drop the original 'TIME' and 'DATE' columns
        X_test_cleaned = X_test.drop(columns=['TIME', 'DATE'])

        numerical_features = []

        # Loop through each column in the DataFrame
        for column in X_train_cleaned.columns:
            # Check if the column contains numerical data
            if pd.api.types.is_numeric_dtype(X_train_cleaned[column]):
                numerical_features.append(column)

        numerical_df = X_train_cleaned[numerical_features]

        # Calculate the correlation matrix for numerical features
        correlation_matrix = numerical_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

        threshold = 0.8

        # Find pairs of features with correlation above the threshold
        high_corr_pairs = []

        # Loop through the correlation matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > threshold:
                    colname_i = correlation_matrix.columns[i]
                    colname_j = correlation_matrix.columns[j]
                    high_corr_pairs.append((colname_i, colname_j, correlation_matrix.iloc[i, j]))

        # Print the pairs of features with high correlation
        for pair in high_corr_pairs:
            print(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.2f}")

        X_train_cleaned.drop(columns=['Scheduled_departures'], inplace=True)
        X_test.drop(columns=['Scheduled_departures'], inplace=True)



        X_train = X_train_cleaned

        X_train_cleaned.dtypes

        # prompt: export all cleaned data into one excel file

        
        with pd.ExcelWriter('cleaned_data.xlsx') as writer:
            X_train_cleaned.to_excel(writer, sheet_name='X_train_cleaned')
            y_train.to_excel(writer, sheet_name='y_train_cleaned')
            X_test_cleaned.to_excel(writer, sheet_name='X_test_cleaned')
            y_test.to_excel(writer, sheet_name='y_test_cleaned')

        categorical_features = ['FROM','AIRLINE']


        # Step 1: Initialize the OneHotEncoder with handle_unknown='ignore'
        encoder = OneHotEncoder(sparse=False, drop='first', dtype=int, handle_unknown='ignore')

        # Step 2: Fit the encoder on the training data
        encoder.fit(X_train_cleaned[categorical_features])

        # Step 3: Transform the training data and convert to DataFrame
        X_train_encoded = pd.DataFrame(encoder.transform(X_train_cleaned[categorical_features]),
                                    columns=encoder.get_feature_names_out(categorical_features))

        # Step 4: Transform the test data and convert to DataFrame
        X_test_encoded = pd.DataFrame(encoder.transform(X_test_cleaned[categorical_features]),
                                    columns=encoder.get_feature_names_out(categorical_features))

        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_test:", y_test.shape)

        X_test.isnull().sum()

        X_test_filled = X_test.fillna(median_value)

        y_test.isnull().sum()

        y_test_filled = y_test.fillna(median_value)

        y_test = y_test_filled
        X_test = X_test_filled

        y_test.isnull().sum()

        X_train = X_train_encoded
        X_test = X_test_encoded

        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_test:", y_test.shape)

        return X_train, y_train, X_test, y_test

        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression()
        # model.fit(X_train, y_train)

        # # Predict using the trained model
        # y_pred = model.predict(X_test)

        # # Display the results
        # print("X_test:\n", X_test)
        # print("y_test:\n", y_test)
        # print("y_pred:\n", y_pred)

        # from sklearn.metrics import mean_squared_error,r2_score
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # print(f"RMSE: {rmse:.2f}")

        # r2 = r2_score(y_test, y_pred)
        # print(f"R-squared: {r2:.2f}")

        # from sklearn.ensemble import RandomForestRegressor
        # import matplotlib.pyplot as plt
        # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # rf_model.fit(X_train, y_train)

        # # Predict using the trained model
        # y_pred = rf_model.predict(X_test)

        # # Calculate RMSE
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # # Calculate R-squared
        # r2 = r2_score(y_test, y_pred)
        # print(f"RMSE: {rmse:.2f}")
        # print(f"R-squared: {r2:.2f}")

        # # Plot actual vs predicted values
        # plt.figure(figsize=(10, 5))
        # plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        # plt.plot(y_test, y_test, color='red', label='Ideal Fit')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.title('Actual vs Predicted Values')
        # plt.legend()
        # plt.show()

        # from sklearn.model_selection import GridSearchCV

        # # Define the parameter grid
        # param_grid = {
        #     'n_estimators': [50, 100, 150],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }

        # # Initialize the GridSearchCV object
        # grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

        # # Fit the grid search to the data
        # grid_search.fit(X_train, y_train)

        # # Get the best parameters and the best score
        # print("Best parameters found:", grid_search.best_params_)
        # print("Best RMSE found:", np.sqrt(-grid_search.best_score_))

        # # Predict using the best model
        # best_rf_model = grid_search.best_estimator_
        # y_pred = best_rf_model.predict(X_test)

        # # Calculate RMSE and R-squared for the best model
        # rmse_best = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2_best = r2_score(y_test, y_pred)

        # print(f"RMSE with best model: {rmse_best:.2f}")
        # print(f"R-squared with best model: {r2_best:.2f}")

        # from sklearn.svm import SVR

        # model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)

        # print(f"SVR RMSE: {rmse}")
        # print(f"SVR R-squared: {r2}")

        # from sklearn.pipeline import Pipeline
        # from sklearn.preprocessing import StandardScaler

        # pipeline = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('svr', SVR())
        # ])

        # param_grid = {
        #     'svr__kernel': ['linear', 'poly', 'rbf'],
        #     'svr__C': [0.1, 1, 10, 100],
        #     'svr__gamma': ['scale', 'auto', 0.1, 1, 10],
        #     'svr__epsilon': [0.01, 0.1, 0.2]
        # }

        # # Perform Grid Search with cross-validation
        # grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        # grid_search.fit(X_train, y_train)

        # # Get the best parameters
        # best_params = grid_search.best_params_
        # print("Best parameters found:", best_params)

        # # Train the best model on the training set
        # best_model = grid_search.best_estimator_

        # # Evaluate the model on the validation set
        # y_test_pred = best_model.predict(X_test)
        # val_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        # val_r2 = r2_score(y_test, y_test_pred)

        # print("SVR RMSE after tuning:", val_rmse)
        # print("SVR R-squared after tuning:", val_r2)

        # from sklearn.ensemble import GradientBoostingRegressor

        # model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)

        # print(f"Gradient Boosting Regression RMSE: {rmse}")
        # print(f"Gradient Boosting Regression R-squared: {r2}")

        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
        #     'max_depth': [3, 5, 7],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }

        # # Perform Grid Search with cross-validation
        # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        # grid_search.fit(X_train, y_train)

        # # Get the best parameters
        # best_params = grid_search.best_params_
        # print("Best parameters found:", best_params)

        # # Train the best model on the training set
        # best_model = grid_search.best_estimator_

        # # Evaluate the model on the validation set
        # y_val_pred = best_model.predict(X_test)
        # val_rmse = np.sqrt(mean_squared_error(y_test, y_val_pred))
        # val_r2 = r2_score(y_test, y_val_pred)

        # print("Gradient Boosting RMSE after tuning:", val_rmse)
        # print("Gradient Boosting R-squared after tuning:", val_r2)

        # import xgboost as xgb

        # model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)

        # print(f"XGBoost RMSE: {rmse}")
        # print(f"XGBoost R-squared: {r2}")

        # import lightgbm as lgb

        # model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # r2 = r2_score(y_test, y_pred)

        # print(f"LightGBM RMSE: {rmse}")
        # print(f"LightGBM R-squared: {r2}")