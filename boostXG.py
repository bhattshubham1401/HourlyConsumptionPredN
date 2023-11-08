# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


def create_features(hourly_data):
    hourly_data = hourly_data.copy()
    hourly_data['day'] = hourly_data.index.day
    hourly_data['hour'] = hourly_data.index.hour
    hourly_data['month'] = hourly_data.index.month
    hourly_data['dayofweek'] = hourly_data.index.dayofweek
    hourly_data['quarter'] = hourly_data.index.quarter
    hourly_data['dayofyear'] = hourly_data.index.dayofyear
    hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
    hourly_data['year'] = hourly_data.index.year
    return hourly_data


def add_lags(df):
    target_map = df['hourly_diff'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df


class BoostXG:
    def __init__(self, hourly_data):
        self.location_data = {}
        self.scaler_y = StandardScaler()
        self.hourly_data = hourly_data.copy()

    def featureEngineering(self):
        pd.options.mode.chained_assignment = None
        location_data = self.hourly_data.groupby('location_id')
        grouped_data = location_data.groups

        for location_id in grouped_data:
            group_data = location_data.get_group(location_id)
            group_data['creation_time'] = pd.to_datetime(group_data['creation_time'])
            group_data = group_data.set_index('creation_time')
            group_data.index = pd.to_datetime(group_data.index)
            group_data = create_features(group_data)
            group_data = add_lags(group_data)
            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                        'lag1', 'lag2', 'lag3']
            group_data_features = group_data[FEATURES]

            # Split the data into training and testing sets
            X = group_data_features
            y = group_data['hourly_diff']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(X_train)

            self.location_data[location_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

    def modelXG_boost(self):
        for location_id, data in self.location_data.items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Train an XGBoost model for this location
            model = XGBRegressor()
            model.fit(X_train, y_train)

            # Calculate and print model evaluation metrics for this location
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"Location {location_id} - Train Score: {train_score}, Test Score: {test_score}")

            xgb_model = XGBRegressor()
            param_grid = {
                'n_estimators': [420],
                'max_depth': [5],
                'learning_rate': [0.1]}

            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error',
                                       cv=3)

            # Fit the GridSearchCV to the data
            grid_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = grid_search.best_params_
            print(f"Best Parameters: {best_params}")

            best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                          learning_rate=best_params['learning_rate'], reg_alpha=0.01, reg_lambda=0.01)
            best_xgb_model.fit(X_train, y_train)

            # Evaluate the model
            future_predictions = best_xgb_model.predict(X_test)
            future_predictions = np.round(future_predictions)
            y_test = np.round(y_test)
            rmse = mean_squared_error(y_test, future_predictions, squared=False)
            mae = mean_absolute_error(y_test, future_predictions)
            r2 = r2_score(y_test, future_predictions)
            # print(best_xgb_model.score(X_train, y_train)*100)
            # print(best_xgb_model.score(X_test, y_test)*100)
            print(f"RMSE: {rmse}")
            # print(f"MAE: {mae}")
            # print(f"R-squared: {r2}")
            print(len(future_predictions))
            print(len(y_test))
            print(len(X_test))

            # Uncomment the following code when you want to plot
            # plt.figure(figsize=(10, 6))
            # plt.scatter(X_test.index, y_test, label='Actual')
            # plt.scatter(X_test.index, future_predictions, label='Predicted')
            # plt.title(f'Actual vs. Predicted Hourly Consumption for Location {location_id}')
            # plt.xlabel('Index')
            # plt.ylabel('Hourly Consumption')
            # plt.legend()
            # plt.show()

    def predict_future(self):
        pd.options.mode.chained_assignment = None
        location_data = self.hourly_data.groupby('location_id')
        grouped_data = location_data.groups

        for location_id in grouped_data:
            group_data = location_data.get_group(location_id)
            group_data['creation_time'] = pd.to_datetime(group_data['creation_time'])
            group_data = group_data.set_index('creation_time')
            group_data.index = pd.to_datetime(group_data.index)
            group_data = create_features(group_data)
            group_data = add_lags(group_data)
            # max_date = group_data.index.max()
            # max_date_plus_days = max_date + pd.DateOffset(days=1)
            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                        'lag1', 'lag2', 'lag3']
            X_all = group_data[FEATURES]
            y_all = group_data['hourly_diff']

            reg = XGBRegressor(base_score=0.5,
                               booster='gbtree',
                               n_estimators=420,
                               objective='reg:squarederror',
                               max_depth=5,
                               learning_rate=0.1, reg_alpha=0.01, reg_lambda=0.01)
            reg.fit(X_all, y_all)

            future = pd.date_range('2023-08-17', '2023-08-18', freq='1h')

            group_data = location_data.get_group(location_id)
            future_df = pd.DataFrame(index=future)
            future_df['isFuture'] = True
            group_data['isFuture'] = False
            group_data['creation_time'] = pd.to_datetime(group_data['creation_time'])
            group_data = group_data.set_index('creation_time')
            group_data.index = pd.to_datetime(group_data.index)
            df_and_future = pd.concat([group_data, future_df])
            df_and_future = create_features(df_and_future)
            df_and_future = add_lags(df_and_future)
            future_w_features = df_and_future.query('isFuture').copy()
            future_w_features_filtered = future_w_features[FEATURES]
            future_w_features['pred'] = reg.predict(future_w_features_filtered)

            future_w_features['pred'] = np.round(future_w_features['pred'])

            print(f'Location: {location_id}')
            print(future_w_features['pred'])
            print(group_data.index.max())
            print(f'Sum of Consumed units are {sum(future_w_features["pred"])} units')
            future_w_features['pred'].plot(figsize=(10, 5),
                                           lw=1,
                                           title=f'Future Predictions for Location {location_id}')
            # plt.show()
