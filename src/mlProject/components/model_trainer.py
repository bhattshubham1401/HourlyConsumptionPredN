import os
import traceback

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.best_models = {}

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            for sensor_id in train_data['sensor'].unique():
                # Filter data for the current sensor
                train_data_sensor = train_data[train_data['sensor'] == sensor_id]
                test_data_sensor = test_data[test_data['sensor'] == sensor_id]

                # Separate features and target
                X_train = train_data_sensor.drop(['sensor', 'Kwh'], axis=1)
                y_train = train_data_sensor['Kwh']

                X_test = test_data_sensor.drop(['sensor', 'Kwh'], axis=1)
                y_test = test_data_sensor['Kwh']

                # Reuse the same XGBoost instance for training and hyperparameter tuning
                xgb_model = XGBRegressor()

                # Train an XGBoost model for this sensor
                xgb_model.fit(X_train, y_train)

                # Calculate and print model evaluation metrics for this sensor
                train_score = xgb_model.score(X_train, y_train)
                test_score = xgb_model.score(X_test, y_test)
                print(f"Sensor {sensor_id} - Train Score: {train_score}, Test Score: {test_score}")

                # Perform hyperparameter tuning using GridSearchCV
                param_grid = {
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'subsample': self.config.subsample,
                    'colsample_bytree': self.config.colsample_bytree
                }

                # Initialize GridSearchCV
                grid_search = RandomizedSearchCV(xgb_model,
                                                 param_distributions=param_grid,
                                                 n_iter=10,
                                                 scoring='accuracy',
                                                 cv=5,
                                                 verbose=1,
                                                 n_jobs=-1,
                                                 random_state=42)

                # Fit the GridSearchCV to the data
                grid_search.fit(X_train, y_train)

                # Get the best parameters
                best_params = grid_search.best_params_
                print(f"Best Parameters for Sensor {sensor_id}: {best_params}")

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'],
                                              subsample=best_params['subsample'],
                                              colsample_bytree=best_params['colsample_bytree'],
                                              reg_alpha=0.01,
                                              reg_lambda=0.01)
                best_xgb_model.fit(X_train, y_train)

                # Save the best model for each sensor in the dictionary
                self.best_models[sensor_id] = best_xgb_model

            # Save the dictionary of best models into a single joblib file
            joblib.dump(self.best_models, os.path.join(self.config.root_dir, self.config.model_name))

            # Log information about the best models
            logger.info("Best Models:")
            for sensor_id, model in self.best_models.items():
                logger.info(
                    f"Sensor {sensor_id} - Best Parameters: {model.get_params()}, Score: {model.score(X_test, y_test)}")

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")
