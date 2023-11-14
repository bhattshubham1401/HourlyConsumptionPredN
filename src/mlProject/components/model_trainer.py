import traceback
import joblib
import pandas as pd
from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.best_models = {}  # Dictionary to store the best models for each sensor

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
                    'learning_rate': self.config.learning_rate
                }

                # Initialize GridSearchCV
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error',
                                           cv=3)

                # Fit the GridSearchCV to the data
                grid_search.fit(X_train, y_train)

                # Get the best parameters
                best_params = grid_search.best_params_
                print(f"Best Parameters for Sensor {sensor_id}: {best_params}")

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'], reg_alpha=0.01,
                                              reg_lambda=0.01)
                best_xgb_model.fit(X_train, y_train)

                # Save the best model for each sensor
                model_filename = f"{self.config.model_name}_{sensor_id}.joblib"
                joblib.dump(best_xgb_model, os.path.join(self.config.root_dir, model_filename))
                self.best_models[sensor_id] = best_xgb_model

            # Log information about the best models
            logger.info("Best Models:")
            for sensor_id, model in self.best_models.items():
                logger.info(f"Sensor {sensor_id} - Best Parameters: {model.get_params()}, Score: {model.score(X_test, y_test)}")

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")

