import os
import traceback
import joblib
import pandas as pd
from xgboost import XGBRegressor
from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            data_files = [file for file in os.listdir(self.config.train_data_path) if file.startswith('train')]

            models_dict = {}  # Dictionary to store models

            for data_file in data_files:
                train_data_sensor = pd.read_csv(os.path.join(self.config.train_data_path, data_file))

                # Separate features and target
                X_train = train_data_sensor.drop(['Kwh'], axis=1)
                y_train = train_data_sensor['Kwh']

                # Train an XGBoost model on the sensor's data
                xgb_model = XGBRegressor()
                xgb_model.fit(X_train, y_train)

                # Calculate and print model evaluation metrics for this sensor
                train_score = xgb_model.score(X_train, y_train)
                print(f"Train Score for sensor {data_file}: {train_score}")

                # Perform hyperparameter tuning using RandomizedSearchCV
                best_params = {
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'subsample': self.config.subsample,
                    'colsample_bytree': self.config.colsample_bytree
                }

                # Initialize RandomizedSearchCV
                # random_search = RandomizedSearchCV(xgb_model,
                #                                    param_distributions=param_grid,
                #                                    n_iter=10,
                #                                    scoring='neg_mean_squared_error',
                #                                    cv=5,
                #                                    verbose=1,
                #                                    n_jobs=-1,
                #                                    random_state=42)
                #
                # # Fit the RandomizedSearchCV to the data
                # random_search.fit(X_train, y_train)
                #
                # # Get the best parameters
                # best_params = random_search.best_params_
                # print(f"Best Parameters for sensor {data_file}: {best_params}")

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'],
                                              subsample=best_params['subsample'],
                                              colsample_bytree=best_params['colsample_bytree'],
                                              reg_alpha=0.01,
                                              reg_lambda=0.01)
                best_xgb_model.fit(X_train, y_train)

                # Store the best model for this sensor in the dictionary
                sensor_id = data_file.split("_")[-1].split(".")[0]
                models_dict[sensor_id] = best_xgb_model

                # Log information about the best model
                logger.info(f"Best Model for sensor {data_file} - Best Parameters: {best_xgb_model.get_params()}, Train Score: {train_score}")

            # Save the dictionary of models as a single .joblib file
            joblib.dump(models_dict, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")
