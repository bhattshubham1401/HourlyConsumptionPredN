import os
from pathlib import Path
from urllib.parse import urlparse
from src.mlProject import logger
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback
from datetime import datetime, timedelta
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json
from src.mlProject.components.data_transformation import create_features, add_lags
from src.mlProject.utils.common import store_predictions_in_mongodb

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def save_model_as_dict(self, models_dict):
        # Save the model as a dictionary
        joblib.dump(models_dict, self.config.model_path)

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.config.model_path)

    def predict_future_values(self, model, sensor_id, num_periods=24):
        Current_Date = datetime.today()
        NextDay_Date = datetime.today() + timedelta(days=1)
        # Predict for future dates
        future_dates = pd.date_range(start=Current_Date, end=NextDay_Date, freq='H')
        # future_x[0]= 'Clock'

        future_x = create_features(pd.DataFrame({'sensor': [sensor_id] * len(future_dates)}, index=future_dates))
        future_x['Kwh'] = np.nan
        # Include lag features in future_x
        # print(future_x)
        future_x = add_lags(future_x)
        # print(future_x)
        FEATURES = ['sensor', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2',
                    'lag3']

        X_all = future_x[FEATURES]

        # Predict future values
        future_predictions = model.predict(X_all)

        # Log future predictions to a CSV file
        future_predictions_df = pd.DataFrame({"predicted_kwh": future_predictions}, index=future_dates)
        future_predictions_file_path = f"future_predictions_sensor_{sensor_id}.csv"
        future_predictions_df.to_csv(future_predictions_file_path)
        store_predictions_in_mongodb(sensor_id, future_dates, future_predictions)

        # Log the CSV file to MLflow
        mlflow.log_artifact(future_predictions_file_path)

    def log_into_mlflow(self):
        try:
            data_files = [file for file in os.listdir(self.config.test_data_path) if file.startswith('test')]
            print(data_files)
            test_data_list = []
            for data_file in data_files:
                test_data_sensor = pd.read_csv(os.path.join(self.config.test_data_path, data_file))
                test_data_list.append(test_data_sensor)

            # print(test_data_list)

            # Concatenate data for all sensors
            test_data = pd.concat(test_data_list, ignore_index=True)


            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            for sensor_id in test_data['sensor'].unique():
                model = loaded_model_dict.get(str(sensor_id))

                if model is None:
                    logger.warning(f"Model for sensor {sensor_id} not found.")
                    continue

                # Filter data for the current sensor
                sensor_data = test_data[test_data['sensor'] == sensor_id]
                test_x = sensor_data.drop(['Kwh'], axis=1)
                test_y = sensor_data['Kwh']

                with mlflow.start_run():
                    predicted_kwh = model.predict(test_x)
                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_kwh)

                    # Log predictions to a CSV file
                    predictions_df = pd.DataFrame({"actual_kwh": test_y, "predicted_kwh": predicted_kwh})
                    predictions_file_path = f"predictions_sensor_{sensor_id}.csv"
                    predictions_df.to_csv(predictions_file_path, index=False)

                    # Log the CSV file to MLflow
                    mlflow.log_artifact(predictions_file_path)

                    # Perform future date prediction
                    # last_date = test_data.index[-1]
                    self.predict_future_values(model, sensor_id)

                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_kwh)

                    # Saving metrics as local
                    scores = {"rmse": rmse, "mae": mae, "r2": r2}
                    save_json(path=Path(self.config.metric_file_name), data=scores)

                    mlflow.log_params(self.config.all_params)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    # Model registry does not work with file store
                    if tracking_url_type_store != "file":
                        # Register the model
                        mlflow.sklearn.log_model(model, "model", registered_model_name=f"{sensor_id}_Model")
                    else:
                        mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())


