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

from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)

            # Load the model as a dictionary
            models_dict = joblib.load(self.config.model_path)

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            for sensor_id in test_data['sensor'].unique():
                # Retrieve the specific model for the current sensor
                model = models_dict.get(sensor_id)

                if model is None:
                    # Handle the case where the model for the current sensor is not found
                    logger.warning(f"Model for sensor {sensor_id} not found.")
                    continue

                # Filter data for the current sensor
                sensor_data = test_data[test_data['sensor'] == sensor_id]
                test_x = sensor_data.drop(['sensor', 'Kwh'], axis=1)
                test_y = sensor_data['Kwh']

                with mlflow.start_run():
                    predicted_qualities = model.predict(test_x)

                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

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
            print(traceback.format_exc())
            logger.info(f"Error occur in Model Evaluation {e}")
