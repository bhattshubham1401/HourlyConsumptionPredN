import pandas as pd

from boostXG import BoostXG
import db
from DataAnalatics import DataAnalysis
from datetime import datetime
from modelH import ModelH
MONGO_URL = "mongodb://localhost:27017"

def convert_datetime(input_datetime):
    parsed_datetime = datetime.strptime(input_datetime, '%Y-%m-%dT%H:%M')
    formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_datetime


def main():
    print("Making Connectivity with mongo")
    mongo_url = MONGO_URL
    connector = db.MongoConnector(mongo_url)
    print("Connectivity is established")
    # Data analysis
    analysis = DataAnalysis(connector.db["grid_log"])
    hourly_data = analysis.analizeData()
    memory_usage = hourly_data.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert to megabytes
    print(f"Memory usage of DataFrame: {memory_usage:.2f} MB")
    model = BoostXG(hourly_data)
    # model.splitData()
    model.featureEngineering()
    model.modelXG_boost()
    # future_datetime = datetime(2023, 8, 19, 12, 0)
    model.predict_future()

    # Trains the model using modelXG_boost
    # model.predictXGBoost()
    # model.featureEngineering()

    # model = ModelH(train_data, val_data, train_data)
    # model.buildingModel()

    # future_predictions = model.generate_predictions(test_data, scaler)
    # print(future_predictions)


if __name__ == '__main__':
    main()
