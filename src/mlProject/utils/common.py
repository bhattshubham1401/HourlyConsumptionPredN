'''functionality that we are use in our code'''

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from dotenv import load_dotenv
from ensure import ensure_annotations
from pymongo import MongoClient
from src.mlProject import logger

load_dotenv()


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args\\\\\\\\\\\\\\
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
       memory_usage = hourly_data.memory_usage(deep=True).sum() / (1024 ** 2)
    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def convert_datetime(input_datetime):
    parsed_datetime = datetime.strptime(input_datetime, '%Y-%m-%dT%H:%M')
    formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_datetime


@ensure_annotations
def get_mongoData():
    ''' calling DB configuration '''

    logger.info("calling DB configuration")
    db = os.getenv("db")
    host = os.getenv("host")
    port = os.getenv("port")
    collection = os.getenv("collection")

    MONGO_URL = f"mongodb://{host}:{port}"

    ''' Read data from DB'''

    '''Writing logs'''
    logger.info("Reading data from Mongo DB")

    '''Exception Handling'''

    try:
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection]

        data = collection.find({})

        columns = ['sensor', 'Clock', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
                   'B_Current', 'A', 'BlockEnergy-WhExp', 'B', 'C', 'D', 'BlockEnergy-VAhExp',
                   'Kwh', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp']

        datalist = [(entry['sensor_id'], entry['raw_data']) for entry in data]
        df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)
        '''Dropping Columns'''
        df = df.drop(
            ['BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
             'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
        pd.set_option('display.max_columns', None)

        # print("===============DataType Conversion==================")
        df['Clock'] = pd.to_datetime(df['Clock'])
        df['Kwh'] = df['Kwh'].astype(float)
        df['R_Voltage'] = df['R_Voltage'].astype(float)
        df['Y_Voltage'] = df['Y_Voltage'].astype(float)
        df['B_Voltage'] = df['B_Voltage'].astype(float)
        df['R_Current'] = df['R_Current'].astype(float)
        df['Y_Current'] = df['Y_Current'].astype(float)
        df['B_Current'] = df['B_Current'].astype(float)

        return df

    except Exception as e:
        logger.info(f"Error occurs =========== {e}")


@ensure_annotations
def load_file():
    file = os.getenv("filename")
    return file


@ensure_annotations
def plotData(df1):
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['KWh'], df1['cumm_PF'], label='Actual')
    plt.xlabel('KWh ')
    plt.ylabel('cumm_PF')
    plt.legend()
    plt.show()
    return

    # Line plot
    # sns.lineplot(x='x_column', y='y_column', data=data)
    # plt.show()
    #
    # # Histogram
    # sns.histplot(data['numeric_column'], bins=10)
    # plt.show()
    #
    # # Box plot
    # sns.boxplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Bar plot
    # sns.barplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Pair plot (for exploring relationships between multiple variables)
    # sns.pairplot(data)
    # plt.show()


@ensure_annotations
def sliderPlot(df1):
    fig = px.line(df1, x=df1['meter_timestamp'], y=df1['KWh'])
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )

    )
    fig.show()
    return


@ensure_annotations
def store_predictions_in_mongodb(sensor_id, dates, predictions):
    try:
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection = os.getenv("collection1")

        MONGO_URL = f"mongodb://{host}:{port}"

        ''' Read data from DB'''

        '''Writing logs'''
        logger.info("Inserting Prediction data in Mongo DB")
        labeled_to_original_mapping = {
            0: "5f718c439c7a78.65267835",
            1: "62a9920f75c931.62399458",
        }

        # Use the labeled sensor ID to get the original sensor ID
        original_sensor_id = labeled_to_original_mapping.get(sensor_id, sensor_id)

        data = {
            "_id": f"{original_sensor_id}_{datetime.now().strftime('%Y-%m-%d')}",
            "sensor_id": original_sensor_id,
            "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "millisecond": int(datetime.now().timestamp() * 1000),
            "data": {}
        }

        # Populate the 'data' dictionary with hourly predictions
        for i, date_str in enumerate(dates):
            prediction_float = round(float(predictions[i]), 4)
            data["data"][f"{i}"] = {
                "pre_kwh": prediction_float,
                "pre_current": 0.0,
                "pre_load": 0.0,
                "act_kwh": 0.0,
                "act_load": 0.0
            }
        data_dict = {key: float(value) if isinstance(value, (float, np.integer, float, np.floating)) else value for
                     key, value
                     in data.items()}
        # print("Data Dictionary:", data_dict)

        # Connect to MongoDB
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection]
        # Insert data into MongoDB
        collection.insert_one(data_dict)
        client.close()
    except Exception as e:
        print(e)
