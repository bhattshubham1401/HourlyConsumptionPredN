import os
import warnings

from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


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
    target_map = df['Kwh'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('1 hour')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('1 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    # df['lag4'] = (df.index - pd.Timedelta('30 days')).map(target_map)
    return df


class Datatransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiateDateTransformation(self):
        try:
            df1 = pd.read_parquet(self.config.data_dir)
            # print(df1)
            sensor = df1['sensor'].unique()

            for items in sensor:
                df = df1[['sensor', 'Clock', 'Kwh', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
                          'B_Current']]
                sensor_df = df[df['sensor'] == items]
                filtered_df = sensor_df[
                    ((sensor_df['R_Voltage'] == 0) | (sensor_df['Y_Voltage'] == 0) | (sensor_df['B_Voltage'] == 0)) & (
                            (sensor_df['R_Current'] == 0) | (
                            sensor_df['Y_Current'] == 0) | (sensor_df['B_Current'] == 0))]
                filtered_df['Kwh'] = 0
                df.loc[filtered_df.index, :] = filtered_df

                '''Data Convesion'''
                sensor_df['Clock'] = pd.to_datetime(df['Clock'])
                sensor_df.set_index(['Clock'], inplace=True, drop=True)
                sensor_df = sensor_df[sensor_df.index >= '2022-11-18 00:00:00']
                pd.set_option('display.max_columns', None)

                '''Resampling dataframe into one hour interval '''
                dfresample = sensor_df[['Kwh']].resample(rule='1H').sum()
                dfresample['sensor'] = items

                # print(dfresample)
            # '''Train test Split'''
                tss = TimeSeriesSplit(n_splits=5, max_train_size=24*30*1, gap=24)
                df = dfresample.sort_index()
                df = add_lags(df)

                for idx, (train_idx, val_idx) in enumerate(tss.split(df)):
                    train = df.iloc[train_idx]
                    test = df.iloc[val_idx]

                    train = create_features(train)
                    test = create_features(test)

                    FEATURES = ['sensor', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                                'lag1', 'lag2', 'lag3']
                    TARGET = ['Kwh']

                    train_data = train[FEATURES + TARGET]
                    test_data = test[FEATURES + TARGET]

                    train_data.to_csv(os.path.join(self.config.root_dir, f"train_data.csv"), mode='a',
                                      header=not os.path.exists(os.path.join(self.config.root_dir, f"train_data.csv")),
                                      index=False)
                    test_data.to_csv(os.path.join(self.config.root_dir, f"test_data.csv"), mode='a',
                                     header=not os.path.exists(os.path.join(self.config.root_dir, f"test_data.csv")),
                                     index=False)


        except Exception as e:
            logger.info(f"Error occur in Data Transformation Layer {e}")
