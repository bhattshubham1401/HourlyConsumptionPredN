import pandas as pd


class DataAnalysis:
    def __init__(self, db_collection):
        self.collection = db_collection

    def analizeData(self):
        cursor = self.collection.find({"site_id": "6075bb51153a20.38235471"},
                                      {"location_id": 1, "data.creation_time": 1, "data.grid_reading_kwh": 1})

        data = []
        for doc in cursor:
            location_id = doc["location_id"]
            creation_time = doc["data"]["creation_time"]
            grid_reading_kwh = doc["data"]["grid_reading_kwh"]

            data.append({
                "location_id": location_id,
                "creation_time": creation_time,
                "grid_reading_kwh": grid_reading_kwh
            })
        df = pd.DataFrame(data)
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        df['grid_reading_kwh'] = df['grid_reading_kwh'].astype(float)

        # for displaying all record
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        hourly_data = df.groupby(['location_id', pd.Grouper(freq='H', key='creation_time')]).agg({'grid_reading_kwh': ['first', 'last']}).reset_index()
        # print(hourly_data.head())
        hourly_data.columns = hourly_data.columns.droplevel(0)
        hourly_data['hourly_diff'] = hourly_data['last'] - hourly_data['first']
        hourly_data.drop(columns=['first', 'last'], inplace=True)
        hourly_data.fillna(method='ffill', inplace=True)
        hourly_data.columns = ['location_id', 'creation_time', 'hourly_diff']
        hourly_data['creation_time'] = pd.to_datetime(hourly_data['creation_time'])
        # print(hourly_data.head())
        # exit()
        return hourly_data

