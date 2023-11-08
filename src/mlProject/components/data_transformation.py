import pandas as pd
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.utils.common import plotData, sliderPlot
import seaborn as sns
import matplotlib.pyplot as plt


class Datatransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiateDateTransformation(self):
        try:
            df = pd.read_parquet(self.config.data_dir)
            # print(df['creation_time'].unique())
            df1 = df[['Clock', 'Kwh', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current', 'B_Current']]

            # Taking only those records where probe is connected
            df1 = df1[(df1['Kwh'] != 0)]

        except Exception as e:
            logger.info(f"Error occur in Data Transformation Layer {e}")
