from src.mlProject import logger
from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiateDataIngestion()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e

