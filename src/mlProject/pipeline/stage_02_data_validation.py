from src.mlProject import logger
from src.mlProject.components.data_validation import DataValidation
from src.mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.initiateDataValidation()
        except Exception as e:
            print(f"Error in Data Validation Pipeline: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>Stage {STAGE_NAME}  Started<<<<<<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>Stage {STAGE_NAME} Completed<<<<<<<<")
    except Exception as e:
        print(f"Error in Data Validation Pipeline Main method: {e}")
