from src.mlProject import logger
from src.mlProject.components.data_transformation import Datatransformation
from src.mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation"


class DataTransformationTrainingPipeline:
    def __int__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            # print(data_transformation_config)
            data_transformation = Datatransformation(config=data_transformation_config)
            print(data_transformation)

            data_transformation.initiateDateTransformation()
        except Exception as e:
            logger.info(f"Error Occur in Data Transformation Pipeline {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        data = DataTransformationTrainingPipeline()
        data.main()
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    except Exception as e:
        logger.info(f"Error occur in Stage {STAGE_NAME}. ")
