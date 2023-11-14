from src.mlProject import logger
from src.mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.mlProject.pipeline.stage_04_model_trainer import ModelTrainingPipeline

logger.info("We are printing the logs here!!!")

# STAGE_NAME = "Data Ingestion"
#
# try:
#     logger.info(f"-----------Stage {STAGE_NAME} Started------------")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f"------------Stage {STAGE_NAME} Completed---------------------")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# STAGE_NAME = "Data Validation"
#
# try:
#     logger.info(f">>>>>>>>>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<<<<<<")
#     obj = DataValidationTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Completed<<<<<<<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Transformation"
#
# try:
#     logger.info(f">>>>>>>>>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<<<<<<")
#     obj = DataTransformationTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Completed<<<<<<<<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>>>>>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>>>>>>Stage {STAGE_NAME} Completed<<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e
