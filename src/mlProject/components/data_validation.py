import pandas as pd

from src.mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def initiateDataValidation(self):
        try:
            validation = None
            read_df = pd.read_parquet(self.config.data_dir)
            col = list(read_df.columns)
            schema = self.config.all_schema.keys()

            for items in col:
                if items not in schema:
                    validation = False
                    with open(self.config.STATUS_FILE, 'w') as file:
                        file.write(f"validation Status: {validation}")

                else:
                    validation = True
                    with open(self.config.STATUS_FILE, 'w') as file:
                        file.write(f"validation Status: {validation}")

            return validation

        except Exception as e:
            print(f"Error in Validation Process: {e}")
