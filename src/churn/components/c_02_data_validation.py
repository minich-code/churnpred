
from src.churn import logging
from src.churn.entity.config_entity import DataValidationConfig


# Data validation component class 
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        

    def validate_all_columns(self, data):
        try: 
            validation_status = True
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            missing_columns = [col for col in all_schema if col not in all_cols]
            extra_columns = [col for col in all_cols if col not in all_schema]

            if missing_columns or extra_columns:
                validation_status = False

            return validation_status 

        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False

    def validate_data_types(self, data):
        try: 
            validation_status = True
            all_schema = self.config.all_schema

            type_mismatches = {}
            for col, expected_type in all_schema.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if actual_type != expected_type:
                        type_mismatches[col] = (expected_type, actual_type)
                        validation_status = False

            return validation_status
        
        except Exception as e:
            raise e

    def validate_missing_values(self, data):
        try:
            validation_status = True
            missing_values = {}

            for col in self.config.critical_columns:
                if data[col].isnull().sum() > 0:
                    missing_values[col] = data[col].isnull().sum()
                    validation_status = False

            return validation_status

        except Exception as e:
            raise e


