# Importing necessary libraries to handle entities and configurations
from dataclasses import dataclass
from pathlib import Path

# Importing specific constants and utility functions from custom modules
from src.churn.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.churn.utils.commons import read_yaml, create_directories

import os
import json
from src.churn import logging
import pandas as pd

# Data validation entity 
@dataclass
class DataValidationConfig:
    root_dir: Path
    data_source: Path
    status_file: Path
    all_schema: dict
    critical_columns: list

# Create configuration manager 
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):
        
        """Initialize ConfigurationManager."""
        # Read YAML configuration files to initialize configuration parameters
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Create necessary directories specified in the configuration
        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            data_source=config.data_source,
            all_schema=schema,
            critical_columns=config.critical_columns
        )
        return data_validation_config

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


if __name__ == "__main__":
    try:
        # Initialize ConfigurationManager to get configuration settings
        config = ConfigurationManager()
        # Get data validation configuration
        data_validation_config = config.get_data_validation_config()
        # Create DataValidation object with the obtained configuration
        data_validation = DataValidation(config=data_validation_config)

        # Fetch data from CSV (or alternatively directly from MongoDB)
        data = pd.read_csv(data_validation_config.data_source)
        
        # Perform data validation for all columns
        column_validation_status = data_validation.validate_all_columns(data)
        
        # Perform data type validation for all columns
        type_validation_status = data_validation.validate_data_types(data)

        # Perform missing values validation
        missing_values_status = data_validation.validate_missing_values(data)

        # Create a dictionary with the validation results
        validation_results = {
            "validate_all_columns": column_validation_status,
            "validate_data_types": type_validation_status,
            "validate_missing_values": missing_values_status
        }

        # Save the results to the JSON file
        with open(data_validation_config.status_file, 'w') as f:
            json.dump(validation_results, f, indent=4)

        overall_validation_status = (column_validation_status and type_validation_status and 
                                     missing_values_status)

        if overall_validation_status:
            logging.info("Data Validation Completed Successfully!")
        else:
            logging.info("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logging.error(f"Data validation process failed: {e}")
        raise


