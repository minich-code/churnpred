from src.churn.config.configuration import ConfigurationManager
from src.churn.components.c_03_data_transformation import DataTransformation
from src.churn import logging 
from pathlib import Path 
import json


PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class DataTransformationPipeline:
    def __init__(self):
        pass 


    def run(self):
        try:
            # Read the JSON file
            with open(Path("artifacts/data_validation/status.json"), "r") as f:
                validation_results = json.load(f)

            # Check if all validation statuses are true
            if (
                validation_results["validate_all_columns"]
                and validation_results["validate_data_types"]
                and validation_results["validate_missing_values"]
            ):
                logging.info(
                    f"The data validation pipeline has already been executed Successfully !!!!"
                )
                logging.info(f"#====================== {PIPELINE_NAME} Started ================================#")

 
                config = ConfigurationManager()

                data_transformation_config = config.get_data_transformation_config()

                data_transformation = DataTransformation(config=data_transformation_config)

                X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_splitting()
                
                # Unpack the transformed data
                X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, preprocessor_path = \
                    data_transformation.initiate_data_transformation(X_train, X_val, X_test, y_train, y_val, y_test)
                
                # Store data 
                data_transformation.save_data(X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, preprocessor_path)
                
                logging.info(
                    f"# =========== {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx*************x"
                )


            else:
                raise Exception("The Data Schema is Invalid")

        except Exception as e:
            print(e)

if __name__ == "__main__":
    try:
        logging.info(f"# ============== {PIPELINE_NAME} Started ================#")
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.run()
        logging.info(f"# ============= {PIPELINE_NAME} Terminated Successfully! ======+++=====\n\nx*********************x") 
    except Exception as e:
        logging.exception(e)
        raise e

    
    