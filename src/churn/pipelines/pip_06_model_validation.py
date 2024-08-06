from src.churn.config.configuration import ConfigurationManager
from src.churn.components.c_06_model_validation import ModelValidation
from src.churn import logging 
import joblib 
import pandas as pd 


PIPELINE_NAME = "MODEL VALIDATION PIPELINE"

class ModelValidationPipeline:
    def __init__(self):
        pass 

    def run(self):
        # Initialize configuration
        config = ConfigurationManager()
        model_validation_config = config.get_model_validation_config()

        # Load model and data
        model = joblib.load(model_validation_config.model_path)
        X_test_transformed = joblib.load(model_validation_config.test_data_path)
        y_test_df = pd.read_csv(model_validation_config.test_target_variable)
        y_test = y_test_df[model_validation_config.target_column]

        # Validate model
        model_validation = ModelValidation(config=model_validation_config)
        y_pred, y_pred_proba = model_validation.predictions(model, X_test_transformed)
        model_validation.save_results(y_test, y_pred, y_pred_proba)


if __name__=="__main__":
    try:
        logging.info(f"# ============== {PIPELINE_NAME} Started ================#")
        metrics_validation_pipeline = ModelValidationPipeline()
        metrics_validation_pipeline.run()
        logging.info(f"# ============= {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx******************x") 
    except Exception as e:
        logging.exception(e)
        raise e





