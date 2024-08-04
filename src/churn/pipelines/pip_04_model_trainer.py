from src.churn.config.configuration import ConfigurationManager
from src.churn.components.c_03_data_transformation import DataTransformation
from src.churn.components.c_04_model_trainer import ModelTrainer
from src.churn import logging 


PIPELINE_NAME = "MODEL TRAINER PIPELINE"

class ModelTrainerPipeline:
    def __init__(self):
        pass 

    def run(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_splitting()

        # Unpack the transformed data
        X_train_transformed, X_val_transformed, X_test_transformed, y_train_transformed, y_val_transformed, y_test_transformed, _ = data_transformation.initiate_data_transformation(
            X_train, X_val, X_test, y_train, y_val, y_test)
        
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer(X_train_transformed, X_val_transformed, y_train_transformed, y_val_transformed)
    


if __name__ == "__main__":
    try:
        logging.info(f"# ========= {PIPELINE_NAME} Started ================#")
        model_trainer_pipeline = ModelTrainerPipeline()
        model_trainer_pipeline.run()
        logging.info(f"# ============= {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx********************x")
    except Exception as e:
        logging.exception(e)
        raise e