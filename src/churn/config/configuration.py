
from src.churn.utils.commons import read_yaml, create_directories
from src.churn.constants import *
from src.churn.entity.config_entity import *


# Creating a ConfigurationManager class to manage configurations
class ConfigurationManager:
    def __init__(self, 
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

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        # Get data ingestion section from config
        config = self.config.data_ingestion

        # Create DataIngestionConfig object
        create_directories([config.root_dir])

        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            mongo_uri=config.mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name,
            batch_size=config.get('batch_size', 10000)  # Optional batch size
        )

        return data_ingestion_config


# Data validation config manager 
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
    
# Data transformation config manager

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            numerical_cols=list(config.numerical_cols),
            categorical_cols=list(config.categorical_cols)
        )
        return data_transformation_config
    
# Model trainer configuration 

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        # Get the model trainer configuration 
        config = self.config.model_trainer
        params = self.params.LGBMClassifier
        create_directories([config.root_dir])
        # Create and return the Model Trainer Config object
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            model_name=config.model_name,

            # LGBMClassifier hyperparameters
            boosting_type=params['boosting_type'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            objective=params['objective'],
            min_split_gain=params['min_split_gain'],
            min_child_weight=params['min_child_weight'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=params['random_state'],
            min_child_samples=params['min_child_samples'],

            # mlflow 
            mlflow_uri= config.mlflow_uri,

            
        )
        return model_trainer_config



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.LGBMClassifier  # Update to LGBMClassifier parameters
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            test_target_variable=config.test_target_variable,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            # mlflow 
            mlflow_uri= config.mlflow_uri
        )

        return model_evaluation_config



