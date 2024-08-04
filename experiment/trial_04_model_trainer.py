import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import os
from dataclasses import dataclass
from pathlib import Path
from src.churn.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.churn.utils.commons import read_yaml, create_directories, save_json
from src.churn import logging
from experiment.trial_03_data_transformation import DataTransformationConfig
from experiment.trial_03_data_transformation import DataTransformation


# mlflow
import dagshub
import mlflow 

mlflow.set_tracking_uri('https://dagshub.com/minich-code/churnpred.mlflow')
dagshub.init(repo_owner='minich-code', repo_name='churnpred', mlflow=True)

# Model trainer entity 
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    model_name: str
    # LGBMClassifier 
    boosting_type: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    objective: str
    min_split_gain: float
    min_child_weight: float
    reg_alpha: float
    reg_lambda: float
    random_state: int
    min_child_samples: int
    # mlflow
    mlflow_uri: str


# Class for the configuration manager 
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH, 
                 schema_filepath=SCHEMA_FILE_PATH):
        
        """Initialize ConfigurationManager.""" 
        # Read YAML configurations files to initialize configuration parameters
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Create necessary directories specified in the configuration
        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data ingestion configuration."""
        # Get the data transformation section from the config
        config = self.config.data_transformation 
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            numerical_cols=list(config.numerical_cols),
            categorical_cols=list(config.categorical_cols)
        )
        return data_transformation_config
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
    
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, X_train_transformed, X_val_transformed, y_train, y_val):
        # Start an MLflow run
        with mlflow.start_run(run_name="LGBM_Training"):  # Set a run name for clarity

            # Initialize the Stratified K-Fold cross-validator
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Perform Stratified K-Fold Cross-Validation
            fold_f1_scores = []
            for fold, (train_index, val_index) in enumerate (skf.split(X_train_transformed, y_train)):
                X_train_fold, X_val_fold = X_train_transformed[train_index], X_train_transformed[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                # Train the model on the training fold
                lgbm_model = LGBMClassifier(
                    boosting_type=self.config.boosting_type,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    n_estimators=self.config.n_estimators,
                    objective=self.config.objective,
                    min_split_gain=self.config.min_split_gain,
                    min_child_weight=self.config.min_child_weight,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    min_child_samples=self.config.min_child_samples,
                    verbose=0,
                    force_row_wise=True
                )
                lgbm_model.fit(X_train_fold, y_train_fold)

                # Validate the model on the validation fold
                y_val_pred = lgbm_model.predict(X_val_fold)

                # Evaluate the model
                fold_f1 = f1_score(y_val_fold, y_val_pred, average='macro')
                fold_f1_scores.append(fold_f1)
                print(f"Fold Validation Macro F1-Score: {fold_f1}")

                # Log fold-specific metrics in MLflow
                mlflow.log_metric("fold_f1_score", fold_f1, step=fold)

            # Print average Macro F1-Score across all folds
            avg_f1 = sum(fold_f1_scores) / len(fold_f1_scores)
            print(f"Average Cross-Validation Macro F1-Score: {avg_f1}")

            # Log average F1 in MLflow
            mlflow.log_metric("avg_f1_score", avg_f1)

            # Final training on full training set
            lgbm_model.fit(X_train_transformed, y_train)

            # Log model hyperparameters
            mlflow.log_param("boosting_type", self.config.boosting_type)
            mlflow.log_param("max_depth", self.config.max_depth)
            mlflow.log_param("learning_rate", self.config.learning_rate)
            mlflow.log_param("n_estimators", self.config.n_estimators)
            mlflow.log_param("objective", self.config.objective)
            mlflow.log_param("min_split_gain", self.config.min_split_gain)
            mlflow.log_param("min_child_weight", self.config.min_child_weight)
            mlflow.log_param("reg_alpha", self.config.reg_alpha)
            mlflow.log_param("reg_lambda", self.config.reg_lambda)
            mlflow.log_param("random_state", self.config.random_state)
            mlflow.log_param("min_child_samples", self.config.min_child_samples)

            # Save the model
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(lgbm_model, model_path)

            # Log model artifact in MLflow
            mlflow.log_artifact(model_path)

            # Register the model in MLflow
            mlflow.sklearn.log_model(lgbm_model, "model")
            mlflow.register_model(
                "runs:/" + mlflow.active_run().info.run_id + "/model", "MyLGBMModel" 
            )

        # End the MLflow run from the training 
        mlflow.end_run()

        # Logging info 
        logging.info(f"Model Trainer completed: Saved to {model_path}")

        
if __name__ == "__main__":
    try:
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
    
    except Exception as e:
        mlflow.end_run()  # Ensure the run is ended in case of an exception
        raise e