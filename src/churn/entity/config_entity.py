from dataclasses import dataclass
from pathlib import Path

# Entity Ingestion 
@dataclass
class DataIngestionConfig:
    root_dir: Path 
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int = 10000

# Data validation entity 
@dataclass
class DataValidationConfig:
    root_dir: Path
    data_source: Path
    status_file: Path
    all_schema: dict
    critical_columns: list


# Data Transformation entity
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list

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

# Model Evaluation 
@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    test_target_variable: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    # mlflow
    mlflow_uri: str

# Model Validation 
@dataclass
class ModelValidationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    test_target_variable: Path
    metric_file_name: Path
    all_params: dict
    target_column: str  # Add target_column attribute
    # mlflow
    mlflow_uri: str