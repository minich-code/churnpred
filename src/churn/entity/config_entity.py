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