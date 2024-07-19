from dataclasses import dataclass
from pathlib import Path

# Entity 
@dataclass
class DataIngestionConfig:
    root_dir: Path 
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int = 10000