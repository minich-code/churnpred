from dataclasses import dataclass 
from pathlib import Path 
import pymongo 
import pandas as pd
import os
import logging
import time
from datetime import datetime
from src.churn.utils.commons import read_yaml, create_directories
from src.churn.constants import *

# Entity 
@dataclass
class DataIngestionConfig:
    root_dir: Path 
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int = 1000

# Creating a ConfigurationManager class to manage configurations
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
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

# Data ingestion component 
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(self.config.mongo_uri)
            db = client[self.config.database_name]
            collection = db[self.config.collection_name]

            # Log start of data ingestion
            logging.info(f"Starting data ingestion from MongoDB collection {self.config.collection_name}")

            # Fetch and process data in batches
            batch_size = self.config.batch_size
            batch_num = 0

            # Create an empty DataFrame to store all data
            all_data = pd.DataFrame()

            while True:
                # Create a cursor for each batch 
                cursor = collection.find().skip(batch_num * batch_size).limit(batch_size)
                df = pd.DataFrame(list(cursor))

                if df.empty:
                    break

                # Drop MongoDB's internal _id field if present
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])

                # Append the batch to all_data DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)

                logging.info(f"Fetched batch {batch_num + 1} with {len(df)} records")
                batch_num += 1

            # Save the DataFrame to a CSV file in the root directory
            output_path = os.path.join(self.config.root_dir, 'web_churn.csv')
            all_data.to_csv(output_path, index=False)
            logging.info(f"Data fetched from MongoDB and saved to {output_path}")

            # Log total number of records ingested
            total_records = len(all_data)
            logging.info(f"Total records ingested: {total_records}")

            # Save metadata
            end_time = time.time()
            end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration = end_time - start_time
            metadata = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'duration': duration,
                'total_records': total_records,
                'data_source': self.config.collection_name,
                'output_path': output_path
            }
            metadata_path = os.path.join(self.config.root_dir, 'metadata.json')
            pd.Series(metadata).to_json(metadata_path)
            logging.info(f"Metadata saved to {metadata_path}")

            # Monitoring metrics
            ingestion_speed = total_records / duration
            logging.info(f"Ingestion speed: {ingestion_speed} records/second")
            logging.info(f"Data volume: {total_records} records")

        except Exception as e:
            logging.error(f"Error occurred while importing data from MongoDB: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.import_data_from_mongodb()
        logging.info("Data Ingestion from MongoDB Completed!")
    except Exception as e:
        logging.error(f"Data ingestion process failed: {e}")
        raise



