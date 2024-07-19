import pymongo 
import pandas as pd
import os
import logging
import time
from datetime import datetime
from src.churn.constants import *
from src.churn.entity.config_entity import DataIngestionConfig


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




