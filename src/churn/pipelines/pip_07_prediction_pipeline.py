import numpy as np 
import pandas as pd 
import os 
import sys 

from src.churn import logging 
from src.churn.exception import FileOperationError 
from src.churn.utils.commons import load_object 


class PredictionPipeline:
    def __init__(self):
        pass 

    def make_predictions(self, features):
        try:
            # Log a message 
            logging.info("Making predictions")

            model_path = os.path.join("artifacts", "model_trainer", "model.joblib")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor_obj.joblib")

            # load the preprocessor and model object
            preprocessor_obj = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Transform the features
            features_transformed = preprocessor_obj.transform(features)

            # Make predictions
            predictions = model.predict(features_transformed)

            # Convert NumPy int64 to Python int (for JSON serialization)
            predictions = [int(pred) for pred in predictions] 


            return predictions

        except Exception as e:
            raise FileOperationError(e, sys)
        

# Create a class to represent the input features 
class CustomData:
    def __init__(self, **kwargs):
        # Initiate the attributes using kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Define a method to convert data object to a dataframe 
    def get_data_as_dataframe(self):
        try:
            # log message 
            logging.info("Converting data object to a dataframe")
            # Convert the data object to a dataframe 
            data_dict = {key: [getattr(self, key)] for key in vars(self)}

            # Convert the dictionary to dataframe in the return  
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise FileOperationError(e, sys)
           