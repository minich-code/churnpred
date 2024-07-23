import os
import pandas as pd
import joblib
from category_encoders import TargetEncoder

from src.churn import logging
from src.churn.utils.commons import save_object
from src.churn.entity.config_entity import DataTransformationConfig

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# Create a class to handle the actual data transformation process
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_obj(self, X_train, y_train):
        try:
            # Separate the numeric and categorical columns
            numerical_cols = self.config.numerical_cols
            categorical_cols = self.config.categorical_cols

            # Create a pipeline for numeric columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])

            # Create a target encoder instance for categorical columns. No need to scale after target encoding
            categorical_transformer = Pipeline(steps=[
                # Target encoder 
                ('target_encoder', TargetEncoder(cols=categorical_cols))

            ])

            # Combine the numeric pipeline and target encoder
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_transformer, numerical_cols),
                    ('categorical', categorical_transformer, categorical_cols)
                ],
                remainder='passthrough'
            )

            # Fit the preprocessor with X_train and y_train
            preprocessor.fit(X_train, y_train)

            # Return the preprocessor object
            return preprocessor

        except Exception as e:
            raise e


    # Split the data into training, validation, and testing sets
    def train_val_test_splitting(self):
        try:
            logging.info("Data Splitting process has started")

            df = pd.read_csv(self.config.data_path)

            X = df.drop(columns=["churn_risk_score"])
            y = df["churn_risk_score"]

            logging.info("Splitting data into training, validation, and testing sets")

            X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% remaining
            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42) # Split remaining 30% 

            logging.info("Saving the training, validation, and testing data in artifacts")

            # Save the target variable for each set
            y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
            y_val.to_csv(os.path.join(self.config.root_dir, "y_val.csv"), index=False)
            y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)

            logging.info("Data Splitting process has completed")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            raise e

    # Initiate data transformation
    def initiate_data_transformation(self, X_train, X_val, X_test, y_train, y_val, y_test):
        try:
            logging.info("Data Transformation process has started")

            # Get the preprocessor object
            preprocessor_obj = self.get_transformer_obj(X_train, y_train)

            # Transform the training, validation, and test data
            X_train_transformed = preprocessor_obj.transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the preprocessing object
            preprocessor_path = os.path.join(self.config.root_dir, "preprocessor_obj.joblib")
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            # Save the transformed data
            X_train_transformed_path = os.path.join(self.config.root_dir, "X_train_transformed.joblib")
            X_val_transformed_path = os.path.join(self.config.root_dir, "X_val_transformed.joblib")
            X_test_transformed_path = os.path.join(self.config.root_dir, "X_test_transformed.joblib")
            joblib.dump(X_train_transformed, X_train_transformed_path)
            joblib.dump(X_val_transformed, X_val_transformed_path)
            joblib.dump(X_test_transformed, X_test_transformed_path)

            logging.info("Data Transformation process has completed")

            # Return
            return X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, preprocessor_path

        except Exception as e:
            raise e



