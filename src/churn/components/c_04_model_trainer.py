
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import os
import joblib
from src.churn import logging
from src.churn.entity.config_entity import ModelTrainerConfig

# mlflow
import dagshub
import mlflow 

mlflow.set_tracking_uri('https://dagshub.com/minich-code/churnpred.mlflow')
dagshub.init(repo_owner='minich-code', repo_name='churnpred', mlflow=True)


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

        
