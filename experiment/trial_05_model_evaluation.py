from pathlib import Path
from dataclasses import dataclass 
from src.churn.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.churn.utils.commons import read_yaml, create_directories, save_json


import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier

# mlflow 
import dagshub
import mlflow

# UserWarning
mlflow.set_tracking_uri('https://dagshub.com/minich-code/churnpred.mlflow')
dagshub.init(repo_owner='minich-code', repo_name='churnpred', mlflow=True)



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

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

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



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def predictions(self, model, X_val_transformed):
        y_pred = model.predict(X_val_transformed)
        y_pred_proba = model.predict_proba(X_val_transformed)  # For ROC-AUC and PR-AUC
        return y_pred, y_pred_proba

    def model_evaluation(self, y_val, y_pred, y_pred_proba):
        # Evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
        pr_auc = average_precision_score(y_val, y_pred_proba, average='weighted')



        # Print detailed classification report and confusion matrix
        print("Classification Report:\n", classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        
        # Return evaluation metrics
        return accuracy, precision, recall, f1, roc_auc, pr_auc

    def plot_roc_curve(self, y_val, y_pred_proba):
        # Binarize the output
        y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
        n_classes = y_val_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.config.root_dir, 'roc_curve.png'))
        plt.close()

    def plot_pr_curve(self, y_val, y_pred_proba):
        # Binarize the output
        y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
        n_classes = y_val_bin.shape[1]
        
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_val_bin[:, i], y_pred_proba[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # Plot Precision-Recall curve for each class
        plt.figure()
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], label=f'Class {i} (area = {pr_auc[i]:0.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.config.root_dir, 'pr_curve.png'))
        plt.close()

    def save_results(self, y_val, y_pred, y_pred_proba):
        accuracy, precision, recall, f1, roc_auc, pr_auc = self.model_evaluation(y_val, y_pred, y_pred_proba)

        # Saving metrics as local JSON file
        scores = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

        # Plotting ROC Curve
        self.plot_roc_curve(y_val, y_pred_proba)

        # Plotting Precision-Recall Curve
        self.plot_pr_curve(y_val, y_pred_proba)

        # Log metrics in MLflow
        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("ROC_AUC", roc_auc)
            mlflow.log_metric("PR_AUC", pr_auc)

            # Log model parameters
            mlflow.log_params(self.config.all_params)

            # Log artifacts
            mlflow.log_artifact(os.path.join(self.config.root_dir, 'roc_curve.png'))
            mlflow.log_artifact(os.path.join(self.config.root_dir, 'pr_curve.png'))


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Load the trained model, transformed test data, and y_test data
        model = joblib.load(model_evaluation_config.model_path)
        X_test_transformed = joblib.load(model_evaluation_config.test_data_path)

        y_test_df = pd.read_csv(model_evaluation_config.test_target_variable)
        y_test = y_test_df[model_evaluation_config.target_column]

        # Create ModelEvaluation object
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Get predictions
        y_pred, y_pred_proba = model_evaluation.predictions(model, X_test_transformed)

        # Save evaluation results
        model_evaluation.save_results(y_test, y_pred, y_pred_proba)

    except Exception as e:
        raise e
