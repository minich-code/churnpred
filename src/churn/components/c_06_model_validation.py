from pathlib import Path
from src.churn.utils.commons import save_json

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, 
                             classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize

from src.churn.entity.config_entity import ModelValidationConfig

import mlflow
import dagshub
mlflow.set_tracking_uri('https://dagshub.com/minich-code/churnpred.mlflow')
dagshub.init(repo_owner='minich-code', repo_name='churnpred', mlflow=True)


# Class to handle model validation
class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config

    def predictions(self, model, X_test_transformed):
        # Make predictions
        y_pred = model.predict(X_test_transformed)
        y_pred_proba = model.predict_proba(X_test_transformed)  # For ROC-AUC and PR-AUC
        return y_pred, y_pred_proba

    def model_validation(self, y_test, y_pred, y_pred_proba):
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        pr_auc = average_precision_score(y_test, y_pred_proba, average='weighted')

        # Print detailed classification report and confusion matrix
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
        
        # Return evaluation metrics
        return accuracy, precision, recall, f1, roc_auc, pr_auc

    def plot_roc_curve(self, y_test, y_pred_proba):
        # Binarize the output for multi-class ROC curve plotting
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
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

    def plot_pr_curve(self, y_test, y_pred_proba):
        # Binarize the output for multi-class Precision-Recall curve plotting
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
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

    def save_results(self, y_test, y_pred, y_pred_proba):
        # Calculate and save evaluation metrics
        accuracy, precision, recall, f1, roc_auc, pr_auc = self.model_validation(y_test, y_pred, y_pred_proba)

        scores = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

        # Plot ROC and Precision-Recall curves
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_pr_curve(y_test, y_pred_proba)

        # Log metrics and artifacts in MLflow
        with mlflow.start_run(run_name="Model_Validation"):
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("ROC_AUC", roc_auc)
            mlflow.log_metric("PR_AUC", pr_auc)

            # Log artifacts
            mlflow.log_artifact(os.path.join(
                self.config.root_dir, 'roc_curve.png'))
            mlflow.log_artifact(os.path.join(
                self.config.root_dir, 'pr_curve.png'))
            
            # Log model parameters
            mlflow.log_params(self.config.all_params)

