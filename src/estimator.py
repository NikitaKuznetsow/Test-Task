import pandas as pd
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
)


from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
from typing import Union
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc



class ModelEvaluator:
    """
    A class for evaluating classification models.

    Args:
        model (ClassifierMixin): An instance of a scikit-learn classifier.
    """

    def __init__(self, model: ClassifierMixin):
        self.model = model

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        cut_off: float = 0.5,
        plot_graph: bool = True,
    ):
        """
        Fit the model, calculate metrics, and optionally plot graphs.

        Args:
            X_train (pd.DataFrame): The training feature matrix.
            X_test (pd.DataFrame): The testing feature matrix.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
            cut_off (float, optional): Decision threshold for converting predicted probabilities into class labels.
                Default is 0.5.
            plot_graph (bool, optional): Whether to plot ROC and precision-recall curves. Default is True.
        """

        self.model.fit(X_train, y_train)
        y_proba_test = self.model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test > cut_off).astype("int16")

        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_proba_train > cut_off).astype("int16")

        roc_auc_train = roc_auc_score(y_train, y_proba_train)
        gini_train = 2 * roc_auc_train - 1
        f1_train = f1_score(y_train, y_pred_train)
        logloss_train = log_loss(y_train, y_pred_train)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)
        gini_test = 2 * roc_auc_test - 1
        f1_test = f1_score(y_test, y_pred_test)
        logloss_test = log_loss(y_test, y_pred_test)

        print("Metrics:", "  train", "test")
        print("ROC_AUC:  ", round(roc_auc_train, 3), round(roc_auc_test, 3))
        print("Gini:     ", round(gini_train, 3), round(gini_test, 3))
        print("F1_score: ", round(f1_train, 3), round(f1_test, 3))
        print("Log_loss: ", round(logloss_train, 3), round(logloss_test, 3))

        if plot_graph:
            self._plot_curves(X_train, X_test, y_train, y_test)
        return self.model


    def _plot_curves(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """
        Plot ROC and Precision-Recall curves for the model.

        Args:
            X_train (pd.DataFrame): The training feature matrix.
            X_test (pd.DataFrame): The testing feature matrix.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
        """

        # Calculate ROC curves and AUC scores
        y_prob_train = self.model.predict_proba(X_train)[:, 1]
        y_prob_test = self.model.predict_proba(X_test)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

        roc_auc_train = roc_auc_score(y_train, y_prob_train)
        roc_auc_test = roc_auc_score(y_test, y_prob_test)

        # Calculate Precision-Recall curves and AUC scores
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_prob_train)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_prob_test)

        pr_auc_train = auc(recall_train, precision_train)
        pr_auc_test = auc(recall_test, precision_test)

        # Plot ROC curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC Curve (AUC = {roc_auc_train:.2f})')
        plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Test ROC Curve (AUC = {roc_auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Plot Precision-Recall curves
        plt.subplot(1, 2, 2)
        plt.plot(recall_train, precision_train, color='blue', lw=2, label=f'Train PR Curve (AUC = {pr_auc_train:.2f})')
        plt.plot(recall_test, precision_test, color='green', lw=2, label=f'Test PR Curve (AUC = {pr_auc_test:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        plt.tight_layout()
        plt.show()










