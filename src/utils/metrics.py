import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def multilabel_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        auc_per_class = roc_auc_score(y_true, y_prob, average=None)
    except ValueError:
        metrics["auc_macro"] = float("nan")
        auc_per_class = np.full(y_true.shape[1], np.nan)
    return metrics, auc_per_class

