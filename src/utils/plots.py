from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def save_auc_bar(auc_per_class, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_names)), auc_per_class)
    plt.xticks(range(len(class_names)), list(class_names.values()) if isinstance(class_names, dict) else class_names, rotation=90)
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_roc_curves(y_true, y_prob, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    names = list(class_names.values()) if isinstance(class_names, dict) else class_names
    for idx, name in enumerate(names):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        RocCurveDisplay.from_predictions(y_true[:, idx], y_prob[:, idx], name=name, ax=plt.gca())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

