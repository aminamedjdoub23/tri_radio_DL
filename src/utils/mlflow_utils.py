from pathlib import Path
import re

import mlflow


def setup_mlflow(config):
    mlflow.set_tracking_uri(config["paths"]["mlflow_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_config(config_path: str):
    path = Path(config_path)
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path="config")


def log_metrics_dict(metrics, prefix: str = "", step=None):
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", float(value), step=step)


def safe_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\- /]", "_", str(name)).replace(" ", "_")


def log_auc_per_class(auc_per_class, class_names, prefix: str = "auc"):
    names = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
    for name, auc in zip(names, auc_per_class):
        mlflow.log_metric(f"{prefix}_{safe_metric_name(name)}", float(auc))
