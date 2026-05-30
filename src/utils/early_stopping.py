import math
from typing import Any, Dict, Optional


def parse_early_stopping(config: Dict[str, Any], default_monitor: str) -> Optional[Dict[str, Any]]:
    early = config.get("early_stopping")
    if not early:
        return None
    patience = int(early.get("patience", 0))
    if patience <= 0:
        return None
    return {
        "patience": patience,
        "monitor": str(early.get("monitor", default_monitor)),
        "min_delta": float(early.get("min_delta", 0.0)),
    }


def resolve_monitor_value(metrics: Dict[str, Any], monitor: str) -> Optional[float]:
    if metrics is None:
        return None
    name = monitor
    if name.startswith(("val_", "test_", "train_")):
        name = name.split("_", 1)[1]
    aliases = {
        "auc": "auc_macro",
        "f1": "f1_macro",
        "precision": "precision_macro",
        "recall": "recall_macro",
    }
    name = aliases.get(name, name)
    value = metrics.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_nan(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def is_improved(current: Optional[float], best: Optional[float], monitor: str, min_delta: float) -> bool:
    if is_nan(current):
        return False
    if is_nan(best):
        return True
    monitor_lower = monitor.lower()
    minimize = "loss" in monitor_lower or "mse" in monitor_lower
    if minimize:
        return current <= best - min_delta
    return current >= best + min_delta
