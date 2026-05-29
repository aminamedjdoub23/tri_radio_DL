import argparse
import math
from pathlib import Path

import mlflow
import torch
import yaml
from torch import nn
from tqdm import tqdm

from src.data.chestmnist_dataset import CHESTMNIST_LABELS, NUM_CLASSES, build_chestmnist_loaders
from src.models.simple_cnn import SimpleCNN
from src.models.transfer_model import build_transfer_model
from src.models.vit_model import build_vit_model
from src.training.evaluate import evaluate_multilabel
from src.utils.mlflow_utils import log_auc_per_class, log_config, log_metrics_dict, setup_mlflow
from src.utils.plots import save_auc_bar, save_roc_curves
from src.utils.seed import get_device, set_seed


def build_model(name, config):
    if name == "simple_cnn":
        return SimpleCNN(num_classes=NUM_CLASSES, in_channels=1)
    if name == "transfer":
        return build_transfer_model(
            name=config["models"]["transfer_name"],
            num_classes=NUM_CLASSES,
            pretrained=config["models"]["pretrained"],
        )
    if name == "vit":
        return build_vit_model(
            name=config["models"]["vit_name"],
            num_classes=NUM_CLASSES,
            pretrained=config["models"]["pretrained"],
        )
    raise ValueError(f"Modèle inconnu: {name}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def selection_metric(metrics):
    auc = metrics.get("auc_macro", float("nan"))
    if math.isnan(auc):
        return metrics.get("f1_macro", -1.0)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple_cnn", "transfer", "vit"], required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config)
    setup_mlflow(config)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_chestmnist_loaders(config)
    model = build_model(args.model, config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["chestmnist"]["lr"],
        weight_decay=config["chestmnist"]["weight_decay"],
    )

    best_val_auc = -1.0
    best_path = output_dir / f"best_{args.model}.pt"

    with mlflow.start_run(run_name=f"chestmnist_{args.model}"):
        mlflow.log_params({
            "model": args.model,
            "dataset": "ChestMNIST",
            "loss": "BCEWithLogitsLoss",
            "activation_eval": "sigmoid",
            "image_size": config["chestmnist"]["size"],
            "batch_size": config["chestmnist"]["batch_size"],
            "lr": config["chestmnist"]["lr"],
            "seed": config["seed"],
        })
        log_config(args.config)

        for epoch in range(config["chestmnist"]["epochs"]):
            train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            val_metrics, val_auc_per_class, _, _ = evaluate_multilabel(
                model, loaders["val"], device, threshold=config["chestmnist"]["threshold"]
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            log_metrics_dict({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            current_score = selection_metric(val_metrics)
            if current_score > best_val_auc:
                best_val_auc = current_score
                torch.save({"model_state_dict": model.state_dict(), "config": config, "model_name": args.model}, best_path)

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_auc_per_class, y_true, y_prob = evaluate_multilabel(
            model, loaders["test"], device, threshold=config["chestmnist"]["threshold"]
        )
        log_metrics_dict({f"test_{k}": v for k, v in test_metrics.items()})
        log_auc_per_class(test_auc_per_class, CHESTMNIST_LABELS, prefix="test_auc")

        auc_fig = save_auc_bar(test_auc_per_class, CHESTMNIST_LABELS, output_dir / f"{args.model}_auc_per_class.png")
        roc_fig = save_roc_curves(y_true, y_prob, CHESTMNIST_LABELS, output_dir / f"{args.model}_roc_curves.png")
        mlflow.log_artifact(str(auc_fig), artifact_path="figures")
        mlflow.log_artifact(str(roc_fig), artifact_path="figures")
        mlflow.log_artifact(str(best_path), artifact_path="models")


if __name__ == "__main__":
    main()
