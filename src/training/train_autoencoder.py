import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from src.data.chestmnist_dataset import build_chestmnist_loaders, build_normal_autoencoder_loaders
from src.models.autoencoder import ConvAutoencoder
from src.training.evaluate import reconstruction_errors
from src.utils.early_stopping import is_improved, parse_early_stopping
from src.utils.mlflow_utils import log_config, setup_mlflow
from src.utils.plots import save_reconstruction_examples
from src.utils.seed import get_device, set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, _ in tqdm(loader, desc="ae train", leave=False):
        images = images.to(device)
        optimizer.zero_grad()
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config)
    setup_mlflow(config)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_normal_autoencoder_loaders(config)
    full_loaders = build_chestmnist_loaders(config, ae=True)
    model = ConvAutoencoder(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["autoencoder"]["lr"])
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_path = output_dir / "best_autoencoder.pt"
    early = parse_early_stopping(config, default_monitor="val_reconstruction_mse")
    if early:
        monitor_name = early["monitor"].lower()
        if "loss" not in monitor_name and "mse" not in monitor_name:
            early = None
    early_best = None
    early_bad_epochs = 0

    with mlflow.start_run(run_name="chestmnist_autoencoder"):
        mlflow.log_params({
            "model": "ConvAutoencoder",
            "dataset": "ChestMNIST normal-only subset",
            "normal_strategy": config["autoencoder"]["normal_strategy"],
            "anomaly_score": "mean_squared_reconstruction_error",
            "threshold_rule": f"validation_p{config['autoencoder']['anomaly_percentile']}",
            "seed": config["seed"],
        })
        log_config(args.config)

        epochs = config["autoencoder"]["epochs"]
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            val_errors = reconstruction_errors(model, loaders["val"], device)
            val_loss = float(np.mean(val_errors))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_reconstruction_mse", val_loss, step=epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)
            if early:
                if is_improved(val_loss, early_best, early["monitor"], early["min_delta"]):
                    early_best = val_loss
                    early_bad_epochs = 0
                else:
                    early_bad_epochs += 1
                if early_bad_epochs >= early["patience"]:
                    break

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_errors = reconstruction_errors(model, loaders["val"], device)
        test_normal_errors = reconstruction_errors(model, loaders["test"], device)
        test_all_errors = reconstruction_errors(model, full_loaders["test"], device)
        threshold = float(np.percentile(val_errors, config["autoencoder"]["anomaly_percentile"]))
        checkpoint["anomaly_threshold"] = threshold
        torch.save(checkpoint, best_path)
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_metric("test_normal_reconstruction_mse_mean", float(np.mean(test_normal_errors)))
        mlflow.log_metric("test_all_reconstruction_mse_mean", float(np.mean(test_all_errors)))
        mlflow.log_metric("test_all_anomaly_rate", float(np.mean(np.asarray(test_all_errors) >= threshold)))
        recon_fig = save_reconstruction_examples(
            model,
            full_loaders["test"],
            device,
            output_dir / "autoencoder_reconstructions.png",
        )
        mlflow.log_artifact(str(recon_fig), artifact_path="figures")
        mlflow.log_artifact(str(best_path), artifact_path="models")


if __name__ == "__main__":
    main()
