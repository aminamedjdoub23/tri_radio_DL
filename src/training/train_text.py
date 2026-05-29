import argparse
import math
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.text_model import TextMLP
from src.training.evaluate import evaluate_multilabel
from src.utils.mlflow_utils import log_config, log_metrics_dict, setup_mlflow
from src.utils.seed import get_device, set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="text train", leave=False):
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_text_tensor(model, loader, device, threshold):
    wrapped = []
    for x, y in loader:
        wrapped.append({"tfidf": x, "labels": y})
    return evaluate_multilabel(model, wrapped, device, threshold=threshold, input_mode="text")


def selection_metric(metrics):
    auc = metrics.get("auc_macro", float("nan"))
    if math.isnan(auc):
        return metrics.get("f1_macro", -1.0)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    label_columns = config["openi"]["label_columns"]
    if not label_columns:
        raise ValueError("Renseigner openi.label_columns avant l'entraînement texte.")

    set_seed(config["seed"])
    device = get_device(config)
    setup_mlflow(config)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config["paths"]["openi_csv"])
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config["seed"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config["seed"])

    vectorizer = TfidfVectorizer(max_features=config["openi"]["max_tfidf_features"], min_df=2)
    x_train = vectorizer.fit_transform(train_df[config["openi"]["text_column"]].fillna(""))
    x_val = vectorizer.transform(val_df[config["openi"]["text_column"]].fillna(""))
    x_test = vectorizer.transform(test_df[config["openi"]["text_column"]].fillna(""))

    def make_loader(x, frame, shuffle):
        dataset = TensorDataset(
            torch.tensor(x.toarray(), dtype=torch.float32),
            torch.tensor(frame[label_columns].values, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=config["openi"]["batch_size"], shuffle=shuffle)

    loaders = {
        "train": make_loader(x_train, train_df, True),
        "val": make_loader(x_val, val_df, False),
        "test": make_loader(x_test, test_df, False),
    }

    model = TextMLP(input_dim=x_train.shape[1], num_classes=len(label_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["openi"]["lr"])
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = -1.0
    best_path = output_dir / "best_openi_text.pt"

    with mlflow.start_run(run_name="openi_text_tfidf_mlp"):
        mlflow.log_params({
            "dataset": "OpenI prepared CSV",
            "model": "TFIDF + MLP",
            "tfidf_features": x_train.shape[1],
            "fusion": "none_text_only",
            "seed": config["seed"],
        })
        log_config(args.config)

        for epoch in range(config["openi"]["epochs"]):
            train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            val_metrics, _, _, _ = evaluate_text_tensor(model, loaders["val"], device, threshold=0.5)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            log_metrics_dict({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            current_score = selection_metric(val_metrics)
            if current_score > best_val_auc:
                best_val_auc = current_score
                torch.save({"model_state_dict": model.state_dict(), "vocabulary": vectorizer.vocabulary_}, best_path)

        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
        test_metrics, _, _, _ = evaluate_text_tensor(model, loaders["test"], device, threshold=0.5)
        log_metrics_dict({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(best_path), artifact_path="models")


if __name__ == "__main__":
    main()
