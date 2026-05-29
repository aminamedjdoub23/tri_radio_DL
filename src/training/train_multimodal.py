import argparse
import math
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.openi_dataset import OpenIDataset
from src.data.transforms import get_rgb_transforms
from src.models.multimodal_model import MultimodalFusionModel
from src.models.transfer_model import ImageEncoder
from src.training.evaluate import evaluate_multilabel
from src.utils.mlflow_utils import log_config, log_metrics_dict, setup_mlflow
from src.utils.seed import get_device, set_seed


class ImageOnlyOpenI(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = ImageEncoder(embedding_dim=128, pretrained=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, image):
        return self.classifier(self.encoder(image))


def train_epoch(model, loader, optimizer, criterion, device, mode):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"{mode} train", leave=False):
        labels = batch["labels"].to(device).float()
        optimizer.zero_grad()
        if mode == "multimodal":
            logits = model(batch["image"].to(device), batch["tfidf"].to(device).float())
        else:
            logits = model(batch["image"].to(device))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def make_openi_loaders(config):
    csv_path = config["paths"]["openi_csv"]
    label_columns = config["openi"]["label_columns"]
    if not label_columns:
        raise ValueError("Renseigner openi.label_columns avant l'entraînement multimodal.")

    df = pd.read_csv(csv_path)
    csv_parent = Path(csv_path).parent
    image_column = config["openi"]["image_column"]
    df[image_column] = df[image_column].apply(
        lambda p: str(Path(p)) if Path(str(p)).is_absolute() else str((csv_parent / str(p)).resolve())
    )
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config["seed"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config["seed"])

    tmp_dir = Path(config["paths"]["output_dir"]) / "openi_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for name, frame in {"train": train_df, "val": val_df, "test": test_df}.items():
        path = tmp_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        split_paths[name] = path

    vectorizer = TfidfVectorizer(max_features=config["openi"]["max_tfidf_features"], min_df=2)
    x_train = vectorizer.fit_transform(train_df[config["openi"]["text_column"]].fillna(""))
    x_val = vectorizer.transform(val_df[config["openi"]["text_column"]].fillna(""))
    x_test = vectorizer.transform(test_df[config["openi"]["text_column"]].fillna(""))

    transforms = {
        "train": get_rgb_transforms(config["openi"]["image_size"], train=True),
        "val": get_rgb_transforms(config["openi"]["image_size"], train=False),
        "test": get_rgb_transforms(config["openi"]["image_size"], train=False),
    }
    matrices = {"train": x_train, "val": x_val, "test": x_test}
    datasets = {
        split: OpenIDataset(
            split_paths[split],
            image_column=config["openi"]["image_column"],
            text_column=config["openi"]["text_column"],
            label_columns=label_columns,
            transform=transforms[split],
            tfidf_matrix=matrices[split],
        )
        for split in ["train", "val", "test"]
    }
    loaders = {
        split: DataLoader(ds, batch_size=config["openi"]["batch_size"], shuffle=(split == "train"))
        for split, ds in datasets.items()
    }
    return loaders, vectorizer


def selection_metric(metrics):
    auc = metrics.get("auc_macro", float("nan"))
    if math.isnan(auc):
        return metrics.get("f1_macro", -1.0)
    return auc


def run_model(model, mode, loaders, config, device, vectorizer=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["openi"]["lr"])
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = -1.0
    best_path = Path(config["paths"]["output_dir"]) / f"best_openi_{mode}.pt"

    for epoch in range(config["openi"]["epochs"]):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device, mode)
        val_metrics, _, _, _ = evaluate_multilabel(model, loaders["val"], device, threshold=0.5, input_mode=mode)
        mlflow.log_metric(f"{mode}_train_loss", train_loss, step=epoch)
        log_metrics_dict({f"{mode}_val_{k}": v for k, v in val_metrics.items()}, step=epoch)
        current_score = selection_metric(val_metrics)
        if current_score > best_val_auc:
            best_val_auc = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "label_columns": config["openi"]["label_columns"],
                    "tfidf_dim": len(vectorizer.vocabulary_) if vectorizer is not None else None,
                },
                best_path,
            )

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    test_metrics, _, _, _ = evaluate_multilabel(model, loaders["test"], device, threshold=0.5, input_mode=mode)
    log_metrics_dict({f"{mode}_test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_artifact(str(best_path), artifact_path="models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config)
    setup_mlflow(config)
    Path(config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    loaders, vectorizer = make_openi_loaders(config)

    with mlflow.start_run(run_name="openi_image_vs_multimodal"):
        mlflow.log_params({
            "dataset": "OpenI prepared CSV",
            "comparison": "image_only_vs_multimodal",
            "fusion": "intermediate_concat",
            "tfidf_features": len(vectorizer.vocabulary_),
            "seed": config["seed"],
        })
        log_config(args.config)
        num_classes = len(config["openi"]["label_columns"])

        image_model = ImageOnlyOpenI(num_classes=num_classes).to(device)
        run_model(image_model, "image", loaders, config, device)

        multimodal_model = MultimodalFusionModel(
            tfidf_dim=len(vectorizer.vocabulary_),
            num_classes=num_classes,
            pretrained_image=True,
        ).to(device)
        run_model(multimodal_model, "multimodal", loaders, config, device, vectorizer=vectorizer)
        vectorizer_path = Path(config["paths"]["output_dir"]) / "openi_tfidf_vectorizer.joblib"
        dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(str(vectorizer_path), artifact_path="models")


if __name__ == "__main__":
    main()
