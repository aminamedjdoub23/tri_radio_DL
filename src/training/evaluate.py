import torch
from torch import nn

from src.utils.metrics import multilabel_metrics


@torch.no_grad()
def evaluate_multilabel(model, loader, device, threshold: float = 0.5, input_mode: str = "image"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, y_true, y_prob = [], [], []

    for batch in loader:
        if isinstance(batch, dict):
            labels = batch["labels"].to(device).float()
            if input_mode == "text":
                logits = model(batch["tfidf"].to(device).float())
            elif input_mode == "multimodal":
                logits = model(batch["image"].to(device), batch["tfidf"].to(device).float())
            else:
                logits = model(batch["image"].to(device))
        else:
            images, labels = batch
            labels = labels.to(device).float()
            logits = model(images.to(device))

        loss = criterion(logits, labels)
        losses.append(loss.item())
        y_true.append(labels.cpu())
        y_prob.append(torch.sigmoid(logits).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    metrics, auc_per_class = multilabel_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics, auc_per_class, y_true, y_prob


@torch.no_grad()
def reconstruction_errors(model, loader, device):
    model.eval()
    errors = []
    for images, _ in loader:
        images = images.to(device)
        recon = model(images)
        batch_errors = ((images - recon) ** 2).mean(dim=(1, 2, 3))
        errors.extend(batch_errors.cpu().numpy().tolist())
    return errors

