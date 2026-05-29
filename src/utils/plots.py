from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


@torch.no_grad()
def save_reconstruction_examples(model, loader, device, output_path, max_images: int = 6):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    images, _ = next(iter(loader))
    images = images[:max_images].to(device)
    recon = model(images).cpu().numpy()
    originals = images.cpu().numpy()

    n_images = len(originals)
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    if n_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx in range(n_images):
        axes[0, idx].imshow(originals[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title("Original")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(recon[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, idx].set_title("Reconstruit")
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path
