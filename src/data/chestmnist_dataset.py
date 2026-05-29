from typing import Dict
from pathlib import Path

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader, Subset

from src.data.transforms import get_autoencoder_transforms, get_chestmnist_transforms


CHESTMNIST_LABELS = INFO["chestmnist"]["label"]
NUM_CLASSES = len(CHESTMNIST_LABELS)


def get_chestmnist_dataset(
    split: str,
    root: str,
    image_size: int,
    medmnist_size: int = 64,
    download: bool = True,
    ae: bool = False,
):
    """Charge ChestMNIST avec les splits officiels MedMNIST."""
    Path(root).mkdir(parents=True, exist_ok=True)
    transform = (
        get_autoencoder_transforms(image_size, train=split == "train")
        if ae
        else get_chestmnist_transforms(image_size, train=split == "train")
    )
    dataset_class = getattr(medmnist, INFO["chestmnist"]["python_class"])
    return dataset_class(split=split, root=root, transform=transform, download=download, size=medmnist_size)


def build_chestmnist_loaders(config: Dict, ae: bool = False):
    data_root = config["paths"]["data_root"]
    image_size = config["chestmnist"]["size"]
    medmnist_size = config["chestmnist"].get("medmnist_size", image_size)
    batch_size = config["autoencoder"]["batch_size"] if ae else config["chestmnist"]["batch_size"]
    num_workers = config["chestmnist"]["num_workers"]

    datasets = {
        split: get_chestmnist_dataset(split, data_root, image_size, medmnist_size=medmnist_size, ae=ae)
        for split in ["train", "val", "test"]
    }
    max_samples = config["chestmnist"].get("max_samples", {})
    for split, dataset in list(datasets.items()):
        limit = max_samples.get(split)
        if limit:
            datasets[split] = Subset(dataset, list(range(min(limit, len(dataset)))))
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split, ds in datasets.items()
    }


def build_normal_autoencoder_loaders(config: Dict):
    """Pour l'AE, on entraîne par défaut sur les images sans label positif."""
    loaders = build_chestmnist_loaders(config, ae=True)
    filtered = {}
    for split, loader in loaders.items():
        base_dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        labels = base_dataset.labels
        valid_indices = set(loader.dataset.indices) if isinstance(loader.dataset, Subset) else None
        normal_indices = [i for i, y in enumerate(labels) if y.sum() == 0]
        if valid_indices is not None:
            normal_indices = [i for i in normal_indices if i in valid_indices]
        dataset = Subset(base_dataset, normal_indices)
        filtered[split] = DataLoader(
            dataset,
            batch_size=config["autoencoder"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["chestmnist"]["num_workers"],
        )
    return filtered
