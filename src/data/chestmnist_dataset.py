from typing import Dict

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader, Subset

from src.data.transforms import get_autoencoder_transforms, get_chestmnist_transforms


CHESTMNIST_LABELS = INFO["chestmnist"]["label"]
NUM_CLASSES = len(CHESTMNIST_LABELS)


def get_chestmnist_dataset(split: str, root: str, image_size: int, download: bool = True, ae: bool = False):
    """Charge ChestMNIST avec les splits officiels MedMNIST."""
    transform = (
        get_autoencoder_transforms(image_size, train=split == "train")
        if ae
        else get_chestmnist_transforms(image_size, train=split == "train")
    )
    dataset_class = getattr(medmnist, INFO["chestmnist"]["python_class"])
    return dataset_class(split=split, root=root, transform=transform, download=download, size=image_size)


def build_chestmnist_loaders(config: Dict, ae: bool = False):
    data_root = config["paths"]["data_root"]
    image_size = config["chestmnist"]["size"]
    batch_size = config["autoencoder"]["batch_size"] if ae else config["chestmnist"]["batch_size"]
    num_workers = config["chestmnist"]["num_workers"]

    datasets = {
        split: get_chestmnist_dataset(split, data_root, image_size, ae=ae)
        for split in ["train", "val", "test"]
    }
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split, ds in datasets.items()
    }


def build_normal_autoencoder_loaders(config: Dict):
    """Pour l'AE, on entraîne par défaut sur les images sans label positif."""
    loaders = build_chestmnist_loaders(config, ae=True)
    filtered = {}
    for split, loader in loaders.items():
        labels = loader.dataset.labels
        normal_indices = [i for i, y in enumerate(labels) if y.sum() == 0]
        dataset = Subset(loader.dataset, normal_indices)
        filtered[split] = DataLoader(
            dataset,
            batch_size=config["autoencoder"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["chestmnist"]["num_workers"],
        )
    return filtered

