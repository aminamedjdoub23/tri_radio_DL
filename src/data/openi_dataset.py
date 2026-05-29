from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenIDataset(Dataset):
    """Dataset CSV minimal pour une preuve de concept OpenI image + texte."""

    def __init__(
        self,
        csv_path: str,
        image_column: str,
        text_column: str,
        label_columns: List[str],
        transform=None,
        tfidf_matrix: Optional[object] = None,
    ):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.image_column = image_column
        self.text_column = text_column
        self.label_columns = label_columns
        self.transform = transform
        self.tfidf_matrix = tfidf_matrix
        if not label_columns:
            raise ValueError("Renseigner openi.label_columns dans config.yaml.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = Path(row[self.image_column])
        if not image_path.is_absolute():
            image_path = self.csv_path.parent / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_columns].astype("float32").values)
        text = str(row[self.text_column]) if not pd.isna(row[self.text_column]) else ""

        sample = {"image": image, "text": text, "labels": labels}
        if self.tfidf_matrix is not None:
            sample["tfidf"] = torch.tensor(self.tfidf_matrix[index].toarray().squeeze(0), dtype=torch.float32)
        return sample

