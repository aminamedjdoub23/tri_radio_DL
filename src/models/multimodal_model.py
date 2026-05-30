import torch
from torch import nn

from src.models.text_model import TextEncoder
from src.models.transfer_model import ImageEncoder


class MultimodalFusionModel(nn.Module):
    """Fusion intermédiaire : concaténation embedding image + embedding texte."""

    def __init__(
        self,
        tfidf_dim: int,
        num_classes: int,
        embedding_dim: int = 128,
        image_encoder_name: str = "resnet18",
        pretrained_image: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            name=image_encoder_name,
            pretrained=pretrained_image,
        )
        self.text_encoder = TextEncoder(input_dim=tfidf_dim, embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, tfidf):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(tfidf)
        return self.classifier(torch.cat([image_features, text_features], dim=1))
