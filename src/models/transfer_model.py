import torch
from torch import nn
from torchvision import models


def build_transfer_model(name: str = "resnet18", num_classes: int = 14, pretrained: bool = True):
    if name != "resnet18":
        raise ValueError("Ce squelette garde resnet18 pour rester simple et défendable.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ImageEncoder(nn.Module):
    """Encodeur image réutilisé pour la preuve de concept multimodale."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(in_features, embedding_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))
