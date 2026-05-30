import torch
from torch import nn
from torchvision import models


def build_transfer_model(name: str = "resnet18", num_classes: int = 14, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                model.features.conv0.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise ValueError(f"Modele non supporte: {name}")


class ImageEncoder(nn.Module):
    """Encodeur image réutilisé pour la preuve de concept multimodale."""

    def __init__(self, embedding_dim: int = 128, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        name = name.lower()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Encodeur image non supporte: {name}")
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(in_features, embedding_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))
