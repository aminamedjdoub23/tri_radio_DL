import timm


def build_vit_model(name: str = "vit_tiny_patch16_224", num_classes: int = 14, pretrained: bool = True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=1)

