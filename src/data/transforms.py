from torchvision import transforms


def get_chestmnist_transforms(image_size: int, train: bool):
    """Transformations simples : resize, légère augmentation train, normalisation ImageNet."""
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
    return transforms.Compose(steps)


def get_rgb_transforms(image_size: int, train: bool):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


def get_autoencoder_transforms(image_size: int, train: bool):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [transforms.ToTensor()]
    return transforms.Compose(steps)

