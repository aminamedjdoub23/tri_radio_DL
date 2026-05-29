from torch import nn


class ConvAutoencoder(nn.Module):
    """AE convolutionnel simple pour erreur de reconstruction."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        reconstruction = self.decoder(self.encoder(x))
        return reconstruction[..., : x.shape[-2], : x.shape[-1]]

