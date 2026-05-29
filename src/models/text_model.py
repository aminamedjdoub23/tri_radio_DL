from torch import nn


class TextMLP(nn.Module):
    """Classifieur texte sur vecteurs TF-IDF."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)

