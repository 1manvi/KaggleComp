"""Model architectures for the Digit Recognizer: MLP baseline and CNNs."""
from torch import nn


class MLP(nn.Module):
    """Baseline: a plain fully-connected network on flattened pixels."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # [B, 1, 28, 28] -> [B, 784]
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),           # 10 logits, one per digit
        )

    def forward(self, x):
        return self.net(x)


def conv_block(in_ch, out_ch, batchnorm=False):
    """Conv -> (BN) -> ReLU -> Conv -> (BN) -> ReLU -> MaxPool. Halves H and W."""
    layers = []
    for c_in in (in_ch, out_ch):
        layers.append(nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    """Two convolutional blocks + a small classifier head.

    CNN()                             -> plain CNN
    CNN(batchnorm=True, dropout=0.4)  -> improved CNN (BN + dropout)
    """

    def __init__(self, batchnorm=False, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, batchnorm),    # [B, 1, 28, 28]  -> [B, 32, 14, 14]
            conv_block(32, 64, batchnorm),   # [B, 32, 14, 14] -> [B, 64, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # -> [B, 64*7*7] = [B, 3136]
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)