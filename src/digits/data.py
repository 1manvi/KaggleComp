"""Data loading for the Kaggle Digit Recognizer competition."""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DigitDataset(Dataset):
    """Holds digit images as normalised tensors of shape [1, 28, 28]."""

    def __init__(self, pixels, labels=None, mean=0.0, std=1.0, transform=None):
        x = pixels.astype(np.float32) / 255.0          # 0–255  ->  0–1
        x = (x - mean) / std                            # centre around 0, unit spread
        self.images = torch.tensor(x).view(-1, 1, 28, 28)  # 784 flat -> [N, 1, 28, 28]
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:          # test set: no labels
            return img
        return img, self.labels[idx]


def make_loaders(train_csv, val_size=0.1, batch_size=128, seed=42, train_transform=None):
    """Return (train_loader, val_loader, (mean, std)).

    If val_size == 0, uses all data for training and val_loader is None
    (for the final retrain before submitting).
    """
    df = pd.read_csv(train_csv)
    y = df["label"].values
    X = df.drop(columns="label").values

    if val_size > 0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=seed
        )
    else:
        X_tr, y_tr, X_val, y_val = X, y, None, None

    # Normalisation stats from the TRAINING split only
    mean = float(X_tr.mean() / 255.0)
    std = float(X_tr.std() / 255.0)

    train_ds = DigitDataset(X_tr, y_tr, mean, std, transform=train_transform)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    val_loader = None
    if X_val is not None:
        val_ds = DigitDataset(X_val, y_val, mean, std)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, (mean, std)


def make_test_loader(test_csv, mean, std, batch_size=256):
    """Loader for Kaggle's test.csv (no labels), in original row order."""
    X_test = pd.read_csv(test_csv).values
    test_ds = DigitDataset(X_test, labels=None, mean=mean, std=std)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)