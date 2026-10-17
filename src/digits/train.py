"""Training and evaluation loops."""
import random
import time

import numpy as np
import torch
from torch import nn


def set_seed(seed=42):
    """Make runs reproducible (same weights init, same shuffling)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Use a GPU if available: CUDA (Colab/Kaggle), MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device, scheduler=None):
    """One full pass over the training data. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()          # 1. clear old gradients
        logits = model(x)              # 2. forward pass: predictions
        loss = loss_fn(logits, y)      # 3. how wrong are we?
        loss.backward()                # 4. backprop: compute gradients
        optimizer.step()               # 5. update the weights
        if scheduler is not None:
            scheduler.step()           # adjust learning rate (per batch)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Loss and accuracy on a labelled loader, without updating weights."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader=None, epochs=10, lr=1e-3,
        use_scheduler=False, weight_decay=0.0, device=None):
    """Train a model and return its history (for plotting curves).

    If val_loader is given, the weights from the best validation epoch
    are restored at the end.
    """
    device = device or get_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        msg = f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f}"

        if val_loader is not None:
            va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            msg += f" | val loss {va_loss:.4f} acc {va_acc:.4f}"
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(msg + f" | {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (val acc {best_acc:.4f})")
    return history