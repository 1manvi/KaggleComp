"""Plots for training curves and error analysis."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_history(histories, save_path=None):
    """Loss and accuracy curves. histories = {"model name": history_dict, ...}"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        line, = axes[0].plot(epochs, h["train_loss"], "--", label=f"{name} (train)")
        axes[0].plot(epochs, h["val_loss"], color=line.get_color(), label=f"{name} (val)")
        axes[1].plot(epochs, h["train_acc"], "--", color=line.get_color())
        axes[1].plot(epochs, h["val_acc"], color=line.get_color(), label=name)
    axes[0].set(title="Loss (dashed = train, solid = val)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Accuracy (dashed = train, solid = val)", xlabel="Epoch", ylabel="Accuracy",
                ylim=(0.94, 1.0))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion(y_true, y_pred, save_path=None):
    """Confusion matrix: rows = true digit, columns = predicted digit."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (validation set)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return cm


def plot_misclassified(images, y_true, y_pred, n=16, save_path=None):
    """Grid of misclassified digits with true and predicted labels."""
    wrong = np.where(y_true != y_pred)[0][:n]
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).flat, wrong):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"true {y_true[i]} / pred {y_pred[i]}", fontsize=8, color="red")
    fig.suptitle(f"Misclassified validation digits (showing {len(wrong)})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()