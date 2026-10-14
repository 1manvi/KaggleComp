from digits.data import make_loaders
import matplotlib.pyplot as plt
import numpy as np

train_loader, val_loader, (mean, std) = make_loaders("../data/train.csv")
print(f"mean={mean:.4f}, std={std:.4f}")
print("train batches:", len(train_loader), "| val batches:", len(val_loader))

images, labels = next(iter(train_loader))
print("batch shape:", images.shape, "| labels shape:", labels.shape)
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for ax, img, lbl in zip(axes.flat, images, labels):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(int(lbl)); ax.axis("off")
plt.show()

print("val class counts:", np.bincount(val_loader.dataset.labels.numpy()))