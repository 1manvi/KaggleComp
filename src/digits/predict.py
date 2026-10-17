"""Prediction and Kaggle submission helpers."""
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict_logits(model, loader):
    """Run a model over a loader and return all logits as one [N, 10] tensor (on CPU)."""
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch   # works with or without labels
        outputs.append(model(x.to(device)).cpu())
    return torch.cat(outputs)


def make_submission(preds, path):
    """Save predictions in Kaggle format (ImageId,Label) and validate it."""
    preds = np.asarray(preds).astype(int)
    df = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds})

    assert list(df.columns) == ["ImageId", "Label"]
    assert len(df) == 28000, f"expected 28000 rows, got {len(df)}"
    assert df["Label"].between(0, 9).all()
    assert not df.isna().any().any()

    df.to_csv(path, index=False)
    print(f"Saved {len(df)} predictions to {path}")
    return df