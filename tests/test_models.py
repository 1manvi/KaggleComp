import pytest
import torch

from digits.models import CNN, MLP


@pytest.mark.parametrize("model", [MLP(), CNN(), CNN(batchnorm=True, dropout=0.4)])
def test_model_returns_logits_of_shape_B_by_10(model):
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)