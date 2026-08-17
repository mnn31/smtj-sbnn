"""Dataset loading. Inputs stay real-valued (standard BNN practice: only
hidden activations are binarized); images are flattened and scaled to [0, 1].
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_LOADERS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
}


def load_flat(name: str = "mnist", train: bool = True) -> TensorDataset:
    """Return a TensorDataset of (flattened images [N, 784], labels [N])."""
    ds = _LOADERS[name](DATA_DIR, train=train, download=True,
                        transform=transforms.ToTensor())
    xs = torch.stack([x for x, _ in ds]).reshape(len(ds), -1)
    ys = torch.tensor([y for _, y in ds])
    return TensorDataset(xs, ys)


def loaders(name: str = "mnist", batch_size: int = 256):
    train = DataLoader(load_flat(name, True), batch_size=batch_size,
                       shuffle=True, drop_last=True)
    test = DataLoader(load_flat(name, False), batch_size=1024)
    return train, test
