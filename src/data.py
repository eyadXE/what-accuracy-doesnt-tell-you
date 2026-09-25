"""Dataset loaders for in-distribution (ID) and out-of-distribution (OOD) data.

All loaders return (train_ds, val_ds, test_ds) for ID datasets, or a single
Dataset for OOD datasets. Normalization is parameterized so OOD data can be
matched to whichever ID model is consuming it (see `NORM` below and
`get_ood_dataset`).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

try:
    # Some dataset mirrors (e.g. KMNIST's) fail Python's bundled CA verification
    # on Windows. Use the OS trust store instead of disabling verification.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

DATA_ROOT = "./data"

# Per-ID-dataset normalization stats (mean, std), single-channel or 3-channel.
NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

# Native channel/size of each raw dataset (before any resize/gray/rgb conversion).
NATIVE = {
    "mnist": (1, 28),
    "fashion_mnist": (1, 28),
    "kmnist": (1, 28),
    "emnist_letters": (1, 28),
    "svhn": (3, 32),
    "cifar10": (3, 32),
    "cifar100": (3, 32),
}


def _to_channels_size(channels: int, size: int, grayscale_source: bool):
    """Build a transform list converting a raw PIL image to `channels`x`size`x`size`."""
    ops = [transforms.Resize((size, size))]
    if channels == 3 and grayscale_source:
        ops.append(transforms.Grayscale(num_output_channels=3))
    elif channels == 1 and not grayscale_source:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.ToTensor())
    return ops


def _norm_transform(id_name: str, channels: int):
    mean, std = NORM[id_name]
    if channels == 3 and len(mean) == 1:
        mean, std = mean * 3, std * 3
    if channels == 1 and len(mean) == 3:
        mean, std = (sum(mean) / 3,), (sum(std) / 3,)
    return transforms.Normalize(mean, std)


def get_id_dataset(name: str, val_frac: float = 0.1, seed: int = 0):
    """Return (train_ds, val_ds, test_ds) for an ID dataset, normalized for itself."""
    channels, size = NATIVE[name]
    tfm = transforms.Compose(
        _to_channels_size(channels, size, grayscale_source=(channels == 1))
        + [_norm_transform(name, channels)]
    )

    if name == "mnist":
        train_full = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
        test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "fashion_mnist":
        train_full = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "cifar10":
        train_full = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "cifar100":
        train_full = datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=tfm)
        test = datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown ID dataset: {name}")

    n_val = int(len(train_full) * val_frac)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=gen)
    return train_ds, val_ds, test


def _resolve_targets(ds) -> list[int]:
    """Recursively resolve integer targets for a (possibly Subset-wrapped) torchvision dataset."""
    if isinstance(ds, Subset):
        base = _resolve_targets(ds.dataset)
        return [base[i] for i in ds.indices]
    if hasattr(ds, "targets"):
        return [int(t) for t in ds.targets]
    if hasattr(ds, "labels"):
        return [int(t) for t in ds.labels]
    raise ValueError(f"Cannot resolve targets for dataset of type {type(ds)}")


class ClassSubset(Dataset):
    """Wraps a dataset, keeping only samples whose target is in `classes`, remapped to 0..len(classes)-1.

    Used for small illustrative few-class models (e.g. a 3-class logit-space figure).
    """

    def __init__(self, base_dataset, classes: list[int]):
        self.base = base_dataset
        self.classes = list(classes)
        self.class_map = {c: i for i, c in enumerate(self.classes)}
        targets = _resolve_targets(base_dataset)
        self.indices = [i for i, t in enumerate(targets) if t in self.class_map]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, t = self.base[self.indices[idx]]
        return img, self.class_map[int(t)]


def get_id_dataset_subset(name: str, classes: list[int], val_frac: float = 0.1, seed: int = 0):
    """Like `get_id_dataset` but restricted to a small set of classes, remapped to 0..k-1."""
    train_ds, val_ds, test_ds = get_id_dataset(name, val_frac=val_frac, seed=seed)
    return ClassSubset(train_ds, classes), ClassSubset(val_ds, classes), ClassSubset(test_ds, classes)


class NoiseDataset(Dataset):
    """Synthetic Gaussian or uniform noise images, normalized like `id_name`."""

    def __init__(self, kind: str, id_name: str, channels: int, size: int, n: int = 2000, seed: int = 0):
        assert kind in ("gaussian", "uniform")
        self.kind = kind
        self.n = n
        self.channels = channels
        self.size = size
        self.norm = _norm_transform(id_name, channels)
        self.gen = torch.Generator().manual_seed(seed)
        if kind == "gaussian":
            imgs = torch.clamp(torch.randn(n, channels, size, size, generator=self.gen) * 0.25 + 0.5, 0, 1)
        else:
            imgs = torch.rand(n, channels, size, size, generator=self.gen)
        self.imgs = imgs

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self.norm(self.imgs[idx])
        return img, -1


def get_ood_dataset(name: str, id_name: str, channels: int, size: int, n_max: int | None = None, seed: int = 0):
    """Return an OOD dataset, resized/recolored/normalized to match the ID model's input."""
    if name in ("gaussian", "uniform"):
        n = n_max or 2000
        return NoiseDataset(name, id_name, channels, size, n=n, seed=seed)

    src_channels, _ = NATIVE[name]
    tfm = transforms.Compose(
        _to_channels_size(channels, size, grayscale_source=(src_channels == 1))
        + [_norm_transform(id_name, channels)]
    )

    if name == "mnist":
        ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "fashion_mnist":
        ds = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "kmnist":
        ds = datasets.KMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "emnist_letters":
        ds = datasets.EMNIST(DATA_ROOT, split="letters", train=False, download=True, transform=tfm)
    elif name == "svhn":
        ds = datasets.SVHN(DATA_ROOT, split="test", download=True, transform=tfm)
    elif name == "cifar10":
        ds = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "cifar100":
        ds = datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown OOD dataset: {name}")

    if n_max is not None and n_max < len(ds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=n_max, replace=False)
        ds = Subset(ds, idx.tolist())
    return ds
