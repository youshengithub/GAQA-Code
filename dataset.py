"""CIFAR loading.

``torchvision.datasets`` is used whenever the official archive is available.  A
lot of machines already have the dataset unpacked, though, and then torchvision
refuses it because the md5 of the batch files no longer matches the archive (a
re-packed copy is enough to trigger that).  The fallback below reads the classic
pickle batches directly, so an offline copy is usable without re-downloading
170 MB.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CIFAR = {"cifar10": 10, "cifar100": 100}
_BATCHES = {
    "train": [f"data_batch_{i}" for i in range(1, 6)],
    "test": ["test_batch"],
}


def _read_batches(root: str, name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``cifar-10-batches-py`` / ``cifar-100-python`` pickles directly."""
    folder = Path(root) / ("cifar-10-batches-py" if name == "cifar10" else "cifar-100-python")
    files = _BATCHES[split] if name == "cifar10" else [split]
    images, labels = [], []
    for filename in files:
        with (folder / filename).open("rb") as handle:
            entry = pickle.load(handle, encoding="bytes")
        images.append(np.asarray(entry[b"data"]).reshape(-1, 3, 32, 32))
        key = b"labels" if name == "cifar10" else b"fine_labels"
        labels.append(np.asarray(entry[key]))
    return np.concatenate(images), np.concatenate(labels)


def load_cifar(name: str, root: str, split: str, download: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` with images as ``uint8`` of shape ``(N, 3, 32, 32)``.

    An unpacked copy is used directly; torchvision is only involved when the
    batches are missing and a download is wanted.
    """
    if name not in CIFAR:
        raise ValueError(f"unknown dataset {name!r}")
    folder = Path(root) / ("cifar-10-batches-py" if name == "cifar10" else "cifar-100-python")
    files = _BATCHES[split] if name == "cifar10" else [split]
    if all((folder / filename).is_file() for filename in files):
        images, labels = _read_batches(root, name, split)
        print(f"dataset     : {len(images)} {name}/{split} images read from {folder}")
        return images.astype(np.uint8), labels.astype(np.int64)

    if not download:
        raise FileNotFoundError(f"{folder} does not contain the {split} batches")
    from torchvision import datasets

    factory = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    dataset = factory(root=root, train=(split == "train"), download=True)
    # torchvision keeps the images as uint8 NHWC
    return np.asarray(dataset.data).transpose(0, 3, 1, 2), np.asarray(dataset.targets)
