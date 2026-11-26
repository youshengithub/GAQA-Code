"""Train a victim classifier so that ``attack.py`` has something to attack.

The paper evaluates on WideResNet-based defended models and on MlpMixer /
Swin / ResNet backbones.  Those checkpoints are not redistributed here, so this
script trains a plain CIFAR classifier of your choice in a couple of minutes;
any other checkpoint of the same architecture works as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from dataset import CIFAR, load_cifar
from victim import build_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def to_tensors(images: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the images as uint8; they are normalised in the training loop."""
    return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))


def make_loader(images, labels, train: bool, batch_size: int) -> DataLoader:
    x, y = to_tensors(images, labels)
    normalize = transforms.Normalize(MEAN, STD)
    if train:
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        # one fresh augmentation per image, re-drawn every epoch
        x = torch.stack([normalize(augment(image.float().div(255))) for image in x])
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                          pin_memory=True)
    x = normalize(x.float().div(255))
    return DataLoader(TensorDataset(x, y), batch_size=512, pin_memory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="train a CIFAR victim model")
    parser.add_argument("--data", default="cifar10", choices=sorted(CIFAR))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="victim_cifar10.pt")
    args = parser.parse_args()

    num_classes = CIFAR[args.data]
    train_images, train_labels = load_cifar(args.data, args.data_root, "train")
    test_images, test_labels = load_cifar(args.data, args.data_root, "test")
    test_loader = make_loader(test_images, test_labels, train=False,
                              batch_size=args.batch_size)
    print(f"victim      : {args.arch}, {len(train_images)} train / {len(test_images)} test images")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(train_images, train_labels, train=True,
                                   batch_size=args.batch_size)
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(inputs)
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                correct += (model(inputs).argmax(dim=1) == targets).sum().item()
        accuracy = correct / len(test_images)
        print(f"epoch {epoch:3d}/{args.epochs}  loss={running / len(train_images):.4f}  "
              f"test accuracy={accuracy:.4f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "num_classes": num_classes, "accuracy": accuracy,
                        "mean": MEAN, "std": STD},
                       Path(args.output))

    print(f"saved {args.output} (best test accuracy {best:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
