"""The black-box query interface.

GAQA never touches the weights of the victim: the only thing it is allowed to
do is add a perturbation to the clean image and read back the class
probabilities, exactly as an MLaaS API would return them (Fig. 1 of the paper).
The input normalisation of the service is applied here, inside the interface,
because the adversary neither sees nor controls it.

Every forward pass is counted as one query, which is the cost that the paper
minimises.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import models

# CIFAR-10 statistics; overridden by the checkpoint when it carries them.
DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_STD = (0.2470, 0.2435, 0.2616)


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Instantiate a torchvision classifier without pretrained weights."""
    try:
        model = models.get_model(arch, weights=None, num_classes=num_classes)
    except ValueError as exc:  # pragma: no cover - only for a bad --arch
        raise SystemExit(f"unknown architecture {arch!r}: {exc}") from exc
    return model


class Victim:
    """Wraps a classifier behind a probability-only interface."""

    def __init__(
        self,
        checkpoint: str | None,
        arch: str = "resnet18",
        num_classes: int = 10,
        device: str = "cuda",
        batch_size: int = 256,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = build_model(arch, num_classes)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state.get("model", state))
            mean = mean or state.get("mean")
            std = std or state.get("std")
        self.model.to(self.device).eval()

        # Pre-processing done by the deployed service, not by the attacker.
        self.mean = torch.tensor(mean or DEFAULT_MEAN, dtype=torch.float32,
                                 device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std or DEFAULT_STD, dtype=torch.float32,
                                device=self.device).view(1, -1, 1, 1)

        self.batch_size = batch_size
        self.queries = 0

    def reset_queries(self) -> None:
        self.queries = 0

    @torch.no_grad()
    def _logits(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model((pixels - self.mean) / self.std)

    @torch.no_grad()
    def probabilities(self, image: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Return the softmax output for ``clip(image + delta, 0, 1)``.

        ``image`` has shape ``(C, H, W)`` and ``deltas`` ``(Q, C, H, W)``; the
        result has shape ``(Q, num_classes)``.  One row costs one query.
        """
        outputs = []
        for start in range(0, len(deltas), self.batch_size):
            batch = deltas[start:start + self.batch_size]
            x = torch.from_numpy((image[None] + batch).astype(np.float32))
            x = x.clamp_(0.0, 1.0).to(self.device)
            outputs.append(torch.softmax(self._logits(x), dim=1).cpu().numpy())
        self.queries += len(deltas)
        return np.concatenate(outputs, axis=0)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> int:
        """Predicted label of a clean image (costs one query)."""
        x = torch.from_numpy(image[None].astype(np.float32)).to(self.device)
        self.queries += 1
        return int(self._logits(x).argmax(dim=1).item())
