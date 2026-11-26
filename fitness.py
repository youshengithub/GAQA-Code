"""Fitness and the two auxiliary measures defined in the paper.

Eq. 2  attack effectiveness (confidence margin towards the best non-true class)
Eq. 3  attack evasiveness (normalised L2 distance + MCSD)
Eq. 4  mean convolution standard deviation (MCSD)
Eq. 6  fitness of a perturbation solution

Everything is vectorised over a whole population of perturbations so that one
generation costs a single batch query plus a handful of numpy operations.
"""

from __future__ import annotations

import numpy as np


def margin(probabilities: np.ndarray, label: int) -> np.ndarray:
    """``max_{j != y} (F(x + d)_j - F(x + d)_y)``, the term of Eq. 2 / Eq. 6.

    ``probabilities`` has shape ``(Q, num_classes)`` and holds the softmax
    output of the victim for ``Q`` perturbations of the same clean image, so
    every row shares the true label ``label``.
    """
    true_class = probabilities[:, label]
    other = probabilities.copy()
    other[:, label] = -np.inf
    return other.max(axis=1) - true_class


def effectiveness(probabilities: np.ndarray, label: int) -> np.ndarray:
    """Eq. 2, i.e. the margin floored at zero."""
    return np.maximum(margin(probabilities, label), 0.0)


def mcsd(deltas: np.ndarray) -> np.ndarray:
    """Eq. 4 for a batch of perturbations.

    ``deltas`` has shape ``(Q, C, H, W)``; the result has shape ``(Q,)`` and is
    the mean over channels and interior pixels of the standard deviation of a
    pixel together with its eight neighbours.
    """
    q, c, h, w = deltas.shape
    if h < 3 or w < 3:
        raise ValueError("MCSD needs at least a 3x3 image")
    neighbourhood = np.stack(
        [deltas[:, :, dy:dy + h - 2, dx:dx + w - 2] for dy in (0, 1, 2) for dx in (0, 1, 2)],
        axis=0,
    )
    return neighbourhood.std(axis=0).mean(axis=(1, 2, 3))


def mcsd_reference(shape: tuple[int, int, int], eps: float) -> float:
    """``MCSD_max`` of Eq. 6: the checkerboard perturbation at the bound.

    The checkerboard is the highest-frequency pattern that fits inside
    ``[-eps, eps]``, which makes it a natural normaliser for the smoothness
    term instead of a population-dependent maximum.
    """
    c, h, w = shape
    yy, xx = np.indices((h, w))
    checkerboard = np.where((yy + xx) % 2 == 0, 1.0, -1.0) * eps
    tile = np.broadcast_to(checkerboard, (1, c, h, w)).copy()
    return float(mcsd(tile)[0])


def normalised_distance(deltas: np.ndarray, eps: float) -> np.ndarray:
    """``||x_adv - x||_2 / sqrt(dim(x) * eps^2)``, the first term of Eq. 3."""
    flat = deltas.reshape(len(deltas), -1).astype(np.float64)
    return np.linalg.norm(flat, axis=1) / np.sqrt(flat.shape[1] * eps ** 2)


def fitness(
    probabilities: np.ndarray,
    label: int,
    deltas: np.ndarray,
    eps: float,
    psi: float = 1.0,
    mcsd_max: float | None = None,
) -> np.ndarray:
    """Eq. 6, evaluated for a whole population.

    ``psi`` is the minimal attack effectiveness of the paper.  A successful
    solution is capped at ``1 + psi`` and the evasiveness term then decides
    between successful solutions; a failed one keeps its (negative) margin.
    The maximal attainable value is therefore ``1 + 2 * psi``.
    """
    if mcsd_max is None:
        mcsd_max = mcsd_reference(deltas.shape[1:], eps)
    effective = np.minimum(1.0 + margin(probabilities, label), 1.0 + psi)
    smoothness = (1.0 - normalised_distance(deltas, eps)) + (1.0 - mcsd(deltas) / mcsd_max)
    return effective + 0.5 * psi * smoothness
