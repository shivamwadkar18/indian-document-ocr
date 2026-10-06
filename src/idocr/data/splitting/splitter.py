"""Deterministic, leakage-free splitting of source groups into train/valid/test sets."""

from __future__ import annotations

import random
from typing import Sequence

from idocr.data.splitting.grouping import SourceGroup


def split_source_groups(
    groups: Sequence[SourceGroup],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[SourceGroup]]:
    """Partition source groups into train, valid, and test splits.

    Crucially, all images in any single SourceGroup are assigned to the
    same split, preventing any cross-split leakage.

    Args:
        groups: Collection of SourceGroup objects.
        ratios: Target proportions for (train, valid, test). Must sum to 1.0.
        seed: Random seed for deterministic shuffling.

    Returns:
        Dictionary mapping split name ("train", "valid", "test") to list of SourceGroups.
    """
    if abs(sum(ratios) - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    # Sort groups first to guarantee platform-independent determinism
    sorted_groups = sorted(groups, key=lambda g: g.group_id)
    rng = random.Random(seed)
    rng.shuffle(sorted_groups)

    total_images = sum(g.image_count for g in sorted_groups)
    target_train_imgs = int(total_images * ratios[0])
    target_valid_imgs = int(total_images * ratios[1])
    target_test_imgs = total_images - target_train_imgs - target_valid_imgs

    train_groups: list[SourceGroup] = []
    valid_groups: list[SourceGroup] = []
    test_groups: list[SourceGroup] = []

    train_imgs = 0
    valid_imgs = 0
    test_imgs = 0

    for g in sorted_groups:
        n = g.image_count
        # Allocate to train, valid, or test based on remaining target capacities
        if (train_imgs + n <= target_train_imgs) or (valid_imgs >= target_valid_imgs and test_imgs >= target_test_imgs):
            train_groups.append(g)
            train_imgs += n
        elif (valid_imgs + n <= target_valid_imgs) or (test_imgs >= target_test_imgs):
            valid_groups.append(g)
            valid_imgs += n
        else:
            test_groups.append(g)
            test_imgs += n

    return {
        "train": train_groups,
        "valid": valid_groups,
        "test": test_groups,
    }
