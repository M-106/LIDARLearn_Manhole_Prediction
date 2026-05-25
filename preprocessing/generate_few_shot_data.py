"""
Generate ModelNet40 few-shot episodes (.pkl files) for the standard
5-way / 10-way × 10-shot / 20-shot protocol.

For each (way, shot) combo this produces 10 independently-sampled
episodes as ``data/ModelNetFewshot/<way>way_<shot>shot/{0..9}.pkl``.

Reproducibility: every shuffle is seeded from ``(base_seed, prefix_ind)``
so re-running the script on any machine yields byte-identical splits.

Usage (from the LIDARLearn project root):
    python preprocessing/generate_few_shot_data.py
    python preprocessing/generate_few_shot_data.py --seed 0
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / 'data' / 'ModelNet' / 'modelnet40_normal_resampled'
DEFAULT_TARGET = REPO_ROOT / 'data' / 'ModelNetFewshot'

WAYS = [5, 10]
SHOTS = [10, 20]
NUM_EPISODES = 10
NUM_CLASSES = 40


def _episode_seed(base_seed: int, way: int, shot: int, prefix_ind: int) -> int:
    """Derive a deterministic seed for one episode."""
    return (base_seed * 1_000_003
            + way * 10_007
            + shot * 101
            + prefix_ind)


def generate_fewshot_data(
    train_data: tuple[list, list],
    test_data: tuple[list, list],
    way: int,
    shot: int,
    prefix_ind: int,
    target: Path,
    base_seed: int,
    eval_sample: int = 20,
) -> None:
    train_list_of_points, train_list_of_labels = train_data
    test_list_of_points, test_list_of_labels = test_data

    rng = random.Random(_episode_seed(base_seed, way, shot, prefix_ind))

    train_cls_dataset: dict[int, list] = {}
    for point, label in zip(train_list_of_points, train_list_of_labels):
        train_cls_dataset.setdefault(label[0], []).append(point)

    test_cls_dataset: dict[int, list] = {}
    for point, label in zip(test_list_of_points, test_list_of_labels):
        test_cls_dataset.setdefault(label[0], []).append(point)

    assert sum(len(v) for v in train_cls_dataset.values()) > 0, "empty train set"
    assert sum(len(v) for v in test_cls_dataset.values()) > 0, "empty test set"

    keys = list(train_cls_dataset.keys())
    rng.shuffle(keys)

    train_dataset, test_dataset = [], []
    for i, key in enumerate(keys[:way]):
        train_data_list = list(train_cls_dataset[key])
        rng.shuffle(train_data_list)
        assert len(train_data_list) > shot, f"class {key}: only {len(train_data_list)} train samples"
        for data in train_data_list[:shot]:
            train_dataset.append((data, i, key))

        test_data_list = list(test_cls_dataset[key])
        rng.shuffle(test_data_list)
        assert len(test_data_list) >= eval_sample, f"class {key}: only {len(test_data_list)} test samples"
        for data in test_data_list[:eval_sample]:
            test_dataset.append((data, i, key))

    rng.shuffle(train_dataset)
    rng.shuffle(test_dataset)

    out_dir = target / f"{way}way_{shot}shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{prefix_ind}.pkl", 'wb') as f:
        pickle.dump({'train': train_dataset, 'test': test_dataset}, f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate reproducible ModelNet40 few-shot episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                    help="ModelNet40 resampled directory.")
    ap.add_argument('--target', type=Path, default=DEFAULT_TARGET,
                    help="Where to write <way>way_<shot>shot/{0..9}.pkl.")
    ap.add_argument('--seed', type=int, default=0,
                    help="Base seed — every episode derives its own seed from this.")
    args = ap.parse_args()

    train_path = args.root / 'modelnet40_train_8192pts_fps.dat'
    test_path = args.root / 'modelnet40_test_8192pts_fps.dat'
    for p in (train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    n_train = sum(1 for _ in train_data[0])
    n_test = sum(1 for _ in test_data[0])
    print(f"Loaded ModelNet40: {n_train} train, {n_test} test clouds "
          f"(from {args.root})")
    print(f"Base seed: {args.seed}  →  writing to {args.target}")

    for way in WAYS:
        for shot in SHOTS:
            for prefix_ind in range(NUM_EPISODES):
                generate_fewshot_data(
                    train_data=train_data,
                    test_data=test_data,
                    way=way,
                    shot=shot,
                    prefix_ind=prefix_ind,
                    target=args.target,
                    base_seed=args.seed,
                )
            print(f"  {way}way_{shot}shot: wrote {NUM_EPISODES} episodes")


if __name__ == '__main__':
    main()
