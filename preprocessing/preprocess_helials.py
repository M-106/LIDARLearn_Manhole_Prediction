#!/usr/bin/env python3
"""
HELIALS — full preprocessing pipeline.

Reads per-tree LAS files from ``preprocessing/full_data_HeliALS/`` (raw
Zenodo release) and writes per-tree ``.xyz`` text files into
``data/HELIALS/`` in the canonical repo format:

    x y z R G B
    <2048 rows, space-separated, 6 decimals>

Pipeline, per LAS:
    1. Load XYZ + RGB via laspy (LAS 1.2, point format 3).
    2. Downsample to 2048 points using GPU farthest-point sampling
       (``pointnet2_ops.pointnet2_utils.furthest_point_sample``).
    3. Center XYZ on centroid (original scale preserved — no unit-sphere
       normalisation, matching the existing preprocessed clouds).
    4. Normalise RGB from LAS uint16 to ``[0, 1]`` by dividing by 65535.
    5. Write ``.xyz`` with a ``"x y z R G B"`` header + 2048 data rows.

Usage:
    python preprocessing/preprocess_helials.py \\
        --input_dir preprocessing/full_data_HeliALS \\
        --output_dir data/HELIALS

Requires:
    - CUDA + pointnet2_ops installed (``bash extensions/install_extensions.sh``)
    - ``laspy``, ``numpy``, ``torch``, ``tqdm``

The species-metadata CSV (``training-and-test-segments-with-species.csv``)
is not generated here — copy it from the raw Zenodo download into
``--output_dir`` before training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import laspy
except ImportError:
    sys.exit("laspy is required: pip install laspy")

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample
except ImportError:
    sys.exit(
        "pointnet2_ops is required. Build the CUDA extensions first:\n"
        "    bash extensions/install_extensions.sh"
    )


TARGET_POINTS = 2048
RGB_SCALE = 65535.0  # LAS uint16 → [0, 1]


def fps_downsample_gpu(xyz: np.ndarray, target: int) -> np.ndarray:
    """Return indices of `target` points selected from `xyz` by GPU FPS.

    `xyz` is [N, 3] float64. Returns int64 indices of shape (target,).
    """
    pts = torch.from_numpy(xyz).float().cuda().unsqueeze(0).contiguous()  # [1, N, 3]
    idx = furthest_point_sample(pts, target)  # [1, target] int32
    return idx[0].long().cpu().numpy()


def load_las_xyz_rgb(las_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xyz [N, 3] float64, rgb [N, 3] float32 in [0, 1]) or None on error."""
    las = laspy.read(str(las_path))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    if not all(hasattr(las, c) for c in ("red", "green", "blue")):
        print(f"  [SKIP] {las_path.name}: no RGB channels in LAS")
        return None

    rgb = np.stack(
        [
            np.asarray(las.red, dtype=np.float32),
            np.asarray(las.green, dtype=np.float32),
            np.asarray(las.blue, dtype=np.float32),
        ],
        axis=1,
    ) / RGB_SCALE
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz, rgb


def write_xyz(out_path: Path, points_xyz_rgb: np.ndarray) -> None:
    """Write a (2048, 6) array in the canonical ``x y z R G B`` format."""
    lines = ["x y z R G B\n"]
    for row in points_xyz_rgb:
        lines.append(
            f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
            f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f}\n"
        )
    out_path.write_text("".join(lines))


def process_one(las_path: Path, out_dir: Path) -> tuple[bool, str | None]:
    loaded = load_las_xyz_rgb(las_path)
    if loaded is None:
        return False, "no RGB"
    xyz, rgb = loaded

    n = xyz.shape[0]
    if n < TARGET_POINTS:
        return False, f"only {n} points (< {TARGET_POINTS})"

    idx = fps_downsample_gpu(xyz, TARGET_POINTS)
    xyz_s = xyz[idx]
    rgb_s = rgb[idx]

    xyz_s = xyz_s - xyz_s.mean(axis=0)

    out_path = out_dir / f"{las_path.stem}.xyz"
    combined = np.concatenate([xyz_s, rgb_s.astype(np.float64)], axis=1)
    write_xyz(out_path, combined)
    return True, None


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess HELIALS LAS files to 2048-point .xyz trees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("preprocessing/full_data_HeliALS"),
        help="Directory containing raw .las files.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/HELIALS"),
        help="Where to write per-tree .xyz files.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the torch/numpy fallback paths (FPS itself is deterministic).",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for GPU FPS — no GPU detected.")

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    las_files = sorted(args.input_dir.glob("*.las"))
    if not las_files:
        sys.exit(f"No .las files found in {args.input_dir}")

    print(
        f"HELIALS preprocessing:\n"
        f"  input : {args.input_dir}  ({len(las_files)} files)\n"
        f"  output: {args.output_dir}\n"
        f"  target: {TARGET_POINTS} points per tree (GPU FPS)\n"
    )

    ok, skipped = 0, []
    for las_path in tqdm(las_files, desc="Preprocessing", unit="file"):
        success, reason = process_one(las_path, args.output_dir)
        if success:
            ok += 1
        else:
            skipped.append((las_path.name, reason))

    print(f"\nDone. {ok}/{len(las_files)} trees written to {args.output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped[:20]:
            print(f"  {name}  — {reason}")
        if len(skipped) > 20:
            print(f"  ... ({len(skipped) - 20} more)")

    csv_name = "training-and-test-segments-with-species.csv"
    if not (args.output_dir / csv_name).exists():
        print(
            f"\n[!] Reminder: copy '{csv_name}' from the raw Zenodo download "
            f"into {args.output_dir}/ before training — the loader requires it."
        )


if __name__ == "__main__":
    main()
