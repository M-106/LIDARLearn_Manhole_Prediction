"""S3DIS semantic segmentation dataset loader with block sampling."""

import glob
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .build import DATASETS


S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter',
]


@DATASETS.register_module()
class S3DIS(Dataset):
    """S3DIS semantic segmentation dataset (block-sampling loader)."""

    num_seg_classes = 13
    num_obj_classes = 1  # single "scene" category
    seg_classes = None   # no per-shape category → semantic mIoU path
    category_names = list(S3DIS_CLASSES)

    _FEATURE_CHANNELS = {'xyz': 3, 'xyz_rgb': 6, 'xyz_rgb_norm': 9}

    def __init__(self, config):
        self.root = config.DATA_PATH                             # data/S3DIS/s3dis_npy/
        self.num_point = int(getattr(config, 'N_POINTS', 4096))
        self.block_size = float(getattr(config, 'block_size', 1.0))
        self.sample_rate = float(getattr(config, 'sample_rate', 1.0))
        self.partition = config.subset                           # 'train'|'val'|'test'
        self.test_area = int(getattr(config, 'test_area', 5))
        self.feature_mode = str(getattr(config, 'feature_mode', 'xyz_rgb_norm'))
        # Speed-test subsampling: fraction of rooms AND blocks to keep.
        # 1.0 = full data (default), 0.1 = 10% for quick timing runs.
        self.speed_test_fraction = float(getattr(config, 'speed_test_fraction', 1.0))
        if not 0.0 < self.speed_test_fraction <= 1.0:
            raise ValueError(
                f"speed_test_fraction must be in (0, 1], got {self.speed_test_fraction}"
            )
        if self.feature_mode not in self._FEATURE_CHANNELS:
            raise ValueError(
                f"feature_mode must be one of {list(self._FEATURE_CHANNELS)}, "
                f"got '{self.feature_mode}'"
            )
        self.num_features = self._FEATURE_CHANNELS[self.feature_mode]

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"S3DIS .npy root not found: {self.root}. "
                f"Run `python preprocessing/preprocess_s3dis.py` first."
            )

        # --- Select rooms for this split -------------------------------------
        all_room_paths = sorted(glob.glob(os.path.join(self.root, 'Area_*_*.npy')))
        if not all_room_paths:
            raise FileNotFoundError(
                f"No Area_*_*.npy files in {self.root}. "
                f"Run `python preprocessing/preprocess_s3dis.py` first."
            )

        tag_test = f"Area_{self.test_area}_"
        if self.partition == 'train':
            room_paths = [p for p in all_room_paths if tag_test not in os.path.basename(p)]
        elif self.partition in ('val', 'test'):
            room_paths = [p for p in all_room_paths if tag_test in os.path.basename(p)]
        else:
            raise ValueError(f"Unknown subset: {self.partition}")

        # Speed-test mode: keep only a fraction of rooms (deterministic — first N)
        if self.speed_test_fraction < 1.0:
            keep_n = max(1, int(round(len(room_paths) * self.speed_test_fraction)))
            room_paths = room_paths[:keep_n]
            if self.partition in ('val', 'test'):
                import warnings
                warnings.warn(
                    f"speed_test_fraction={self.speed_test_fraction} is active for the "
                    f"'{self.partition}' split — evaluation covers only {keep_n}/{len(room_paths) + keep_n} "
                    f"rooms. Do NOT report these numbers as final accuracy.",
                    UserWarning, stacklevel=2,
                )

        self.room_paths = room_paths
        self.room_names = [Path(p).stem for p in room_paths]

        # --- Load rooms into memory (~a few hundred MB per area) -------------
        self.room_coord_min: list = []
        self.room_coord_max: list = []
        self.room_points: list = []   # list of [N, 6] (xyz rgb)
        self.room_labels: list = []   # list of [N]
        room_lens = []
        for p in room_paths:
            data = np.load(p)                                     # [N, 7]
            pts = data[:, :6].astype(np.float32)                  # xyz rgb
            seg = data[:, 6].astype(np.int64)
            # Normalize colors to [0,1] if they are in [0,255]
            if pts[:, 3:].max() > 1.5:
                pts[:, 3:] = pts[:, 3:] / 255.0
            self.room_points.append(pts)
            self.room_labels.append(seg)
            self.room_coord_min.append(pts[:, :3].min(axis=0))
            self.room_coord_max.append(pts[:, :3].max(axis=0))
            room_lens.append(pts.shape[0])

        # --- Sampling probability per room (prop. to point count) [train only] ---
        room_lens_arr = np.array(room_lens, dtype=np.float64)
        total_points = room_lens_arr.sum()

        if self.partition == 'train':
            # Training: stochastic block sampling weighted by room size.
            # speed_test_fraction also shrinks the per-epoch block budget.
            self.num_samples = int(total_points * self.sample_rate / self.num_point
                                   * self.speed_test_fraction)
            self.num_samples = max(self.num_samples, len(room_paths))
            self.room_probs = room_lens_arr / total_points
            self._val_blocks = None  # not used for train
        else:
            # Val/test: pre-enumerate ALL non-overlapping 1×1 blocks
            # deterministically so __getitem__ is index-stable and comparable
            # across epochs and runs.
            self.room_probs = None
            self._val_blocks = self._enumerate_val_blocks()
            self.num_samples = len(self._val_blocks)

    def _enumerate_val_blocks(self):
        """Pre-compute a fixed, non-overlapping tiling of blocks for all rooms.

        For each room, tiles the XY plane with a regular grid of block_size×block_size
        cells and keeps only cells that contain at least num_point//4 points.
        Within each cell, points are sorted by (x, y, z) for a fixed, seeded draw.

        Returns a list of (room_idx, sorted_point_indices) tuples — one per block.
        """
        blocks = []
        half = self.block_size / 2.0
        stride = self.block_size  # non-overlapping grid

        for room_idx, pts in enumerate(self.room_points):
            x_min_r = self.room_coord_min[room_idx][0]
            y_min_r = self.room_coord_min[room_idx][1]
            x_max_r = self.room_coord_max[room_idx][0]
            y_max_r = self.room_coord_max[room_idx][1]

            # Grid of block centres covering the room footprint
            cx = x_min_r + half
            while cx <= x_max_r + half:
                cy = y_min_r + half
                while cy <= y_max_r + half:
                    # Half-open intervals [lower, upper) prevent boundary points
                    # from being double-counted in adjacent cells.
                    mask = (
                        (pts[:, 0] >= cx - half) & (pts[:, 0] < cx + half)
                        & (pts[:, 1] >= cy - half) & (pts[:, 1] < cy + half)
                    )
                    idx = np.flatnonzero(mask)
                    if idx.size >= self.num_point // 4:
                        # Sort for determinism (same block → same point order)
                        sort_key = np.lexsort((pts[idx, 2], pts[idx, 1], pts[idx, 0]))
                        blocks.append((room_idx, idx[sort_key], cx, cy))
                    cy += stride
                cx += stride

        return blocks

    def _build_block_features(self, room_idx, choice, cx, cy):
        """Build the feature tensor for one block given chosen point indices."""
        pts = self.room_points[room_idx]
        seg = self.room_labels[room_idx]
        coord_min = self.room_coord_min[room_idx]
        coord_max = self.room_coord_max[room_idx]

        block_pts = pts[choice]   # [num_point, 6]
        block_seg = seg[choice]   # [num_point]

        xyz = block_pts[:, :3]
        xyz_block = xyz.copy()
        xyz_block[:, 0] -= cx
        xyz_block[:, 1] -= cy

        parts = [xyz_block]
        if self.feature_mode in ('xyz_rgb', 'xyz_rgb_norm'):
            parts.append(block_pts[:, 3:6])
        if self.feature_mode == 'xyz_rgb_norm':
            xyz_room_norm = (xyz - coord_min) / np.maximum(
                coord_max - coord_min, 1e-6
            )
            parts.append(xyz_room_norm)
        feats = np.concatenate(parts, axis=1).astype(np.float32)
        return feats, block_seg.astype(np.int32)

    def _sample_block(self, room_idx: int):
        """Randomly pick a 1x1 (x,y) block center and sample num_point points.

        Used only during training.
        """
        pts = self.room_points[room_idx]
        seg = self.room_labels[room_idx]

        N = pts.shape[0]
        half = self.block_size / 2.0
        idx = np.array([], dtype=np.int64)
        cx, cy = pts[0, 0], pts[0, 1]  # fallback centre
        for _ in range(10):
            center_idx = np.random.randint(0, N)
            cx, cy = pts[center_idx, 0], pts[center_idx, 1]
            mask = (
                (pts[:, 0] >= cx - half) & (pts[:, 0] <= cx + half)
                & (pts[:, 1] >= cy - half) & (pts[:, 1] <= cy + half)
            )
            idx = np.flatnonzero(mask)
            if idx.size >= self.num_point // 4:
                break
        if idx.size == 0:
            idx = np.arange(N)

        replace = idx.size < self.num_point
        choice = np.random.choice(idx, self.num_point, replace=replace)
        return self._build_block_features(room_idx, choice, cx, cy)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.partition == 'train':
            # Stochastic: sample a room proportional to its size, ignore index.
            room_idx = int(np.random.choice(len(self.room_paths), p=self.room_probs))
            points, seg = self._sample_block(room_idx)
            room_name = self.room_names[room_idx]
        else:
            # Deterministic: use pre-enumerated block at this index.
            room_idx, sorted_idx, cx, cy = self._val_blocks[index]
            N = sorted_idx.size
            if N >= self.num_point:
                # Take every k-th point so the sample covers the block uniformly.
                step = N // self.num_point
                choice = sorted_idx[::step][:self.num_point]
            else:
                # Tile the available points to reach num_point (no randomness).
                reps = (self.num_point + N - 1) // N
                choice = np.tile(sorted_idx, reps)[:self.num_point]
            points, seg = self._build_block_features(room_idx, choice, cx, cy)
            room_name = self.room_names[room_idx]

        return 'S3DIS', room_name, (points, seg)
