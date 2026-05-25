"""HELIALS SSL pretraining dataset with stratified splits and H5 caching."""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

from .build import DATASETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_point_cloud(file_path):
    """Load point cloud data from various formats (xyz, pts, txt)."""
    file_extension = Path(file_path).suffix.lower()
    try:
        if file_extension == '.xyz':
            points = np.loadtxt(file_path)
            return points[:, :3]
        elif file_extension == '.pts':
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if not all(c.isdigit() or c == '.' or c == '-' or c.isspace() for c in first_line):
                    points = np.loadtxt(file_path, skiprows=1)
                else:
                    f.seek(0)
                    points = np.loadtxt(file_path)
            return points[:, :3]
        elif file_extension == '.txt':
            points = np.loadtxt(file_path)
            return points[:, :3]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


@DATASETS.register_module()
class TreeSpeciesPretrain(Dataset):
    """
    HELIAS pretraining dataset with stratified splits and H5 caching.

    Config keys:
    - DATA_PATH: root folder with species subfolders
    - N_POINTS / npoints: number of points per sample
    - subset: 'train' (SSL), 'extra_train' (SVM train), 'val' (SVM test)
    - seed: random seed for reproducibility (default: 42)
    - pretrain_ratio: ratio for SSL pretrain split (default: 0.5)
    - svm_train_ratio: ratio of eval pool for SVM train (default: 0.75)
    """

    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.get('npoints', config.get('N_POINTS', 1024))
        self.subset = config.get('subset', 'train')
        self.seed = config.get('seed', 42)
        self.pretrain_ratio = config.get('pretrain_ratio', 0.5)
        self.svm_train_ratio = config.get('svm_train_ratio', 0.75)

        # Cache filename (distinct from finetuning)
        cache_name = config.get('cache_name', f'point_cloud_data_pretrain_{self.npoints}.h5')
        self.cache_path = os.path.join(self.data_root, cache_name)

        # Load or create cache
        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._create_cache()

        # Select subset indices
        self._select_subset()

        # Permutation for random sampling
        self.permutation = np.arange(self.npoints)

        logger.info(f'[TreeSpeciesPretrain] Loaded {len(self.indices)} samples for subset "{self.subset}"')

    def _create_cache(self):
        """Process raw files and create H5 cache with stratified splits."""
        logger.info(f'[TreeSpeciesPretrain] Creating cache at {self.cache_path}')

        # Discover classes from folder structure
        classes = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        if not classes:
            raise ValueError(f"No class directories found in {self.data_root}")

        logger.info(f'[TreeSpeciesPretrain] Found {len(classes)} classes: {", ".join(classes)}')

        point_clouds = []
        labels = []
        file_names = []

        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(self.data_root, class_name)
            files = [
                f for f in os.listdir(class_path)
                if f.endswith(('.xyz', '.pts', '.txt'))
            ]

            for file_name in files:
                try:
                    file_path = os.path.join(class_path, file_name)
                    points = load_point_cloud(file_path)

                    # Sample or pad to npoints
                    if len(points) >= self.npoints:
                        idx = np.random.choice(len(points), self.npoints, replace=False)
                        points = points[idx]
                    else:
                        # Pad by repeating
                        repeats = (self.npoints // len(points)) + 1
                        points = np.tile(points, (repeats, 1))[:self.npoints]

                    point_clouds.append(points)
                    labels.append(class_idx)
                    file_names.append(f"{class_name}/{file_name}")
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")

        point_clouds = np.array(point_clouds, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        logger.info(f'[TreeSpeciesPretrain] Processed {len(point_clouds)} point clouds')

        # Create stratified splits. ``train_test_split`` consumes the
        # ``random_state`` argument directly, so we don't need — and must
        # not — mutate the global numpy RNG here (doing so would leak into
        # unrelated code running in the same process).
        all_indices = np.arange(len(point_clouds))

        # Split: pretrain (50%) vs eval pool (50%)
        idx_pretrain, idx_eval = train_test_split(
            all_indices,
            test_size=1 - self.pretrain_ratio,
            stratify=labels,
            random_state=self.seed
        )

        # Split eval pool: extra_train (75%) vs val (25%)
        labels_eval = labels[idx_eval]
        idx_extra_train, idx_val = train_test_split(
            idx_eval,
            test_size=1 - self.svm_train_ratio,
            stratify=labels_eval,
            random_state=self.seed
        )

        logger.info(f'[TreeSpeciesPretrain] Splits: train={len(idx_pretrain)}, '
                    f'extra_train={len(idx_extra_train)}, val={len(idx_val)}')

        # Save to H5
        with h5py.File(self.cache_path, 'w') as f:
            f.attrs['purpose'] = 'pretrain'
            f.attrs['npoints'] = self.npoints
            f.attrs['seed'] = self.seed
            f.attrs['pretrain_ratio'] = self.pretrain_ratio
            f.attrs['svm_train_ratio'] = self.svm_train_ratio

            f.create_dataset('point_clouds', data=point_clouds)
            f.create_dataset('labels', data=labels)
            f.create_dataset('classes', data=np.array(classes, dtype='S'))
            f.create_dataset('file_names', data=np.array(file_names, dtype='S'))

            f.create_dataset('idx_pretrain', data=idx_pretrain)
            f.create_dataset('idx_extra_train', data=idx_extra_train)
            f.create_dataset('idx_val', data=idx_val)

        logger.info(f'[TreeSpeciesPretrain] Cache saved to {self.cache_path}')

        # Load into memory
        self._load_cache()

    def _load_cache(self):
        """Load cached data from H5."""
        logger.info(f'[TreeSpeciesPretrain] Loading cache from {self.cache_path}')

        with h5py.File(self.cache_path, 'r') as f:
            # Verify it's a pretrain cache
            purpose = f.attrs.get('purpose', '')
            if purpose != 'pretrain':
                raise ValueError(f"H5 cache is not a pretrain cache (purpose={purpose})")

            self.point_clouds = f['point_clouds'][:]
            self.labels = f['labels'][:]
            self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
            self.file_names = [n.decode() if isinstance(n, bytes) else n for n in f['file_names'][:]]

            self.idx_pretrain = f['idx_pretrain'][:]
            self.idx_extra_train = f['idx_extra_train'][:]
            self.idx_val = f['idx_val'][:]

    def _select_subset(self):
        """Select indices based on subset."""
        if self.subset == 'train':
            self.indices = self.idx_pretrain
            self.return_labels = False
        elif self.subset == 'extra_train':
            self.indices = self.idx_extra_train
            self.return_labels = True
        elif self.subset == 'val':
            self.indices = self.idx_val
            self.return_labels = True
        else:
            raise ValueError(f"Unknown subset: {self.subset}. Use 'train', 'extra_train', or 'val'.")

    def pc_norm(self, pc):
        """Normalize point cloud: center and scale to unit sphere."""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if m > 0:
            pc = pc / m
        return pc

    def random_sample(self, pc, num):
        """Random sample points."""
        np.random.shuffle(self.permutation)
        return pc[self.permutation[:num]]

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        points = self.point_clouds[real_idx].copy()
        label = self.labels[real_idx]
        file_name = self.file_names[real_idx]

        # Random sample and normalize
        points = self.random_sample(points, self.npoints)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()

        # Return format depends on subset
        if self.return_labels:
            # For SVM validation: (taxonomy_id, model_id, (points, label))
            return 'HELIAS', file_name, (points, torch.tensor(label, dtype=torch.long))
        else:
            # For SSL training: (taxonomy_id, model_id, points)
            return 'HELIAS', file_name, points

    def __len__(self):
        return len(self.indices)
