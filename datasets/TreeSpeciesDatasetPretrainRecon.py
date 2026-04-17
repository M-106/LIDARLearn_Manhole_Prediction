"""HELIALS pretraining dataset for ReCon cross-modal learning."""

import os
import torch
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
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


def project_to_image_xz(points, size=224):
    """
    Project 3D point cloud to 2D image using XZ plane (front view).
    Good for trees as it shows the vertical structure.

    Args:
        points: numpy array (N, 3) - normalized point cloud
        size: output image size (default 224x224 for ViT)

    Returns:
        img: numpy array (size, size, 3) - RGB image
    """
    # Use X and Z coordinates (front view)
    xz = points[:, [0, 2]]  # X horizontal, Z vertical

    # Normalize to [0, size-1]
    xz_min = xz.min(axis=0)
    xz_max = xz.max(axis=0)
    xz_range = xz_max - xz_min + 1e-8
    xz_norm = (xz - xz_min) / xz_range * (size - 1)

    # Create image (white points on black background)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw points with some thickness for visibility
    for x, z in xz_norm.astype(int):
        # Flip Z so tree grows upward (higher Z = lower row)
        row = size - 1 - z
        col = x
        # Clamp to valid range
        row = max(0, min(size - 1, row))
        col = max(0, min(size - 1, col))
        # Draw with small radius for visibility
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size:
                    img[r, c] = 255

    return img


@DATASETS.register_module()
class TreeSpeciesPretrainRecon(Dataset):
    """
    HELIAS pretraining dataset for ReCon with cross-modal learning.

    Returns: (taxonomy_id, model_id, points, img, text, label)
    - points: normalized point cloud tensor
    - img: 2D projection (XZ plane, 224x224)
    - text: class name (species folder name)
    - label: class index

    Config keys:
    - DATA_PATH: root folder with species subfolders
    - N_POINTS / npoints: number of points per sample
    - subset: 'train' (SSL), 'extra_train' (SVM train), 'val' (SVM test)
    - seed: random seed for reproducibility (default: 42)
    - pretrain_ratio: ratio for SSL pretrain split (default: 0.5)
    - svm_train_ratio: ratio of eval pool for SVM train (default: 0.75)
    - img_size: projection image size (default: 224)
    """

    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.get('npoints', config.get('N_POINTS', 1024))
        self.subset = config.get('subset', 'train')
        self.seed = config.get('seed', 42)
        self.pretrain_ratio = config.get('pretrain_ratio', 0.5)
        self.svm_train_ratio = config.get('svm_train_ratio', 0.75)
        self.img_size = config.get('img_size', 224)
        self.whole = config.get('whole', False)

        # Cache filename (distinct from other pretraining datasets)
        cache_name = config.get('cache_name', f'point_cloud_data_pretrain_recon_{self.npoints}.h5')
        self.cache_path = os.path.join(self.data_root, cache_name)

        # Image transforms (normalize for ViT)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load or create cache
        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._create_cache()

        # Select subset indices
        self._select_subset()

        # Permutation for random sampling
        self.permutation = np.arange(self.npoints)

        logger.info(f'[TreeSpeciesPretrainRecon] Loaded {len(self.indices)} samples for subset "{self.subset}"')

    def _create_cache(self):
        """Process raw files and create H5 cache with stratified splits."""
        logger.info(f'[TreeSpeciesPretrainRecon] Creating cache at {self.cache_path}')

        # Discover classes from folder structure
        classes = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        if not classes:
            raise ValueError(f"No class directories found in {self.data_root}")

        logger.info(f'[TreeSpeciesPretrainRecon] Found {len(classes)} classes: {", ".join(classes)}')

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

        logger.info(f'[TreeSpeciesPretrainRecon] Processed {len(point_clouds)} point clouds')

        # Create stratified splits
        np.random.seed(self.seed)
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

        logger.info(f'[TreeSpeciesPretrainRecon] Splits: train={len(idx_pretrain)}, '
                    f'extra_train={len(idx_extra_train)}, val={len(idx_val)}')

        # Save to H5
        with h5py.File(self.cache_path, 'w') as f:
            f.attrs['purpose'] = 'pretrain_recon'
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

        logger.info(f'[TreeSpeciesPretrainRecon] Cache saved to {self.cache_path}')

        # Load into memory
        self._load_cache()

    def _load_cache(self):
        """Load cached data from H5."""
        logger.info(f'[TreeSpeciesPretrainRecon] Loading cache from {self.cache_path}')

        with h5py.File(self.cache_path, 'r') as f:
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
            # If whole=True, use all data for training
            if self.whole:
                self.indices = np.concatenate([self.idx_pretrain, self.idx_extra_train, self.idx_val])
        elif self.subset == 'extra_train':
            self.indices = self.idx_extra_train
        elif self.subset == 'val':
            self.indices = self.idx_val
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
        class_name = self.classes[label]

        # Random sample and normalize
        points = self.random_sample(points, self.npoints)
        points = self.pc_norm(points)

        # Generate 2D projection (XZ plane - front view)
        img = project_to_image_xz(points, size=self.img_size)
        img = self.img_transform(img)

        # Convert points to tensor
        points = torch.from_numpy(points).float()

        # Text is the class name (species)
        text = class_name

        # Label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Return format: (taxonomy_id, model_id, points, img, text, label)
        return 'HELIAS', file_name, points, img, text, label_tensor

    def __len__(self):
        return len(self.indices)
