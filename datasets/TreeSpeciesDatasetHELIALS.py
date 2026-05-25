"""HeliALS tree species dataset loader."""

import os
import sys
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
import open3d as o3d
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from tqdm import tqdm
from .build import DATASETS
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Species code to name mapping (from HeliALS dataset)
SPECIES_NAMES = {
    1: 'Pine',    # Pinus sylvestris
    2: 'Spruce',  # Picea sp.
    3: 'Birch',   # Betula sp.
    4: 'Maple',   # Acer platanoides
    5: 'Aspen',   # Populus tremula
    6: 'Rowan',   # Sorbus sp.
    7: 'Oak',     # Quercus robur
    8: 'Linden',  # Tilia sp.
    9: 'Alder',   # Alnus sp.
}


def load_point_cloud(file_path):
    """Load point cloud data from XYZ file, skipping header row if present."""
    try:
        # Try loading directly first
        try:
            points = np.loadtxt(file_path)
        except ValueError:
            # If it fails, try skipping the first row (header)
            points = np.loadtxt(file_path, skiprows=1)
        return points[:, :3]  # Take only x, y, z coordinates
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def point_selection(point_cloud_path, target_point_count):
    """Downsample point cloud to target point count using farthest point sampling."""
    try:
        points = load_point_cloud(point_cloud_path)
        if len(points) < target_point_count:
            logger.warning(f"Point cloud {point_cloud_path} has fewer points ({len(points)}) than target ({target_point_count})")
            # Duplicate points to reach target count
            points = np.repeat(points, (target_point_count // len(points)) + 1, axis=0)[:target_point_count]
            return points

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        pcd_down = point_cloud.farthest_point_down_sample(target_point_count)
        return np.asarray(pcd_down.points)
    except Exception as e:
        logger.error(f"Error in point selection for {point_cloud_path}: {str(e)}")
        raise


def load_helials_data(num_points, data_path):
    """Load HeliALS point cloud data using CSV metadata.

    Args:
        num_points: Number of points to sample from each point cloud
        data_path: Path to the HeliALS data directory

    Returns:
        Path to the created H5 file
    """
    csv_path = os.path.join(data_path, "training-and-test-segments-with-species.csv")
    h5_path = os.path.join(data_path, "helials_data.h5")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("=" * 80)
    logger.info(f"Loading HeliALS data from: {data_path}")
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Target points per cloud: {num_points}")
    logger.info("=" * 80)
    print(f"\n{'='*80}")
    print(f"Loading HeliALS dataset")
    print(f"{'='*80}")

    # Read CSV metadata
    df = pd.read_csv(csv_path)
    logger.info(f"CSV contains {len(df)} rows")
    print(f"\nCSV contains {len(df)} rows")

    # Get class names from species codes
    classes = [SPECIES_NAMES[i] for i in range(1, 10)]  # codes 1-9
    logger.info(f"Classes: {classes}")
    print(f"Classes: {classes}")

    point_clouds = []
    labels = []
    test_flags = []
    skipped_files = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading point clouds"):
        # Construct filename: {test_site_name}_{segment_id}.xyz
        site_name = row['test_site_name']
        segment_id = row['segment_id']
        xyz_filename = f"{site_name}_{segment_id}.xyz"
        xyz_path = os.path.join(data_path, xyz_filename)

        if not os.path.exists(xyz_path):
            logger.warning(f"File not found: {xyz_path}")
            skipped_files += 1
            continue

        try:
            # Load and downsample point cloud
            points = point_selection(xyz_path, num_points)
            point_clouds.append(points)

            # Label is species_code - 1 (0-indexed)
            label = int(row['species_code']) - 1
            labels.append(label)

            # Store test flag
            test_flags.append(int(row['test_set_flag']))

        except Exception as e:
            logger.error(f"Error processing {xyz_path}: {str(e)}")
            skipped_files += 1
            continue

    if not point_clouds:
        raise ValueError("No valid point cloud data was loaded")

    logger.info(f"\nLoaded {len(point_clouds)} point clouds, skipped {skipped_files} files")
    print(f"\n✓ Loaded {len(point_clouds)} point clouds, skipped {skipped_files} files")

    # Convert to numpy arrays
    point_clouds = np.array(point_clouds)
    labels = np.array(labels)
    test_flags = np.array(test_flags)

    logger.info(f"Point clouds shape: {point_clouds.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Test flags: {np.sum(test_flags == 1)} test, {np.sum(test_flags == 0)} train")
    print(f"  Point clouds: {point_clouds.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Test flags: {np.sum(test_flags == 1)} test, {np.sum(test_flags == 0)} train")

    # Save to H5 file
    logger.info(f"Saving to H5 file: {h5_path}")
    print(f"\n💾 Saving to {h5_path}...")

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('point_clouds', data=point_clouds)
        f.create_dataset('labels', data=labels)
        f.create_dataset('test_flags', data=test_flags)
        f.create_dataset('classes', data=np.array(classes, dtype='S'))

    file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    logger.info(f"Data saved successfully. H5 file size: {file_size_mb:.2f} MB")
    print(f"✓ Data saved successfully ({file_size_mb:.2f} MB)")
    print("=" * 80)

    return h5_path


def create_splits(h5_path, data_path, val_ratio=0.1, random_state=42):
    """Create train/val/test splits based on test_set_flag.

    Args:
        h5_path: Path to the H5 file with all data
        data_path: Directory to save split file
        val_ratio: Ratio of train pool to use for validation (default: 0.1)
        random_state: Random seed for reproducibility
    """
    split_h5_path = os.path.join(data_path, 'data_split_helials.h5')

    logger.info("=" * 80)
    logger.info(f"Creating train/val/test splits")
    logger.info(f"Validation ratio from train pool: {val_ratio:.1%}")
    print(f"\n{'='*80}")
    print(f"🔄 Creating train/val/test splits")
    print(f"{'='*80}")

    # Load data
    with h5py.File(h5_path, 'r') as f:
        point_clouds = f['point_clouds'][:]
        labels = f['labels'][:]
        test_flags = f['test_flags'][:]
        classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

    # Split by test_set_flag
    # flag = 0 -> train pool
    # flag = 1 -> test set
    train_pool_mask = test_flags == 0
    test_mask = test_flags == 1

    train_pool_pcs = point_clouds[train_pool_mask]
    train_pool_labels = labels[train_pool_mask]
    test_pcs = point_clouds[test_mask]
    test_labels = labels[test_mask]

    logger.info(f"Train pool (flag=0): {len(train_pool_pcs)} samples")
    logger.info(f"Test set (flag=1): {len(test_pcs)} samples")
    print(f"\nTrain pool (flag=0): {len(train_pool_pcs)} samples")
    print(f"Test set (flag=1): {len(test_pcs)} samples")

    # Split train pool into train and val (stratified)
    train_pcs, val_pcs, train_labels, val_labels = train_test_split(
        train_pool_pcs, train_pool_labels,
        test_size=val_ratio,
        stratify=train_pool_labels,
        random_state=random_state
    )

    logger.info(f"After split:")
    logger.info(f"  Train: {len(train_pcs)} samples")
    logger.info(f"  Val: {len(val_pcs)} samples")
    logger.info(f"  Test: {len(test_pcs)} samples")
    print(f"\n✂️  After split:")
    print(f"  Train: {len(train_pcs)} samples ({len(train_pcs)/len(point_clouds)*100:.1f}%)")
    print(f"  Val: {len(val_pcs)} samples ({len(val_pcs)/len(point_clouds)*100:.1f}%)")
    print(f"  Test: {len(test_pcs)} samples ({len(test_pcs)/len(point_clouds)*100:.1f}%)")

    # Save to H5 file
    logger.info(f"Saving split data to: {split_h5_path}")
    print(f"\n💾 Saving split data to {split_h5_path}...")

    with h5py.File(split_h5_path, 'w') as f:
        # Store classes
        f.create_dataset('classes', data=np.array(classes, dtype='S'))

        # Store metadata
        f.attrs['val_ratio'] = val_ratio
        f.attrs['random_state'] = random_state

        # Train data
        train_group = f.create_group('train')
        train_group.create_dataset('point_clouds', data=train_pcs)
        train_group.create_dataset('labels', data=train_labels)

        # Validation data
        val_group = f.create_group('val')
        val_group.create_dataset('point_clouds', data=val_pcs)
        val_group.create_dataset('labels', data=val_labels)

        # Test data
        test_group = f.create_group('test')
        test_group.create_dataset('point_clouds', data=test_pcs)
        test_group.create_dataset('labels', data=test_labels)

    # Print class distribution for each split
    print(f"\n📊 Class distribution:")
    for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n  {split_name}:")
        for u, c in zip(unique, counts):
            print(f"    {classes[u]}: {c} ({c/len(split_labels)*100:.1f}%)")

    file_size_mb = os.path.getsize(split_h5_path) / (1024 * 1024)
    logger.info(f"Split data saved successfully. File size: {file_size_mb:.2f} MB")
    print(f"\n✓ Split data saved successfully ({file_size_mb:.2f} MB)")
    print("=" * 80 + "\n")

    return split_h5_path


@DATASETS.register_module()
class TreeSpeciesDatasetHELIALS(Dataset):
    """HeliALS Tree Species Dataset.

    Loads tree species point cloud data from HeliALS dataset.
    Supports train/val/test partitions based on CSV metadata.
    """

    def __init__(self, config):
        self.num_points = config.N_POINTS
        self.partition = config.subset
        data_path = config.DATA_PATH

        # Get random seed for reproducibility (default: 42)
        self.seed = getattr(config, 'seed', 42)

        # Get validation ratio (default: 0.1)
        self.val_ratio = getattr(config, 'val_ratio', 0.1)

        try:
            split_h5_path = os.path.join(data_path, 'data_split_helials.h5')

            if not os.path.exists(split_h5_path):
                logger.info(f"Split file not found. Creating data split...")
                print(f"\n⚠️  Split file not found at {split_h5_path}")
                print("Creating data split (this may take a while)...")

                # First load raw data
                h5_path = os.path.join(data_path, "helials_data.h5")
                if not os.path.exists(h5_path):
                    h5_path = load_helials_data(self.num_points, data_path)

                # Then create splits
                create_splits(h5_path, data_path, val_ratio=self.val_ratio, random_state=self.seed)
            else:
                # Check if existing split has matching parameters
                with h5py.File(split_h5_path, 'r') as f:
                    existing_seed = f.attrs.get('random_state', None)
                    existing_val_ratio = f.attrs.get('val_ratio', None)

                # Recreate if parameters mismatch
                if existing_seed != self.seed or existing_val_ratio != self.val_ratio:
                    logger.info(f"Existing split has different parameters. Recreating...")
                    print(f"\n⚠️  Existing split has different parameters. Recreating...")
                    h5_path = os.path.join(data_path, "helials_data.h5")
                    if not os.path.exists(h5_path):
                        h5_path = load_helials_data(self.num_points, data_path)
                    create_splits(h5_path, data_path, val_ratio=self.val_ratio, random_state=self.seed)
                else:
                    logger.info(f"Loading existing split from: {split_h5_path}")
                    print(f"✓ Using existing split file: {split_h5_path}")

            # Load the split data
            with h5py.File(split_h5_path, 'r') as f:
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

                if self.partition not in f:
                    available = [k for k in f.keys() if k != 'classes']
                    raise ValueError(f"Partition '{self.partition}' not found. Available: {available}")

                self.data = f[self.partition]['point_clouds'][:]
                self.label = f[self.partition]['labels'][:]
                print(f"✓ Loaded {len(self.data)} samples from '{self.partition}' partition ({len(self.classes)} classes)")

            # Ensure labels are properly shaped
            if self.label.ndim > 1:
                self.label = self.label.flatten()

            logger.info(f"Dataset loaded: {len(self.data)} samples in '{self.partition}' partition")
            logger.info(f"Number of classes: {len(self.classes)}")

        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __getitem__(self, item):
        try:
            pointcloud = self.data[item][:self.num_points].copy()
            label = self.label[item]

            # Extract label value robustly
            if isinstance(label, (np.ndarray, list, tuple)):
                label_value = int(label[0] if len(label) > 0 else label)
            else:
                label_value = int(label)

            # Normalize point cloud to unit sphere
            pointcloud = normalize_pc(pointcloud)

            return 'HeliALS', 'sample', (pointcloud.astype(np.float32), label_value)
        except Exception as e:
            logger.error(f"Error getting item {item}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)


def normalize_pc(points):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points


def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and display class distribution in a dataset."""
    class_counts = {}
    total_samples = len(dataset)

    for i in range(len(dataset)):
        label = int(dataset.label[i])
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Class':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)

    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_name:<20} {count:<10} {percentage:>6.2f}%")

    print("-" * 50)
    print(f"Total samples: {total_samples}\n")

    return class_counts


if __name__ == '__main__':
    try:
        # Test with HELIALS dataset
        train_cfg = SimpleNamespace(N_POINTS=2048, subset='train', DATA_PATH='data/HELIALS')
        val_cfg = SimpleNamespace(N_POINTS=2048, subset='val', DATA_PATH='data/HELIALS')
        test_cfg = SimpleNamespace(N_POINTS=2048, subset='test', DATA_PATH='data/HELIALS')

        train_dataset = TreeSpeciesDatasetHELIALS(train_cfg)
        val_dataset = TreeSpeciesDatasetHELIALS(val_cfg)
        test_dataset = TreeSpeciesDatasetHELIALS(test_cfg)

        print(f"\nClasses: {train_dataset.classes}")

        # Analyze class distributions
        analyze_class_distribution(train_dataset, "Training Set Distribution")
        analyze_class_distribution(val_dataset, "Validation Set Distribution")
        analyze_class_distribution(test_dataset, "Test Set Distribution")

        # Verify data loading
        _, _, (sample_data, sample_label) = train_dataset[0]
        print(f"\nSample point cloud shape: {sample_data.shape}")
        print(f"Sample label: {sample_label} ({train_dataset.classes[sample_label]})")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
