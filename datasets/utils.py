import os
import random

import h5py
import numpy as np
import torch

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt



def load_h5_as_numpy(path, instance_segmentation=False):
    if path.endswith(".h5"):
        with h5py.File(path, "r") as file_:
            # extract data
            coordinates = file_["coords"][:]
            # instances = file_["instances"][:]
            intensities = file_["intensity"][:]
            semantics = file_["semantics"][:]
            # number_returns = file_["number_returns"][:]

        intensities = intensities.astype(np.float32)
        # if intensities.max() > 1.0:
        #     intensities = intensities / 255

        # stack features -> [N, 4]
        features = np.hstack([coordinates, intensities.reshape(-1, 1)]).astype(
            np.float32
        )

        labels = semantics

        # # filter labels -> only instances with label 104002
        # mask = semantics == 104002
        # if instance_segmentation:
        #     labels = np.full(instances.shape, -1, dtype=np.int64)

        #     # only take valid instances
        #     instances_filtered = instances[mask]

        #     # reindex
        #     # [1040023, 1040023, 5550001, 5550001, 9999999] -> [0, 0, 1, 1, 2]
        #     _, new_ids = np.unique(instances_filtered, return_inverse=True)

        #     labels[mask] = new_ids
        # else:
        #     labels = np.zeros(semantics.shape, dtype=np.int64)
        #     labels[mask] = 1

        return features, labels
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't load '{file_}' as point-cloud.")


def create_sliding_window_patch(
    point_cloud_paths, block_size, overlap=0.5
):  # in percentage
    stride = block_size * (1 - overlap)
    patches = []

    for cur_pc_path in point_cloud_paths:
        features, labels = load_h5_as_numpy(cur_pc_path, instance_segmentation=False)

        coords = features[:, :3]

        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        # maybe change to np.linspace(...)
        x_points = np.arange(x_min, x_max + block_size, stride)
        y_points = np.arange(y_min, y_max + block_size, stride)

        for x0 in x_points:
            for y0 in y_points:

                x1 = x0 + block_size
                y1 = y0 + block_size

                mask = (
                    (coords[:, 0] >= x0)
                    & (coords[:, 0] < x1)
                    & (coords[:, 1] >= y0)
                    & (coords[:, 1] < y1)
                )

                point_indices = np.where(mask)[0]

                # skip empty patches
                if len(point_indices) < 100:
                    continue

                patches.append(
                    {"point_cloud_path": cur_pc_path, "indices": point_indices}
                )
    return patches


def extract_manholes(points: np.ndarray):
    db = DBSCAN(eps=0.15, min_samples=20).fit(points[:, :2])
    cluster_labels = db.labels_

    unique_labels = np.unique(cluster_labels)

    manholes = []
    for cur_label in unique_labels:
        if cur_label == -1:
            continue

        indices = np.where(cluster_labels == cur_label)[0]
        manholes.append(points[indices])

    return manholes


def random_hex_color():
    """
    Generates a random hex color code in the format #RRGGBB.
    """
    # Generate a random integer between 0x000000 and 0xFFFFFF
    color_int = random.randint(0, 0xFFFFFF)
    # Format as hex with leading zeros and prepend '#'
    return f"#{color_int:06X}"


def save_manhole_visualization(manholes, save_path, title="Cluster Investigation"):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, cur_manhole in enumerate(manholes):
        ax.scatter(
            cur_manhole[:, 0],
            cur_manhole[:, 1],
            # c=random_hex_color(),
            color=random_hex_color(),
            s=10,
        )

        center = cur_manhole.mean(axis=0)

        ax.text(center[0], center[1], str(i), fontsize=8)

        # print(f"Cluster {i}: {len(cur_manhole)} points")

    ax.set_aspect("equal")

    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)

    plt.close(fig)


def save_patch_visualization(features, labels, save_path, title="Patch Investigation"):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, cur_label in enumerate(np.unique(labels)):
        ax.scatter(
            features[labels == cur_label][:, 0],
            features[labels == cur_label][:, 1],
            # c=random_hex_color(),
            color=random_hex_color(),
            s=10,
        )

    # print("num points:", len(features))
    # print("pos ratio:", np.mean(labels == 1))

    ax.set_aspect("equal")

    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)

    plt.close(fig)

    # plot intensity

    fig, ax = plt.subplots(figsize=(7, 7))

    color = features[:, 3]
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    color = (color - np.min(color)) / (np.max(color) - np.min(color))
    # color = np.repeat(color[:, np.newaxis], 3, axis=1).squeeze()
    ax.scatter(
        features[:, 0],
        features[:, 1],
        # c=random_hex_color(),
        c=color,
        s=5,
        cmap="viridis"
    )

    # print("num points:", len(features))
    # print("pos ratio:", np.mean(labels == 1))

    ax.set_aspect("equal")

    plt.title(title)

    if save_path is not None:
        save_path_root, save_path_file = os.path.split(save_path)
        save_path = os.path.join(save_path_root, "intensities_"+save_path_file)
        plt.savefig(save_path)

    plt.close(fig)


def extract_samples(point_cloud_paths, amount=3, min_manhole_points=100) -> list:
    samples = []

    for cur_pc_path in point_cloud_paths:
        _, scene_name = os.path.split(cur_pc_path)
        scene_name = ".".join(scene_name.split(".")[:-1])

        features, labels = load_h5_as_numpy(
            cur_pc_path, instance_segmentation=False
        )

        if np.sum(labels == 1) > min_manhole_points:
            samples.append(cur_pc_path)

            if amount > 0 and len(samples) >= amount:
                break

    return samples



def normalize_features(features, debug_prints=False):
    """
    Patch-local normalization for point cloud features.
    """
    if debug_prints:
        print(f"  Input shape:  {features.shape}")
        print(f"  Input dtype:  {features.dtype}")
        print(f"  XYZ min/max:  {features[:, :3].min():.2f} / {features[:, :3].max():.2f}")
        print(f"  Intensity:    min={features[:, 3].min()}, max={features[:, 3].max()}, "
              f"p2={np.percentile(features[:, 3], 2):.1f}, p98={np.percentile(features[:, 3], 98):.1f}")
        print(f"  NaN in input: {np.isnan(features.astype(np.float32)).any()}")
        print(f"  Inf in input: {np.isinf(features.astype(np.float32)).any()}")

    # features = features.copy()
    features = features.astype(np.float32)

    # XYZ
    xyz = features[:, :3]
    centroid = xyz.mean(axis=0)
    xyz -= centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))

    if debug_prints:
        print(f"  XYZ scale:    {scale:.4f}  (0 → Normalisierung wird übersprungen!)")

    if scale > 0:
        xyz /= scale
    features[:, :3] = xyz

    # Intensity - uint16 range 0–65535, but now: float32
    intensity = features[:, 3]
    p2, p98 = np.percentile(intensity, [2, 98])
    denom = p98 - p2

    if debug_prints:
        print(f"  Intensity denom (p98-p2): {denom:.4f}  (< 1e-6 → wird auf 0 gesetzt!)")

    if denom > 1e-6:
        intensity = np.clip((intensity - p2) / denom, 0.0, 1.0)
    else:
        intensity = np.zeros_like(intensity)
    features[:, 3] = intensity

    if debug_prints:
        print(f"  Output shape: {features.shape}")
        print(f"  Output dtype: {features.dtype}")
        print(f"  NaN in output:{np.isnan(features).any()}")

    return features

SENSOR_CONFIGS = {
    'as-900hl': {  # WHU
        'intensity_p1': 1100.0, 
        'intensity_p99': 44403.0,
    },
    'riegl_vux-1ha_mls': {  # sud
        'intensity_p1': 1935.0, 
        'intensity_p99': 3384.0, 
    },
   'default': {'intensity_p1': 0.0, 'intensity_p99': 255.0, 'max_density': 100}
}

def normalize_features_robust(
    features,
    intensity_p1,
    intensity_p99,
    eps=1e-6,
    debug_prints=False
):
    """
    Standard Point Cloud Normalization:
    - XYZ: Centered at centroid and scaled into a unit sphere (max distance = 1.0)
    - Intensity: Scaled using global dataset percentiles mapped to [0, 1]
    """
    features = features.astype(np.float32, copy=True)
    
    # Spatial Coordinates (XYZ) -> Unit-Sphere Scaling
    xyz = features[:, :3]
    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid
    
    #     Calculate max distance from centroid (radius of unit sphere)
    distances = np.linalg.norm(xyz_centered, axis=1)
    scale = np.max(distances)
    
    if scale > eps:
        features[:, :3] = xyz_centered / scale
    else:
        features[:, :3] = np.zeros_like(xyz_centered)

    # Intensity -> Global Percentile Min-Max Normalization
    intensity = features[:, 3]
    denom = intensity_p99 - intensity_p1
    
    if denom > eps:
        norm_intensity = (intensity - intensity_p1) / denom
        features[:, 3] = np.clip(norm_intensity, 0.0, 1.0)
    else:
        features[:, 3] = 0.0

    # Save normalization parameters required for denormalization
    norm_params = {
        'centroid': centroid,
        'scale': scale,
        'intensity_p1': intensity_p1,
        'intensity_p99': intensity_p99
    }

    if debug_prints:
        print(f"Centroid: {centroid}")
        print(f"Scale (Sphere Radius): {scale:.4f}")
        print(f"Intensity Range: [{features[:, 3].min():.2f}, {features[:, 3].max():.2f}]")

    return features, norm_params


def denormalize_features_unit_sphere(features_norm, norm_params):
    """
    Reconstructs original spatial coordinates and global intensity scale.
    """
    features_orig = features_norm.astype(np.float32, copy=True)

    # Restore XYZ Coordinates
    # Reverse scaling and shift back by the centroid
    features_orig[:, :3] = (features_norm[:, :3] * norm_params['scale']) + norm_params['centroid']

    # Restore Intensity
    denom = norm_params['global_p99'] - norm_params['global_p1']
    features_orig[:, 3] = (features_norm[:, 3] * denom) + norm_params['global_p1']

    return features_orig



def augment_point_cloud(features, labels, intensity_dropout=0.0):
    """
    Standard augmentations for manhole detection.
    expected features: (N, 4) -> [x, y, z, intensity]
    """
    # 1. Random Rotation around the Z-axis (Vertical)
    # Manholes are circular, so 360-degree rotation is perfect.
    if np.random.random() > 0.8:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        features[:, :3] = features[:, :3] @ rotation_matrix.T

    # 2. Random Scaling
    # Simulates different sensor distances/resolutions
    if np.random.random() > 0.8:
        scale = np.random.uniform(0.9, 1.1)
        features[:, :3] *= scale

    # # 3. Random Jitter (Noise)
    # # Simulates sensor inaccuracy. Standard dev of 0.01m is typical.
    # if np.random.random() > 0.8:
    #     noise = np.random.normal(0, 0.005, size=(features.shape[0], 3))
    #     features[:, :3] += noise

    # 4. Random Flip
    if np.random.random() > 0.9:
        features[:, 0] = -features[:, 0]  # Flip X
    if np.random.random() > 0.9:
        features[:, 1] = -features[:, 1]  # Flip Y

    # 5. Random Intensity Shifting
    if np.random.random() > 0.9:
        shift = np.random.uniform(-0.1, 0.1)
        features[:, 3] = np.clip(features[:, 3] + shift, 0.0, 1.0)

    # 6. Random Intensity Dropout
    if np.random.random() < intensity_dropout:
        # num_points = len(features)
        # num_drop = int(num_points * intensity_dropout)
        # if num_drop > 0:
        #     drop_idx = np.random.choice(num_points, size=num_drop, replace=False)
        #     features[drop_idx, 3] = 0.0
        features[:, 3] = 0.0

    # 7. Random Intensity Noise
    # if np.random.random() > 0.5:
    #     intensity_noise = np.random.normal(0, 0.02, size=(features.shape[0],))
    #     features[:, 3] = np.clip(features[:, 3] + intensity_noise, 0.0, 1.0)

    return features, labels







