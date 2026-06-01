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
            instances = file_["instances"][:]
            intensities = file_["intensity"][:]
            semantics = file_["semantics"][:]
            number_returns = file_["number_returns"][:]

        intensities = intensities.astype(np.float32)
        if intensities.max() > 1.0:
            intensities = intensities / 255

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


def extract_samples(point_cloud_paths, amount=3) -> list:
    samples = []

    for cur_pc_path in point_cloud_paths:
        _, scene_name = os.path.split(cur_pc_path)
        scene_name = ".".join(scene_name.split(".")[:-1])

        features, labels = load_h5_as_numpy(
            cur_pc_path, instance_segmentation=False
        )

        if np.sum(labels == 1) > 100:
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








