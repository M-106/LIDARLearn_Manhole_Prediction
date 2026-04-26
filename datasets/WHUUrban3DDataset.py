import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS



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
        features = np.hstack([
            coordinates,
            intensities.reshape(-1, 1)
        ]).astype(np.float32)

        # filter labels -> only instances with label 104002
        mask = semantics == 104002
        if instance_segmentation:
            labels = np.full(instances.shape, -1, dtype=np.int64)

            # only take valid instances
            instances_filtered = instances[mask]

            # reindex
            # [1040023, 1040023, 5550001, 5550001, 9999999] -> [0, 0, 1, 1, 2]
            _, new_ids = np.unique(instances_filtered, return_inverse=True)

            labels[mask] = new_ids
        else:
            labels = np.zeros(semantics.shape, dtype=np.int64)
            labels[mask] = 1

        return features, labels
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't load '{file_}' as point-cloud.")


@DATASETS.register_module()
class WHUUrban3DDataset(Dataset):
    """
    Note: This is an changed version of the 
    dataset implemented in https://github.com/M-106/MCR-Lab/blob/main/src/mcrlab/point_cloud/data.py
    """
    def __init__(self, config):
        self.path = os.path.join(config.DATA_PATH, "mls", "h5")
        self.num_point = int(getattr(config, 'N_POINTS', 4096))
        self.block_size = float(getattr(config, 'block_size', 1.0))
        self.sample_rate = float(getattr(config, 'sample_rate', 1.0))
        self.partition = config.subset    # 'train'|'val'|'test'
        self.test_area = int(getattr(config, 'test_area', 5))
        self.feature_mode = str(getattr(config, 'feature_mode', 'xyz_i'))

        self.transform = None

        # train normally include: '0404' but added to val, so that it have at least one
        self.train_ids = ['0424', '0434', '0444', '0940', '0947', '2002', '2321', '2322', '2422', '2447', '2719', '3405', '3648', '3918', '4333', '4629', '4938', '5642', '6017', '6027', '6037', '8018', '1046', '0414', '0502']
        self.val_ids = ['2323', '8008', '8038', '2421', '0404']  # FIXME -> '0404'
        self.test_ids = ['2323', '2522', '2810', '5627', '8008', '8038', '2421', '2423', '2521']

        self.partition_to_ids = {
            'train': self.train_ids,
            'val': self.val_ids,
            'test': self.test_ids
        }

        self.point_cloud_paths = []
        preprocesed_path = os.path.join(self.path, "preprocessed")
        preprocessed = False  # maybe fix in future to make preprocessed WHU dataset possible
        path = preprocesed_path if preprocessed else self.path
        for cur_file in os.listdir(path):
            # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
            if cur_file.endswith((".h5", ".ply")):
                if preprocessed and not cur_file.startswith("preprocessed_"):
                    continue
                elif not preprocessed and cur_file.startswith("preprocessed_"):
                    continue

                if any([cur_file.startswith(cur_id) for cur_id in self.partition_to_ids[self.partition]]):
                    self.point_cloud_paths.append(os.path.join(path, cur_file))

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud_path = self.point_cloud_paths[idx]
        _, scene_name = os.path.split(point_cloud_path)
        scene_name = ".".join(scene_name.split(".")[:-1])

        features, labels = load_h5_as_numpy(point_cloud_path, instance_segmentation=False)

        if self.transform:
            features = self.transform(features)

        # reduce size to fix size
        current_points = features.shape[0]
        goal_points = self.num_point

        if current_points != goal_points:

            # pre-height filtering
            if current_points > goal_points:
                z_threshold = 50.0  # threshol ok? => 100 cm?
                mask = features[:, 2] <= z_threshold  # only keep points with z <= 0.1
                features = features[mask]
                labels = labels[mask]

                # update current updated points
                current_points = features.shape[0]
                if current_points == 0:
                    raise ValueError("All Points got removed during z-filter downsampling.")

            if current_points > goal_points:
                weights = np.ones(current_points)
                weights[labels == 1] = 10.0
                probabilities = weights / weights.sum()
        
                # downsampling
                sample_idx = np.random.choice(current_points, 
                                              size=goal_points, 
                                              replace=False,
                                              p=probabilities)
            elif current_points < goal_points:
                # upsampling
                extra = np.random.choice(current_points, size=(goal_points-current_points), replace=True)
                sample_idx = np.concatenate([np.arange(current_points), extra])
            
            # apply down or upsampling
            features = features[sample_idx]
            labels = labels[sample_idx]


        # convert to torch
        features = torch.from_numpy(features).float()      # (N, 4)
        labels = torch.from_numpy(labels).long()           # (N,)
        
        return 'WHUUrban3D', scene_name, (features, labels)



















