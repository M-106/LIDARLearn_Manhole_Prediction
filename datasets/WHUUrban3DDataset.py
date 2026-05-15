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



def create_sliding_window_patch(point_cloud_paths,
                                block_size,
                                overlap=0.5):  # in percentage
    stride = block_size * (1 - overlap)
    patches = []

    for cur_pc_path in point_cloud_paths:
        features, labels = load_h5_as_numpy(
            cur_pc_path,
            instance_segmentation=False
        )

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
                    (coords[:, 0] >= x0) &
                    (coords[:, 0] < x1) &
                    (coords[:, 1] >= y0) &
                    (coords[:, 1] < y1)
                )

                point_indices = np.where(mask)[0]

                # skip empty patches
                if len(point_indices) < 100:
                    continue

                patches.append({
                    "point_cloud_path": cur_pc_path,
                    "indices": point_indices
                })
    return patches



@DATASETS.register_module()
class WHUUrban3DDataset(Dataset):
    """
    Note: This is an changed version of the 
    dataset implemented in https://github.com/M-106/MCR-Lab/blob/main/src/mcrlab/point_cloud/data.py
    
    FIXME -> need evaluation mode and also an logits mean or majority voting for overlap prediction of test/val preds
    """
    def __init__(self, config):
        preprocessed = True
        self.path = os.path.join(config.DATA_PATH, "mls", "h5")
        if preprocessed:
            self.path = os.path.join(self.path, "preprocessed")
        print(self.path)
        os.system("echo 'Data Dir Test'")
        os.system(f"ls {self.path}")
        self.num_point = int(getattr(config, 'N_POINTS', 4096))
        self.block_size = float(getattr(config, 'block_size', 5.0))
        self.sample_rate = float(getattr(config, 'sample_rate', 1.0))
        self.partition = config.subset    # 'train'|'val'|'test'
        self.test_area = int(getattr(config, 'test_area', 5))
        self.feature_mode = str(getattr(config, 'feature_mode', 'xyz_i'))

        self.transform = None

        print(f"Got partition: {self.partition}")

        # train normally include: '0404' but added to val, so that it have at least one
        self.train_ids = ['8018', '4938', '0414', '2002', '0444', '1046', '5642', '4333', '4629', '0424', '2421', '0947', '0434', '2022', '2719', '2810', '8048', '2423', '2522', '8008', '0502', '6017', '3918', '2422', '2322', '3405', '2323', '8038']
        self.val_ids = ['0404', '6027', '3648']
        self.test_ids = ['0940', '2447', '6037', '2321', '8028', '5627', '2521']

        if preprocessed:
            self.train_ids = ['preprocessed_'+x for x in self.train_ids]
            self.val_ids = ['preprocessed_'+x for x in self.val_ids]
            self.test_ids = ['preprocessed_'+x for x in self.test_ids]

        self.partition_to_ids = {
            'train': self.train_ids,
            'val': self.val_ids,
            'test': self.test_ids
        }

        self.point_cloud_paths = []
        # preprocesed_path = os.path.join(self.path, "preprocessed")
          # maybe fix in future to make preprocessed WHU dataset possible
        # path = preprocesed_path if preprocessed else self.path
        for cur_file in os.listdir(self.path):
            # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
            # print(f"File name: {cur_file}")
            if cur_file.endswith((".h5", ".ply")):
                if preprocessed and not cur_file.startswith("preprocessed_"):
                    continue
                elif not preprocessed and cur_file.startswith("preprocessed_"):
                    continue

                if any([cur_file.startswith(cur_id) for cur_id in self.partition_to_ids[self.partition]]):
                    self.point_cloud_paths.append(os.path.join(self.path, cur_file))

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

        # create sliding window patches
        if self.partition != "train":
            self.patches = create_sliding_window_patch(
                point_cloud_paths=self.point_cloud_paths,
                block_size=self.block_size,
                overlap=0.5
            )

    def __len__(self):
        if self.partition == "train":
            return len(self.point_cloud_paths) * 10
        else:
            return len(self.patches)

    def __getitem__(self, idx):
        if self.partition == "train":
            idx = int(idx / 10)
            point_cloud_path = self.point_cloud_paths[idx]
            _, scene_name = os.path.split(point_cloud_path)
            scene_name = ".".join(scene_name.split(".")[:-1])

            features, labels = load_h5_as_numpy(point_cloud_path, instance_segmentation=False)
        
            # choose local patch -> else too big
            # no point reduction wanted
            if np.any(labels == 1):
                probs = np.ones(len(features))
                probs[labels == 1] = 5.0

                probs = probs / probs.sum()
                center_idx = np.random.choice(len(features), p=probs)
            else:
                center_idx = np.random.randint(len(features))
        
            center = features[center_idx, :3]

            radius = self.block_size

            dist = np.linalg.norm(features[:, :3] - center, axis=1)
            mask = dist < radius

            features = features[mask]
            labels = labels[mask]

            # print(f"[DEBUGGING] Point Amount: {features.shape} → changed to {self.num_point}")
        else:
            # center_idx = np.random.randint(len(features))
            patch_info = self.patches[idx]

            point_cloud_path = patch_info["point_cloud_path"]
            patch_indices = patch_info["indices"]

            _, scene_name = os.path.split(point_cloud_path)
            scene_name = ".".join(scene_name.split(".")[:-1])
        
            features, labels = load_h5_as_numpy(
                point_cloud_path,
                instance_segmentation=False
            )

            features = features[patch_indices]
            labels = labels[patch_indices]

        # preprocessing, augmentation
        if self.transform:
            features = self.transform(features)

        # reduce size to fix size
        current_points = features.shape[0]
        goal_points = self.num_point

        if current_points != goal_points:

            if current_points > goal_points:
                weights = np.ones(current_points)
                # weights[labels == 1] = 500.0
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



















