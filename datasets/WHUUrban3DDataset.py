import os
import shutil
import random
from datetime import datetime

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (load_h5_as_numpy,
                    create_sliding_window_patch,
                    random_hex_color,
                    save_manhole_visualization,
                    save_patch_visualization,
                    extract_samples,
                    normalize_features,
                    augment_point_cloud)
from .build import DATASETS






@DATASETS.register_module()
class WHUUrban3DDataset(Dataset):
    """
    Note: This is an changed version of the
    dataset implemented in https://github.com/M-106/MCR-Lab/blob/main/src/mcrlab/point_cloud/data.py

    FIXME -> need evaluation mode and also an logits mean or majority voting for overlap prediction of test/val preds
    """

    def __init__(self, config, is_primary_dataset=True):
        super().__init__()
        self.is_primary_dataset = is_primary_dataset

        preprocessed = True
        self.path = os.path.join(config.DATA_PATH, "mls", "h5")
        if preprocessed:
            self.path = os.path.join(self.path, "preprocessed")
        # print(self.path)
        # os.system("echo 'Data Dir Test'")
        # os.system(f"ls {self.path}")
        self.num_point = int(getattr(config, "N_POINTS", 4096))
        self.block_size = float(getattr(config, "block_size", 5.0))
        self.sample_rate = float(getattr(config, "sample_rate", 1.0))
        self.partition = config.subset  # 'train'|'val'|'test'
        self.test_area = int(getattr(config, "test_area", 5))
        self.feature_mode = str(getattr(config, "feature_mode", "xyz_i"))
        self.intensity_dropout = float(getattr(config, "intensity_dropout", 0.4))

        if not hasattr(config, "intensity_dropout"):
            print("[WARNING] Config does not have 'intensity_dropout' attribute!")
        # raise RuntimeError("[WARNING] Config does not have 'intensity_dropout' attribute!")

        self.epoch = 0
        self.transform = None

        print(f"Got partition: {self.partition}")

        # train normally include: '0404' but added to val, so that it have at least one
        self.train_ids = [
            "8018",
            "4938",
            "0414",
            "2002",
            "0444",
            "1046",
            "5642",
            "4333",
            "4629",
            "0424",
            "2421",
            "0947",
            "0434",
            "2022",
            "2719",
            "2810",
            "8048",
            "2423",
            "2522",
            "8008",
            "0502",
            "6017",
            "3918",
            "2422",
            "2322",
            "3405",
            "2323",
            "8038",
        ]
        self.val_ids = ["0404", "6027", "3648"]
        self.test_ids = ["0940", "2447", "6037", "2321", "8028", "5627", "2521"]

        if preprocessed:
            self.train_ids = ["preprocessed_patch_" + x for x in self.train_ids]
            self.val_ids = ["preprocessed_patch_" + x for x in self.val_ids]
            self.test_ids = ["preprocessed_patch_" + x for x in self.test_ids]

        self.partition_to_ids = {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids,
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

                if any(
                    [
                        cur_file.startswith(cur_id)
                        for cur_id in self.partition_to_ids[self.partition]
                    ]
                ):
                    self.point_cloud_paths.append(os.path.join(self.path, cur_file))

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

        if self.partition == "train" and self.is_primary_dataset:
            # Single-Sample_Overfit FIXME
            now = datetime.now()

            year = now.year
            month = now.month
            day = now.day
            hour = now.hour
            minute = now.minute

            self.debug_out_path = f"./debugging/debugging_{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{config.exp_name}"

            # for file in os.listdir("."):
            #     if file.startswith("debugging_"):
            #         shutil.rmtree(os.path.join(".", file))
            # raise ValueError("DEBUGGING REMOVAL")

            # shutil.rmtree(f"./debugging/")
            # shutil.rmtree(f"./experiments/")
            # os.makedirs(f"./debugging/", exist_ok=True)
            # os.makedirs(f"./experiments/", exist_ok=True)
            # raise ValueError("DEBUGGING REMOVAL")
            # ls ./src/LIDARLearn_Manhole_Prediction/debugging

            os.makedirs(self.debug_out_path, exist_ok=True)
            shutil.rmtree(self.debug_out_path)
            os.makedirs(self.debug_out_path, exist_ok=True)

            self.debug_log_path = os.path.join(self.debug_out_path, "log_.txt")
            with open(self.debug_log_path, "w") as file_:
                file_.write(f"Logging from '{self.debug_out_path}'")

            self.original_point_cloud_paths = self.point_cloud_paths
            # self.point_cloud_paths_8_manholes = extract_samples(self.point_cloud_paths, amount=8)
            # self.point_cloud_paths_16_manholes = extract_samples(self.point_cloud_paths, amount=16)
            # self.point_cloud_paths_32_manholes = extract_samples(self.point_cloud_paths, amount=32)
            self.point_cloud_paths_manholes_only = extract_samples(self.point_cloud_paths, amount=-1)

            self.point_cloud_paths = self.point_cloud_paths_manholes_only
            # self.point_cloud_paths = extract_samples(self.point_cloud_paths, amount=3)
            print(f"Updated to {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        cur_pc_path = self.point_cloud_paths[idx]

        _, scene_name = os.path.split(cur_pc_path)
        scene_name = ".".join(scene_name.split(".")[:-1]).replace("preprocessed_patch_", "")

        features, labels = load_h5_as_numpy(
            cur_pc_path, instance_segmentation=False
        )

        # augmentation
        if self.partition == "train":
            features, labels = augment_point_cloud(features, labels, intensity_dropout=self.intensity_dropout)

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
                sample_idx = np.random.choice(
                    current_points, size=goal_points, replace=False, p=probabilities
                )
            elif current_points < goal_points:
                # upsampling
                extra = np.random.choice(
                    current_points, size=(goal_points - current_points), replace=True
                )
                sample_idx = np.concatenate([np.arange(current_points), extra])

            # apply down or upsampling
            features = features[sample_idx]
            labels = labels[sample_idx]

        # if self.partition == "train" and idx < 3 and self.is_primary_dataset:
        #     save_patch_visualization(
        #         features,
        #         labels,
        #         save_path=f"{self.debug_out_path}/patch_viz_{scene_name}.png",  # _epoch_{self.epoch}
        #         title="Patch Investigation",
        #     )

        # normalize
        features = normalize_features(features, debug_prints=False)

        # convert to torch
        features = torch.from_numpy(features).float()  # (N, 4)
        labels = torch.from_numpy(labels).long()  # (N,)

        # normalize
        # features[:, -1] = features[:, -1] - features[:, -1].mean(0)
        # features[:, -1] = features[:, -1] / features[:, -1].abs().max() # normalize

        return "WHUUrban3D", scene_name, (features, labels)

    def epoch_update(self, epoch):
        self.epoch = epoch

        # if self.partition == "train" and self.epoch < 50:
        #     self.point_cloud_paths = self.point_cloud_paths_8_manholes
        # elif self.partition == "train" and self.epoch >= 50 and self.epoch < 100:
        #     self.point_cloud_paths = self.point_cloud_paths_16_manholes
        # elif self.partition == "train" and self.epoch >= 100 and self.epoch < 150:
        #     self.point_cloud_paths = self.point_cloud_paths_32_manholes
        # elif self.partition == "train" and self.epoch >= 150 and self.epoch < 200:
        #     self.point_cloud_paths = self.point_cloud_paths_all_manholes
        # elif self.partition == "train" and self.epoch >= 200:
        #     self.point_cloud_paths = self.original_point_cloud_paths

        if epoch < 200:
            self.point_cloud_paths = self.point_cloud_paths_manholes_only
        else:
            self.point_cloud_paths = self.original_point_cloud_paths







