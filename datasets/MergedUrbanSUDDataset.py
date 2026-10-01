import os
import shutil
import random
from datetime import datetime
from copy import deepcopy

from torch.utils.data import Dataset

from .WHUUrban3DDataset import WHUUrban3DDataset
from .SUDROADDataset import SUDROADDataset

from .build import DATASETS

# from helper import save_dir_creation



@DATASETS.register_module()
class MergedUrbanSUDDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        self.num_point = int(getattr(config, "N_POINTS", 4096))
        self.block_size = float(getattr(config, "block_size", 5.0))
        self.sample_rate = float(getattr(config, "sample_rate", 1.0))
        self.partition = config.subset  # 'train'|'val'|'test'
        self.test_area = int(getattr(config, "test_area", 5))
        self.feature_mode = str(getattr(config, "feature_mode", "xyzi"))
        self.intensity_dropout = float(getattr(config, "intensity_dropout", 0.5))

        self.urban_config = deepcopy(config)
        self.urban_config.DATA_PATH = config.URBAN_DATA_PATH
        self.urban3d = WHUUrban3DDataset(self.urban_config, is_primary_dataset=False)

        self.sud_config = deepcopy(config)
        self.sud_config.DATA_PATH = config.SUD_DATA_PATH
        self.sud = SUDROADDataset(self.sud_config, is_primary_dataset=False)


        if self.partition == "train":
            # Single-Sample_Overfit FIXME
            now = datetime.now()

            year = now.year
            month = now.month
            day = now.day
            hour = now.hour
            minute = now.minute

            self.debug_out_path = f"/out/checkpoints/3d/debugging/debugging_{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{config.exp_name}"

            # shutil.rmtree(f"./debugging/debugging_2026_05_27_22_16_manhole_seg_ptv3_nl_loss")
            # raise ValueError("DEBUGGING REMOVAL")
            # ls ./src/LIDARLearn_Manhole_Prediction/debugging

            # save_dir_creation(self.debug_out_path)
            os.makedirs(self.debug_out_path, exist_ok=True)
            shutil.rmtree(self.debug_out_path)
            # save_dir_creation(self.debug_out_path)
            os.makedirs(self.debug_out_path, exist_ok=True)

            self.debug_log_path = os.path.join(self.debug_out_path, "log_.txt")
            with open(self.debug_log_path, "w") as file_:
                file_.write(f"Logging from '{self.debug_out_path}'")

            print(f"Updated to {len(self.urban3d) + len(self.sud)} point clouds.")

    def __len__(self):
        return len(self.urban3d) + len(self.sud)

    def __getitem__(self, idx):
        len_urban3d = len(self.urban3d)
        if idx < len_urban3d:
            return self.urban3d[idx]
        else:
            new_idx = idx - len_urban3d
            return self.sud[new_idx]

    def epoch_update(self, epoch):
        self.urban3d.epoch_update(epoch)
        self.sud.epoch_update(epoch)









