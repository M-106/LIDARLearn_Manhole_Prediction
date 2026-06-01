from copy import deepcopy

from torch.utils.data import Dataset

from .build import DATASETS

from .WHUUrban3DDataset import WHUUrban3DDataset
from .SUDROADDataset import SUDROADDataset



@DATASETS.register_module()
class MergedUrbanSUDDataset(Dataset):
    def __init__(self, config):
        super().__init__()
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

            self.debug_out_path = f"./debugging/debugging_{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{config.exp_name}"

            # shutil.rmtree(f"./debugging/debugging_2026_05_27_22_16_manhole_seg_ptv3_nl_loss")
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









