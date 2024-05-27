import os

import torch

from .common import BaseDataset


def balance_with_actions(samples, increase_factor=5, exceptions=None):
    if exceptions is None:
        exceptions = [2, 3]
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if each_sample["cmd"] not in exceptions:
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


def resample_complete_samples(samples, increase_factor=5):
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if (each_sample["speed"] and each_sample["angle"] and each_sample["z"] > 0
                    and 0 < each_sample["goal"][0] < 1600 and 0 < each_sample["goal"][1] < 900):
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


class NuScenesDataset(BaseDataset):
    def __init__(self, data_root="data/nuscenes", anno_file="data/nuScenes_vista.json",
                 target_height=320, target_width=576, num_frames=25):
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        print("nuScenes loaded:", len(self))
        self.samples = balance_with_actions(self.samples, increase_factor=5)
        print("nuScenes balanced:", len(self))
        self.samples = resample_complete_samples(self.samples, increase_factor=2)
        print("nuScenes resampled:", len(self))
        self.action_mod = 0

    def get_image_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["frames"][current_index])

    def build_data_dict(self, image_seq, sample_dict):
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug
        }
        if self.action_mod == 0:
            data_dict["command"] = torch.tensor(sample_dict["cmd"])
        elif self.action_mod == 1:
            data_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
        elif self.action_mod == 2:
            # scene might be empty
            if sample_dict["speed"]:
                data_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
            # scene might be empty
            if sample_dict["angle"]:
                data_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
        elif self.action_mod == 3:
            # point might be invalid
            if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                data_dict["goal"] = torch.tensor([
                    sample_dict["goal"][0] / 1600,
                    sample_dict["goal"][1] / 900
                ])
        else:
            raise ValueError
        return data_dict

    def __getitem__(self, index):
        sample_dict = self.samples[index]
        self.action_mod = (self.action_mod + index) % 4

        image_seq = list()
        for i in range(self.num_frames):
            current_index = i
            img_path = self.get_image_path(sample_dict, current_index)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
        return self.build_data_dict(image_seq, sample_dict)
