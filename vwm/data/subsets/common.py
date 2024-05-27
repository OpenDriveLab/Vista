import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, data_root, anno_file, target_height=320, target_width=576, num_frames=25):
        self.data_root = data_root

        assert target_height % 64 == 0 and target_width % 64 == 0, "Resize to integer multiple of 64"
        self.img_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

        if isinstance(anno_file, list):
            self.samples = list()
            for each_file in anno_file:
                with open(each_file, "r") as anno_json:
                    self.samples += json.load(anno_json)
        else:
            with open(anno_file, "r") as anno_json:
                self.samples = json.load(anno_json)

        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

        # self.log_cond_aug_dist = torch.distributions.Normal(-3.0, 0.5)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.img_preprocessor(image)
        return image

    def get_image_path(self, sample_dict, current_index):
        pass

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
        return data_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dict = self.samples[index]

        image_seq = list()
        for i in range(self.num_frames):
            current_index = i
            img_path = self.get_image_path(sample_dict, current_index)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
        return self.build_data_dict(image_seq, sample_dict)
