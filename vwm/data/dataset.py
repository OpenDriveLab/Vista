import random

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .subsets import YouTubeDataset, NuScenesDataset


def dataset_mapping(subset_list: list, target_height: int, target_width: int, num_frames: int):
    datasets = list()
    for subset_name in subset_list:
        if subset_name == "YouTube":
            datasets.append(
                YouTubeDataset(target_height=target_height, target_width=target_width, num_frames=num_frames)
            )
        elif subset_name == "NuScenes":
            datasets.append(
                NuScenesDataset(target_height=target_height, target_width=target_width, num_frames=num_frames)
            )
        else:
            raise NotImplementedError(f"Please define {subset_name} as a subset")
    return datasets


class MultiSourceSamplerDataset(Dataset):
    def __init__(self, subsets, probs, samples_per_epoch=1000, target_height=320, target_width=576, num_frames=25):
        self.subsets = dataset_mapping(subsets, target_height, target_width, num_frames)
        # if probabilities not provided, sample uniformly from all samples
        if probs is None:
            probs = [len(d) for d in self.subsets]
        # normalize
        total_prob = sum(probs)
        self.sample_probs = [x / total_prob for x in probs]
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        """
        Args:
            index (int): Index (ignored since we sample randomly).

        Returns:
            TensorDict: Dict containing all the data blocks.
        """

        # randomly select a subset based on weights
        subset = random.choices(self.subsets, self.sample_probs)[0]

        # sample a valid sample with a random index
        while True:
            try:
                sample_item = random.choice(subset)
                # return the sampled item
                return sample_item
            except:
                pass


class Sampler(LightningDataModule):
    def __init__(self, batch_size, num_workers=0, prefetch_factor=2, shuffle=True, subsets=None, probs=None,
                 samples_per_epoch=None, target_height=320, target_width=576, num_frames=25):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle
        self.train_dataset = MultiSourceSamplerDataset(
            subsets=subsets, probs=probs, samples_per_epoch=samples_per_epoch,
            target_height=target_height, target_width=target_width, num_frames=num_frames
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,  # we disable online testing to improve training efficiency
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
