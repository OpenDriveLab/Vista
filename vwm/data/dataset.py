import random
from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .subsets import YouTubeDataset, NuScenesDataset

try:
    from sdata import create_dataset, create_dummy_dataset, create_loader
except ImportError as e:
    print("#" * 100)
    print("Datasets not yet available")
    print("To enable, we need to add stable-datasets as a submodule")
    print("Please use ``git submodule update --init --recursive``")
    print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    print("#" * 100)
    exit(1)


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
            self,
            train: DictConfig,
            validation: Optional[DictConfig] = None,
            test: Optional[DictConfig] = None,
            skip_val_loader: bool = False,
            dummy: bool = False
    ):
        super().__init__()
        self.train_config = train
        assert (
                "datapipeline" in self.train_config and "loader" in self.train_config
        ), "Train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                        "datapipeline" in self.val_config and "loader" in self.val_config
                ), "Validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "WARNING: no validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                    "datapipeline" in self.test_config and "loader" in self.test_config
            ), "Test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("Using dummy dataset, hope you are debugging")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_data_pipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_data_pipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_data_pipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        return create_loader(self.train_data_pipeline, **self.train_config.loader)

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_data_pipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_data_pipeline, **self.test_config.loader)


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
