import os

from loguru import logger

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.augment import build_augmentor
from src.datasets.multi_modality import MultiModality


class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_manifest_path = config.DATASET.TRAIN_MANIFEST_PATH
        self.test_manifest_path = config.DATASET.TEST_MANIFEST_PATH
        if self.train_manifest_path is not None:
            self.train_manifest_path = os.path.expanduser(self.train_manifest_path)
            if not os.path.isfile(self.train_manifest_path):
                raise FileNotFoundError(f"Training manifest does not exist: {self.train_manifest_path}")
        else:
            if self.train_data_root is None:
                raise ValueError("DATASET.TRAIN_DATA_ROOT or DATASET.TRAIN_MANIFEST_PATH must be set for training.")
            self.train_data_root = os.path.expanduser(self.train_data_root)
            if not os.path.isdir(self.train_data_root):
                raise FileNotFoundError(f"Training data root does not exist: {self.train_data_root}")
        if self.test_manifest_path is not None:
            self.test_manifest_path = os.path.expanduser(self.test_manifest_path)
            if not os.path.isfile(self.test_manifest_path):
                raise FileNotFoundError(f"Validation manifest does not exist: {self.test_manifest_path}")

        # 2. dataset config
        # general options
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mtmd_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mtmd_img_pad = config.DATASET.MGDPT_IMG_PAD  # True
        self.mtmd_df = config.DATASET.MGDPT_DF  # 8
        self.pseudo_thermal_prob = config.DATASET.PSEUDO_THERMAL_PROB
        self.coarse_scale = 1 / config.SWINMATCHER.RESOLUTION[0]  # 0.125. for training swinmatcher.

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

        # 4. misc configurations
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase.
        """

        assert stage == 'fit', "stage must be fit"

        try:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed process group is not initialized.")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except (AssertionError, RuntimeError, ValueError) as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set world_size = 1 and rank = 0)")

        if stage == 'fit':
            self.train_dataset = MultiModality(self.train_data_root,
                                               mode='train',
                                               img_resize=self.mtmd_img_resize,
                                               df=self.mtmd_df,
                                               img_padding=self.mtmd_img_pad,
                                               augment_fn=self.augment_fn,
                                               manifest_path=self.train_manifest_path,
                                               pseudo_thermal_prob=self.pseudo_thermal_prob,
                                               coarse_scale=self.coarse_scale)
            logger.info(f'[rank:{self.rank}] Train Dataset loaded!')
            self.val_dataset = None
            if self.test_manifest_path is not None:
                self.val_dataset = MultiModality(self.train_data_root,
                                                 mode='val',
                                                 img_resize=self.mtmd_img_resize,
                                                 df=self.mtmd_df,
                                                 img_padding=self.mtmd_img_pad,
                                                 augment_fn=None,
                                                 manifest_path=self.test_manifest_path,
                                                 pseudo_thermal_prob=0.0,
                                                 coarse_scale=self.coarse_scale)
                logger.info(f'[rank:{self.rank}] Validation Dataset loaded!')

    def train_dataloader(self):
        """
        Build training dataloader for MultiModality.
        """
        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
            )
            shuffle = False
            logger.info(f"[rank:{self.rank}] Using DistributedSampler for {self.world_size} ranks.")

        dataloader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=shuffle,
            **self.train_loader_params,
        )
        return dataloader

    def val_dataloader(self):
        """
        Build a lightweight validation dataloader for training-time paper metrics.
        """
        if getattr(self, 'val_dataset', None) is None:
            return None

        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self.seed,
                drop_last=False,
            )

        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            shuffle=False,
            **self.train_loader_params,
        )
