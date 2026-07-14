import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os
import random
import numpy as np
from loguru import logger

from src.utils.dataset import read_multi_modality_gray


def load_transform_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        matrix = np.load(path)
    else:
        matrix = np.loadtxt(path)
    return matrix.astype(np.float32)


class MultiModality(Dataset):
    def __init__(self,
                 root_dir=None,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 augment_fn=None,
                 manifest_path=None,
                 check_manifest_files=False,
                 **kwargs):
        """
        Manage one scene(npz_path) of Multi-Modality dataset.

        Args:
            root_dir (str): multi-modality root directory that has `phoenix`.
            mode (str): options are ['train', 'val', 'test'].
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.manifest_path = manifest_path
        self.check_manifest_files = check_manifest_files
        self.mode = mode
        self.dataset = []
        self.build_dataset()

        # parameters for image resizing and padding
        # if mode == 'train':
        #     assert img_resize is not None and img_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.pseudo_thermal_prob = kwargs.get('pseudo_thermal_prob', 0.0)

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = kwargs.get('coarse_scale', 0.125)

    def build_dataset(self):
        if self.manifest_path:
            self.build_dataset_from_manifest()
            return

        if self.root_dir is None:
            raise ValueError("Either root_dir or manifest_path must be provided.")

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        skipped_pairs = 0
        scenes = sorted(os.listdir(self.root_dir))
        for scene in scenes:
            scene_path = self.root_dir + f"/{scene}"
            if not os.path.isdir(scene_path):
                continue
            scene_pairs = sorted(os.listdir(scene_path))
            for scene_pair in scene_pairs:
                scene_pair_path = scene_path + f"/{scene_pair}"
                if not os.path.isdir(scene_pair_path):
                    continue
                folders = sorted(os.listdir(scene_pair_path))
                for folder in folders:
                    folder_path = scene_pair_path + f"/{folder}"
                    if not os.path.isdir(folder_path):
                        continue
                    files = sorted(os.listdir(folder_path))
                    images = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
                    if len(images) < 2:
                        continue

                    image0 = images[0]
                    candidates = images[2:] if len(images) > 2 else images[1:]
                    for image1 in candidates:  # pass the first pair
                        stem = os.path.splitext(image1)[0]
                        affine_matrix = f"{stem}.txt"
                        affine_matrix_path = f"{folder_path}/{affine_matrix}"
                        if not os.path.exists(affine_matrix_path):
                            skipped_pairs += 1
                            continue
                        self.dataset.append({"image0_path": f"{folder_path}/{image0}",
                                             "image1_path": f"{folder_path}/{image1}",
                                             "affine_matrix_path": affine_matrix_path})
        if skipped_pairs:
            logger.warning(f"Skipped {skipped_pairs} image pairs without affine matrix files.")
        np.random.seed(42)
        np.random.shuffle(self.dataset)

    def build_dataset_from_manifest(self):
        skipped = 0
        with open(self.manifest_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("mode") != "gt_pairs":
                    continue
                image0_path = record.get("image0")
                image1_path = record.get("image1")
                matrix_path = record.get("gt")
                if not image0_path or not image1_path or not matrix_path:
                    skipped += 1
                    continue
                if self.check_manifest_files and not (
                    os.path.exists(image0_path) and os.path.exists(image1_path) and os.path.exists(matrix_path)
                ):
                    skipped += 1
                    continue
                self.dataset.append({
                    "image0_path": image0_path,
                    "image1_path": image1_path,
                    "affine_matrix_path": matrix_path,
                    "gt_direction": record.get("gt_direction", "0to1"),
                    "pair_id": record.get("id"),
                    "dataset_name": record.get("dataset", "Multi-Modality"),
                    "metadata": record,
                })
        if skipped:
            logger.warning(f"Skipped {skipped} manifest rows with missing fields or files.")
        np.random.seed(42)
        np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # TODO: Support augmentation & handle seeds for each worker correctly.
        # ----- pretrain -----
        thermal_option0, thermal_option1 = False, False
        apply_gamma = self.mode == 'train'
        # ----- pseudo cross-modal enhancement -----
        if self.mode == 'train' and self.pseudo_thermal_prob > 0 and random.random() < self.pseudo_thermal_prob:
            if random.choice([0, 1]) == 0:
                thermal_option0, thermal_option1 = True, False
            else:
                thermal_option0, thermal_option1 = False, True

        # noinspection PyTypeChecker
        image0, mask0, scale0 = read_multi_modality_gray(
            item["image0_path"], self.img_resize, self.df, self.img_padding, None,
            thermal_option0, apply_gamma)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # noinspection PyTypeChecker
        image1, mask1, scale1 = read_multi_modality_gray(
            item["image1_path"], self.img_resize, self.df, self.img_padding, None,
            thermal_option1, apply_gamma)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read and compute relative poses
        # noinspection PyTypeChecker
        T_0to1 = torch.from_numpy(load_transform_matrix(item["affine_matrix_path"]))  # (3, 3)
        if item.get("gt_direction") == "1to0":
            T_0to1 = T_0to1.inverse()
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'image0_path': item["image0_path"],
            'image1_path': item["image1_path"],
            'T_0to1': T_0to1,  # (3, 3)
            'T_1to0': T_1to0,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': item.get('dataset_name', 'Multi-Modality'),
            'pair_id': item.get('pair_id', idx),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
