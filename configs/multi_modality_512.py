from src.config.default import _CN as cfg

cfg.DATASET.TRAINVAL_DATA_SOURCE = "Multi-Modality"
cfg.DATASET.TRAIN_DATA_ROOT = "/home/disk1/Data/datasets/SwinMatcher/BIT_multi-modal_datasets_homography_180_rotation"

cfg.DATASET.MGDPT_IMG_RESIZE = 512
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 8
