from src.config.default import _CN as cfg

cfg.DATASET.TRAINVAL_DATA_SOURCE = "Multi-Modality"
cfg.DATASET.TRAIN_DATA_ROOT = None
cfg.DATASET.TRAIN_MANIFEST_PATH = "manifests/train_SwinMatcherBIT_gt.jsonl"
cfg.DATASET.TEST_MANIFEST_PATH = "manifests/test_SwinMatcherBIT_gt.jsonl"

cfg.DATASET.MGDPT_IMG_RESIZE = 512
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 8
