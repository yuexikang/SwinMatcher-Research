import torch
import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_swinmatcher import PL_SwinMatcher

loguru_logger = get_rank_zero_only_logger(loguru_logger)

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, default='configs/multi_modality_512.py',
        help='data config path')
    parser.add_argument(
        '--main_cfg_path', type=str, default='configs/swinmatcher_ds.py',
        help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='SwinMatcher')
    parser.add_argument(
        '--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='model-only pretrained checkpoint path. This starts a new optimizer/scheduler state.')
    parser.add_argument(
        '--resume_ckpt_path', type=str, default=None,
        help='full Lightning checkpoint path for continuing training with optimizer/scheduler/global_step.')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='also log training metrics to Weights & Biases.')
    parser.add_argument(
        '--wandb_project', type=str, default='SwinMatcher',
        help='Weights & Biases project name.')
    parser.add_argument(
        '--wandb_entity', type=str, default=None,
        help='Weights & Biases entity/user/team.')
    parser.add_argument(
        '--wandb_name', type=str, default=None,
        help='Weights & Biases run name. Defaults to exp_name.')
    parser.add_argument(
        '--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
        help='Weights & Biases mode.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def configure_warmup_steps(config, args, data_module):
    """Derive optimizer warm-up steps from the selected training manifest."""
    warmup_epochs = int(getattr(config.TRAINER, 'WARMUP_EPOCHS', 0))
    if warmup_epochs <= 0:
        config.TRAINER.WARMUP_STEP = 0
        return

    # Avoid DataModule.setup() here: DDP has not initialized its process group
    # yet, and Trainer.fit() must own rank/sampler setup.
    num_samples = data_module.train_dataset_size()
    samples_per_rank = math.ceil(num_samples / config.TRAINER.WORLD_SIZE)
    batches_per_epoch = math.ceil(samples_per_rank / args.batch_size)
    accumulate = getattr(args, 'accumulate_grad_batches', 1)
    if not isinstance(accumulate, int) or accumulate < 1:
        accumulate = 1
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulate)
    config.TRAINER.WARMUP_STEP = warmup_epochs * optimizer_steps_per_epoch
    loguru_logger.info(
        "Configured warm-up: {} epochs x {} optimizer steps/epoch = {} steps "
        "(samples={}, world_size={}, batch_size={}, accumulate={}).",
        warmup_epochs, optimizer_steps_per_epoch, config.TRAINER.WARMUP_STEP,
        num_samples, config.TRAINER.WORLD_SIZE, args.batch_size, accumulate)


def main():
    # parse arguments
    args = parse_args()
    if args.gpus is None:
        args.gpus = -1
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(1, _n_gpus * args.num_nodes)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    # config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    resume_ckpt_path = args.resume_ckpt_path or getattr(args, 'resume_from_checkpoint', None)
    if resume_ckpt_path and args.ckpt_path:
        raise ValueError("--resume_ckpt_path/--resume_from_checkpoint cannot be used together with --ckpt_path.")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_SwinMatcher(config, pretrained_ckpt=None if resume_ckpt_path else args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"SwinMatcher LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    configure_warmup_steps(config, args, data_module)
    loguru_logger.info(f"SwinMatcher DataModule initialized!")

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=config.TRAINER.LOG_DIR, name=args.exp_name, default_hp_metric=False)
    loggers = [tb_logger]
    if args.use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as exc:
            raise ImportError("wandb is not installed. Run: conda run -n swinmatcher python -m pip install wandb") from exc
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.exp_name,
            save_dir=config.TRAINER.LOG_DIR,
            offline=args.wandb_mode == 'offline',
            log_model=False,
        )
        wandb_config = {
            "data_cfg_path": args.data_cfg_path,
            "main_cfg_path": args.main_cfg_path,
            "batch_size_per_gpu": args.batch_size,
            "world_size": config.TRAINER.WORLD_SIZE,
            "true_batch_size": config.TRAINER.TRUE_BATCH_SIZE,
            "true_lr": config.TRAINER.TRUE_LR,
            "warmup_epochs": config.TRAINER.WARMUP_EPOCHS,
            "warmup_steps": config.TRAINER.WARMUP_STEP,
            "train_manifest_path": config.DATASET.TRAIN_MANIFEST_PATH,
            "pseudo_thermal_prob": config.DATASET.PSEUDO_THERMAL_PROB,
        }
        try:
            wandb_run_config = getattr(wandb_logger.experiment, "config", None)
            if hasattr(wandb_run_config, "update"):
                wandb_run_config.update(wandb_config, allow_val_change=True)
            else:
                wandb_logger.log_hyperparams(wandb_config)
        except Exception as exc:
            loguru_logger.warning(f"Skip W&B config update: {exc}")
        loggers.append(wandb_logger)

    ckpt_dir = Path(tb_logger.log_dir) / 'checkpoints'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(dirpath=str(ckpt_dir), filename='{epoch:02d}', save_top_k=-1, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    strategy = DDPPlugin(find_unused_parameters=True) if config.TRAINER.WORLD_SIZE > 1 else None
    trainer_max_epochs = args.max_epochs if getattr(args, 'max_epochs', None) is not None else config.TRAINER.MAX_EPOCHS
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=loggers,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 1,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_n_epochs=0,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler,
        limit_val_batches=config.TRAINER.VAL_BATCHES,
        enable_checkpointing=not args.disable_ckpt,
        max_epochs=trainer_max_epochs)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    if resume_ckpt_path:
        loguru_logger.info(f"Resume full training state from: {resume_ckpt_path}")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)


if __name__ == '__main__':
    main()
