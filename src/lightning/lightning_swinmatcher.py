from loguru import logger
from pathlib import Path
import cv2
import numpy as np
import torch
import pytorch_lightning as pl

from src.swinmatcher import SwinMatcher
from src.swinmatcher.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.swinmatcher_loss import SwinMatcherLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


def _tensor_image_to_bgr(image):
    image = image.detach().float().cpu()
    if image.ndim == 3:
        image = image[0]
    image = image.clamp(0, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _feature_to_heatmap(feature, out_hw):
    feature = feature.detach().float().cpu()
    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    heat = torch.linalg.vector_norm(feature, ord=2, dim=0)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp(min=1e-6)
    heat = (heat.numpy() * 255).astype(np.uint8)
    heat = cv2.resize(heat, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)


def _short_path(path, max_chars=86):
    if isinstance(path, (list, tuple)):
        path = path[0]
    path = str(path)
    if len(path) <= max_chars:
        return path
    parts = Path(path).parts
    tail = str(Path(*parts[-4:])) if len(parts) >= 4 else path[-max_chars:]
    text = f".../{tail}"
    return text if len(text) <= max_chars else "..." + text[-max_chars + 3:]


def _put_label(image, text):
    out = image.copy()
    lines = text if isinstance(text, (list, tuple)) else [text]
    height = 8 + 22 * len(lines)
    cv2.rectangle(out, (0, 0), (out.shape[1], height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(out, str(line), (8, 21 + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _loss_scalars_to_text(loss_scalars):
    items = []
    for key, value in loss_scalars.items():
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.flatten()[:4].tolist()
        items.append(f"{key}={value}")
    return ", ".join(items)


@torch.no_grad()
def _coarse_diagnostics(batch, coarse_thr=0.3):
    diagnostics = {}

    # Counts are written before GT padding and before fine matching overwrites
    # ``m_bids``.  They are therefore genuine coarse/fine control-flow signals.
    for key in (
        'coarse_raw_pred_match_count',
        'coarse_gt_padded_for_fine_count',
        'coarse_fine_input_match_count',
    ):
        value = batch.get(key)
        if torch.is_tensor(value):
            diagnostics[f'coarse/{key.removeprefix("coarse_")}'] = value.detach()

    mconf = batch.get('mconf')
    if torch.is_tensor(mconf):
        mconf = mconf.detach()
        diagnostics.update({
            'coarse/pred_mconf_mean': mconf.mean() if mconf.numel() else mconf.new_tensor(0.0),
            'coarse/pred_mconf_max': mconf.max() if mconf.numel() else mconf.new_tensor(0.0),
        })

    if 'm_bids' in batch and torch.is_tensor(batch['m_bids']):
        diagnostics['fine/output_matches'] = batch['m_bids'].detach().numel()
    if 'fine_fallback_used' in batch and torch.is_tensor(batch['fine_fallback_used']):
        diagnostics['fine/fallback_used'] = batch['fine_fallback_used'].detach().float()

    if 'spv_b_ids' in batch and torch.is_tensor(batch['spv_b_ids']):
        diagnostics['coarse/supervision_matches'] = batch['spv_b_ids'].detach().numel()

    conf0 = batch.get('conf_matrix_0_to_1')
    conf1 = batch.get('conf_matrix_1_to_0')
    if torch.is_tensor(conf0) and torch.is_tensor(conf1):
        conf0 = conf0.detach()
        conf1 = conf1.detach()
        diagnostics.update({
            'coarse/conf0_max': conf0.max(),
            'coarse/conf1_max': conf1.max(),
            'coarse/conf0_over_thr_ratio': (conf0 > coarse_thr).float().mean(),
            'coarse/conf1_over_thr_ratio': (conf1 > coarse_thr).float().mean(),
        })

        spv_b = batch.get('spv_b_ids')
        spv_i = batch.get('spv_i_ids')
        spv_j = batch.get('spv_j_ids')
        if all(torch.is_tensor(x) and x.numel() for x in (spv_b, spv_i, spv_j)):
            spv_b, spv_i, spv_j = (x.detach().long() for x in (spv_b, spv_i, spv_j))
            gt0 = conf0[spv_b, spv_i, spv_j]
            gt1 = conf1[spv_b, spv_i, spv_j]

            # Exact rank, chunked so the diagnostic has a bounded memory cost.
            ranks0, ranks1 = [], []
            for start in range(0, len(spv_b), 256):
                stop = start + 256
                b, i, j = spv_b[start:stop], spv_i[start:stop], spv_j[start:stop]
                scores0, scores1 = gt0[start:stop], gt1[start:stop]
                ranks0.append((conf0[b, i, :] > scores0[:, None]).sum(dim=1) + 1)
                ranks1.append((conf1[b, :, j] > scores1[:, None]).sum(dim=1) + 1)
            ranks0, ranks1 = torch.cat(ranks0), torch.cat(ranks1)

            diagnostics.update({
                'coarse/gt_conf0_mean': gt0.mean(),
                'coarse/gt_conf1_mean': gt1.mean(),
                'coarse/gt_conf0_median': gt0.median(),
                'coarse/gt_conf1_median': gt1.median(),
                'coarse/gt_conf0_over_thr_ratio': (gt0 > coarse_thr).float().mean(),
                'coarse/gt_conf1_over_thr_ratio': (gt1 > coarse_thr).float().mean(),
                'coarse/gt_rank0_mean': ranks0.float().mean(),
                'coarse/gt_rank1_mean': ranks1.float().mean(),
                'coarse/gt_rank0_median': ranks0.float().median(),
                'coarse/gt_rank1_median': ranks1.float().median(),
                'coarse/gt_top1_0_ratio': (ranks0 == 1).float().mean(),
                'coarse/gt_top1_1_ratio': (ranks1 == 1).float().mean(),
            })

    return diagnostics


def _make_feature_panel(batch, sample_idx=0):
    image0 = _tensor_image_to_bgr(batch['image0'][sample_idx])
    image1 = _tensor_image_to_bgr(batch['image1'][sample_idx])
    h, w = image0.shape[:2]
    image0_label = ['image0', _short_path(batch.get('image0_path', ''))]
    image1_label = ['image1', _short_path(batch.get('image1_path', ''))]

    panels0 = [
        _put_label(image0, image0_label),
        _put_label(_feature_to_heatmap(batch['feat_f0_vis'][sample_idx], (h, w)), 'image0 1/2 feature'),
        _put_label(_feature_to_heatmap(batch['feat_c0_vis'][sample_idx], (h, w)), 'image0 1/8 feature'),
    ]
    panels1 = [
        _put_label(image1, image1_label),
        _put_label(_feature_to_heatmap(batch['feat_f1_vis'][sample_idx], (h, w)), 'image1 1/2 feature'),
        _put_label(_feature_to_heatmap(batch['feat_c1_vis'][sample_idx], (h, w)), 'image1 1/8 feature'),
    ]
    top = cv2.hconcat(panels0)
    bottom = cv2.hconcat(panels1)
    return cv2.vconcat([top, bottom])


def _scale_matches_to_tensor_coords(points, scale):
    if len(points) == 0:
        return points
    scale = scale.detach().float().cpu().numpy()
    points = points.copy()
    points[:, 0] /= max(float(scale[0]), 1e-6)
    points[:, 1] /= max(float(scale[1]), 1e-6)
    return points


def _compute_match_metrics(batch, mkpts0, mkpts1, sample_idx=0, correct_thr=5.0, success_ncm=20, failed_rmse=10.0):
    if len(mkpts0) == 0:
        return 0, 0.0, failed_rmse

    scale0 = batch['scale0'][sample_idx].detach().float().cpu().numpy()
    scale1 = batch['scale1'][sample_idx].detach().float().cpu().numpy()
    mkpts0_orig = mkpts0.copy()
    mkpts1_orig = mkpts1.copy()
    mkpts0_orig[:, 0] *= float(scale0[0])
    mkpts0_orig[:, 1] *= float(scale0[1])
    mkpts1_orig[:, 0] *= float(scale1[0])
    mkpts1_orig[:, 1] *= float(scale1[1])

    matrix = batch['T_0to1'][sample_idx].detach().float().cpu().numpy()
    pts_h = np.concatenate([mkpts0_orig, np.ones((len(mkpts0_orig), 1), dtype=np.float32)], axis=1)
    warped_h = pts_h @ matrix.T
    valid = np.abs(warped_h[:, 2]) > 1e-8
    warped = np.full((len(mkpts0_orig), 2), np.nan, dtype=np.float32)
    warped[valid] = warped_h[valid, :2] / warped_h[valid, 2:3]

    errors = np.linalg.norm(warped - mkpts1_orig, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    correct = errors <= correct_thr
    ncm = int(correct.sum())
    pre = float(ncm / len(errors)) if len(errors) else 0.0
    rmse = float(np.sqrt(np.mean(errors[correct] ** 2))) if ncm >= success_ncm else failed_rmse
    return ncm, pre, rmse


def _sample_matches(batch, sample_idx=0):
    if 'mkpts0_f' not in batch or 'mkpts1_f' not in batch:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    mkpts0 = batch['mkpts0_f'].detach().float().cpu().numpy()
    mkpts1 = batch['mkpts1_f'].detach().float().cpu().numpy()
    if 'm_bids' in batch:
        mask = batch['m_bids'].detach().cpu().numpy() == sample_idx
        mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]

    mkpts0 = _scale_matches_to_tensor_coords(mkpts0, batch['scale0'][sample_idx])
    mkpts1 = _scale_matches_to_tensor_coords(mkpts1, batch['scale1'][sample_idx])
    return mkpts0, mkpts1


def _compute_paper_metric_row(batch, mkpts0, mkpts1, sample_idx=0, correct_thr=5.0, success_ncm=20, failed_rmse=10.0):
    num_matches = len(mkpts0)
    if num_matches == 0:
        return {
            'pairs': 1.0,
            'ncm': 0.0,
            'pre': 0.0,
            'sr': 0.0,
            'rmse': failed_rmse,
            'matches': 0.0,
        }

    scale0 = batch['scale0'][sample_idx].detach().float().cpu().numpy()
    scale1 = batch['scale1'][sample_idx].detach().float().cpu().numpy()
    mkpts0_orig = mkpts0.copy()
    mkpts1_orig = mkpts1.copy()
    mkpts0_orig[:, 0] *= float(scale0[0])
    mkpts0_orig[:, 1] *= float(scale0[1])
    mkpts1_orig[:, 0] *= float(scale1[0])
    mkpts1_orig[:, 1] *= float(scale1[1])

    matrix = batch['T_0to1'][sample_idx].detach().float().cpu().numpy()
    pts_h = np.concatenate([mkpts0_orig, np.ones((len(mkpts0_orig), 1), dtype=np.float32)], axis=1)
    warped_h = pts_h @ matrix.T
    valid = np.abs(warped_h[:, 2]) > 1e-8
    warped = np.full((len(mkpts0_orig), 2), np.nan, dtype=np.float32)
    warped[valid] = warped_h[valid, :2] / warped_h[valid, 2:3]

    errors = np.linalg.norm(warped - mkpts1_orig, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    correct = errors <= correct_thr
    ncm = float(correct.sum())
    success = ncm >= success_ncm
    rmse = float(np.sqrt(np.mean(errors[correct] ** 2))) if success else failed_rmse
    return {
        'pairs': 1.0,
        'ncm': ncm,
        'pre': float(ncm / num_matches),
        'sr': 1.0 if success else 0.0,
        'rmse': rmse,
        'matches': float(num_matches),
    }


def _make_matches_panel(batch, sample_idx=0, max_matches=200, mode_label=None):
    image0 = _tensor_image_to_bgr(batch['image0'][sample_idx])
    image1 = _tensor_image_to_bgr(batch['image1'][sample_idx])
    image0 = _put_label(image0, ['image0', _short_path(batch.get('image0_path', ''))])
    image1 = _put_label(image1, ['image1', _short_path(batch.get('image1_path', ''))])
    h = max(image0.shape[0], image1.shape[0])
    if image0.shape[0] < h:
        image0 = cv2.copyMakeBorder(image0, 0, h - image0.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if image1.shape[0] < h:
        image1 = cv2.copyMakeBorder(image1, 0, h - image1.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    canvas = cv2.hconcat([image0, image1])
    x_offset = image0.shape[1]

    mkpts0, mkpts1 = _sample_matches(batch, sample_idx=sample_idx)
    ncm, pre, rmse = _compute_match_metrics(batch, mkpts0, mkpts1, sample_idx=sample_idx)

    if len(mkpts0) > max_matches:
        indices = np.linspace(0, len(mkpts0) - 1, max_matches).astype(np.int64)
        mkpts0, mkpts1 = mkpts0[indices], mkpts1[indices]

    rng = np.random.default_rng(0)
    for p0, p1 in zip(mkpts0, mkpts1):
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        pt0 = tuple(np.round(p0).astype(int))
        pt1 = tuple(np.round([p1[0] + x_offset, p1[1]]).astype(int))
        cv2.circle(canvas, pt0, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, pt0, pt1, color, 1, lineType=cv2.LINE_AA)

    label = []
    if mode_label:
        label.append(mode_label)
    label.extend([
        f'final matches: {len(mkpts0)} shown',
        f'NCM={ncm}  Pre={pre:.3f}  RMSE={rmse:.3f}px'
    ])
    return _put_label(canvas, label)


class PL_SwinMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.swinmatcher_cfg = lower_config(_config['swinmatcher'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher
        self.matcher = SwinMatcher(config=_config['swinmatcher'])
        self.loss = SwinMatcherLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            if any(key.startswith('matcher.') for key in state_dict):
                state_dict = {
                    key[len('matcher.'):]: value
                    for key, value in state_dict.items()
                    if key.startswith('matcher.')
                }
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("SwinMatcher"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self._log_coarse_diagnostics(batch)
        if not torch.isfinite(batch['loss']):
            scalars = _loss_scalars_to_text(batch.get('loss_scalars', {}))
            raise FloatingPointError(
                f"Non-finite training loss at epoch={self.current_epoch}, "
                f"global_step={self.global_step}, batch_idx={batch_idx}. "
                f"Loss scalars: {scalars}"
            )
        self._save_training_visuals(batch)
        
        # logging
        for key, value in batch['loss_scalars'].items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        return {'loss': batch['loss'],
                'loss_c': batch['loss_c'],
                'loss_f': batch['loss_f'],
                'loss_sub': batch['loss_sub']}

    def _log_coarse_diagnostics(self, batch):
        log_every = max(int(getattr(self.trainer, 'log_every_n_steps', 50)), 1)
        if self.global_step % log_every != 0:
            return

        coarse_thr = float(self.swinmatcher_cfg['match_coarse']['thr'])
        diagnostics = _coarse_diagnostics(batch, coarse_thr=coarse_thr)
        for key, value in diagnostics.items():
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=self.device)
            else:
                value = value.to(self.device).float()
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        if self.trainer.global_rank == 0:
            def _value(name):
                value = diagnostics.get(name, torch.tensor(0.0))
                return float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)

            logger.info(
                f"Coarse diagnostic step={self.global_step} "
                f"raw_pred={int(_value('coarse/raw_pred_match_count'))} "
                f"gt_pad={int(_value('coarse/gt_padded_for_fine_count'))} "
                f"fine_input={int(_value('coarse/fine_input_match_count'))} "
                f"gt_conf=({_value('coarse/gt_conf0_mean'):.6f}, "
                f"{_value('coarse/gt_conf1_mean'):.6f}) "
                f"gt_rank=({_value('coarse/gt_rank0_mean'):.1f}, "
                f"{_value('coarse/gt_rank1_mean'):.1f}) "
                f"top1=({_value('coarse/gt_top1_0_ratio'):.3f}, "
                f"{_value('coarse/gt_top1_1_ratio'):.3f}) "
                f"coarse_max=({_value('coarse/conf0_max'):.4f}, "
                f"{_value('coarse/conf1_max'):.4f}) "
                f"fine_fallback={int(_value('fine/fallback_used'))}"
            )

    @torch.no_grad()
    def _save_training_visuals(self, batch):
        vis_interval = int(getattr(self.config.TRAINER, 'VIS_INTERVAL', 0))
        if vis_interval <= 0 or self.trainer.global_rank != 0:
            return
        if self.global_step % vis_interval != 0:
            return

        log_dir = None
        loggers = []
        if getattr(self, 'trainer', None) is not None:
            loggers = list(getattr(self.trainer, 'loggers', []) or [])
        loggers = loggers or list(getattr(self, 'loggers', []) or [])

        for exp_logger in loggers:
            log_dir = getattr(exp_logger, 'log_dir', None)
            if log_dir is not None:
                break

        for exp_logger in loggers:
            if log_dir is not None:
                break
            save_dir = getattr(exp_logger, 'save_dir', None)
            name = getattr(exp_logger, 'name', None)
            version = getattr(exp_logger, 'version', None)
            if save_dir is not None and name is not None:
                log_dir = Path(save_dir) / str(name)
                if version is not None:
                    log_dir = log_dir / str(version)
                break

        if log_dir is None:
            log_dir = Path(self.config.TRAINER.LOG_DIR)
        vis_dir = Path(log_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        max_matches = int(getattr(self.config.TRAINER, 'VIS_MAX_MATCHES', 200))
        prediction_batch = self._eval_pure_prediction_batch(batch)
        feature_panel = _make_feature_panel(prediction_batch, sample_idx=0)
        matches_panel = _make_matches_panel(
            prediction_batch,
            sample_idx=0,
            max_matches=max_matches,
            mode_label='eval_pure_prediction',
        )
        cv2.imwrite(str(vis_dir / 'latest_features.jpg'), feature_panel)
        cv2.imwrite(str(vis_dir / 'latest_matches.jpg'), matches_panel)
        self._log_wandb_visuals(loggers, feature_panel, matches_panel)

    @torch.no_grad()
    def _eval_pure_prediction_batch(self, batch):
        keys = [
            'image0', 'image1', 'mask0', 'mask1',
            'scale0', 'scale1', 'T_0to1', 'T_1to0',
            'image0_path', 'image1_path', 'dataset_name', 'pair_id',
        ]
        pred_batch = {}
        for key in keys:
            if key not in batch:
                continue
            value = batch[key]
            pred_batch[key] = value.detach() if torch.is_tensor(value) else value

        was_training = self.matcher.training
        self.matcher.eval()
        try:
            self.matcher(pred_batch)
        finally:
            self.matcher.train(was_training)
        return pred_batch

    def _log_wandb_visuals(self, loggers, feature_panel, matches_panel):
        try:
            import wandb
        except ImportError:
            return

        for exp_logger in loggers:
            if exp_logger.__class__.__name__ != 'WandbLogger':
                continue
            try:
                exp_logger.experiment.log(
                    {
                        'visualizations/latest_features': wandb.Image(
                            cv2.cvtColor(feature_panel, cv2.COLOR_BGR2RGB),
                            caption='eval_pure_prediction/latest_features',
                        ),
                        'visualizations/latest_matches': wandb.Image(
                            cv2.cvtColor(matches_panel, cv2.COLOR_BGR2RGB),
                            caption='eval_pure_prediction/latest_matches',
                        ),
                    },
                    step=int(self.global_step),
                )
            except Exception as exc:
                logger.warning(f"Skip W&B visual logging: {exc}")
            break

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_c = torch.stack([x['loss_c'] for x in outputs]).mean()
        avg_loss_f = torch.stack([x['loss_f'] for x in outputs]).mean()
        avg_loss_sub = torch.stack([x['loss_sub'] for x in outputs]).mean()

        self.log('train/epoch_avg_loss', avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/epoch_avg_loss_c', avg_loss_c, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/epoch_avg_loss_f', avg_loss_f, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/epoch_avg_loss_sub', avg_loss_sub, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if self.trainer.global_rank == 0:
            print(f"Epoch {self.current_epoch}: the average loss is "
                  f"{avg_loss_c.item():.4f} + {avg_loss_f.item():.4f} + {avg_loss_sub.item():.4f} = {avg_loss.item():.4f}")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.matcher(batch)
        metrics = {
            'pairs': 0.0,
            'ncm': 0.0,
            'pre': 0.0,
            'sr': 0.0,
            'rmse': 0.0,
            'matches': 0.0,
        }
        batch_size = int(batch['image0'].shape[0])
        for sample_idx in range(batch_size):
            mkpts0, mkpts1 = _sample_matches(batch, sample_idx=sample_idx)
            row = _compute_paper_metric_row(batch, mkpts0, mkpts1, sample_idx=sample_idx)
            for key, value in row.items():
                metrics[key] += value

        return {
            key: torch.tensor(value, dtype=torch.float32, device=self.device)
            for key, value in metrics.items()
        }

    def validation_epoch_end(self, outputs):
        if not outputs:
            return

        totals = {
            key: torch.stack([out[key] for out in outputs]).sum()
            for key in outputs[0]
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in totals.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        num_pairs = totals['pairs'].clamp(min=1.0)
        metrics = {
            'val/NCM': totals['ncm'] / num_pairs,
            'val/Pre': totals['pre'] / num_pairs,
            'val/SR': totals['sr'] / num_pairs,
            'val/RMSE': totals['rmse'] / num_pairs,
            'val/mean_matches': totals['matches'] / num_pairs,
        }
        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        if self.trainer.global_rank == 0:
            print(
                f"Validation epoch {self.current_epoch}: "
                f"NCM={metrics['val/NCM'].item():.3f}, "
                f"Pre={metrics['val/Pre'].item():.4f}, "
                f"SR={metrics['val/SR'].item():.4f}, "
                f"RMSE={metrics['val/RMSE'].item():.4f}, "
                f"matches={metrics['val/mean_matches'].item():.1f}"
            )
