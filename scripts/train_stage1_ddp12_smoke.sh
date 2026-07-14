#!/usr/bin/env bash
set -euo pipefail

read -r -a PYTHON_CMD <<< "${PYTHON_CMD:-conda run -n swinmatcher python}"

CUDA_VISIBLE_DEVICES=1,2 "${PYTHON_CMD[@]}" train.py \
  --data_cfg_path configs/swinmatcherbit_stage1_512.py \
  --main_cfg_path configs/swinmatcher_ds.py \
  --exp_name SwinMatcherBIT_stage1_ddp12_smoke \
  --gpus 2 \
  --batch_size 1 \
  --num_workers 0 \
  --precision 16 \
  --fast_dev_run True \
  --disable_ckpt
