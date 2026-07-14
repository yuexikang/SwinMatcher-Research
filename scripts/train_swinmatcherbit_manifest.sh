#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data_cfg_path configs/swinmatcherbit_manifest_512.py \
  --main_cfg_path configs/swinmatcher_ds.py \
  --exp_name SwinMatcherBIT_manifest \
  --gpus 1 \
  --batch_size 1 \
  --num_workers 4
