#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python evaluate.py \
  --manifest_path manifests/test_SwinMatcherBIT_gt.jsonl \
  --ckpt_path weights/swinmatcher_512.ckpt \
  --output_dir outputs_eval/swinmatcherbit_paper_test \
  --device cuda \
  --half
