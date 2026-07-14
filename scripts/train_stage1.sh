#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES="${CUDA_DEVICES:-0}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PRECISION="${PRECISION:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-}"
CKPT_PATH="${CKPT_PATH:-}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
if [[ -n "${PYTHON_CMD:-}" ]]; then
  read -r -a PYTHON_CMD <<< "$PYTHON_CMD"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "swinmatcher" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(conda run --no-capture-output -n swinmatcher python)
fi
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-SwinMatcher}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME="${WANDB_NAME:-SwinMatcherBIT_stage1_geometry}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

cmd=(
  "${PYTHON_CMD[@]}" train.py
  --data_cfg_path configs/swinmatcherbit_stage1_512.py
  --main_cfg_path configs/swinmatcher_ds.py
  --exp_name SwinMatcherBIT_stage1_geometry
  --gpus "$GPUS"
  --batch_size "$BATCH_SIZE"
  --num_workers 4
  --precision "$PRECISION"
)

if [[ -n "$ACCUMULATE_GRAD_BATCHES" ]]; then
  cmd+=(--accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES")
fi
if [[ -n "$CKPT_PATH" ]]; then
  cmd+=(--ckpt_path "$CKPT_PATH")
fi
if [[ -n "$RESUME_CKPT_PATH" ]]; then
  cmd+=(--resume_ckpt_path "$RESUME_CKPT_PATH")
fi
if [[ -n "$MAX_EPOCHS" ]]; then
  cmd+=(--max_epochs "$MAX_EPOCHS")
fi
if [[ "$USE_WANDB" == "1" ]]; then
  cmd+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_name "$WANDB_NAME")
  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "GPUS=$GPUS BATCH_SIZE_PER_GPU=$BATCH_SIZE TOTAL_BATCH=$((GPUS * BATCH_SIZE)) PRECISION=$PRECISION"
if [[ -n "$ACCUMULATE_GRAD_BATCHES" ]]; then
  echo "ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES EFFECTIVE_BATCH=$((GPUS * BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"
fi
if [[ -n "$MAX_EPOCHS" ]]; then
  echo "MAX_EPOCHS=$MAX_EPOCHS"
fi
if [[ -n "$CKPT_PATH" ]]; then
  echo "CKPT_PATH=$CKPT_PATH (model-only initialization)"
fi
echo "WANDB_MODE=$WANDB_MODE WANDB_START_METHOD=$WANDB_START_METHOD USE_WANDB=$USE_WANDB"
printf 'Command: CUDA_VISIBLE_DEVICES=%q' "$CUDA_DEVICES"
printf ' %q' "${cmd[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${cmd[@]}"
