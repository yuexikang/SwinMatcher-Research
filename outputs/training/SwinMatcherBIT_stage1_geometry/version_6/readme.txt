实验二 A-v2 已就绪。上一轮二 A 的结果不应再用于归因，因为它混入了 FP16 和 fine padding 改动。
现在的 v2：
Stage 1 脚本默认 PRECISION=32。
coarse loss 保持方向独立监督。
fine padding 恢复为原始双方向 GT union，并对重复 pair 去重。
valid mask 不再把 coarse token 0 当作无效 sentinel。
rank 采用 tie-aware 统计：gt_rank*_best_mean
gt_rank*_worst_mean
gt_rank*_avg_mean
gt_top1_*_ratio 现在是严格 top1。

新增 coarse/gt_mutual_ratio 与 coarse/gt_direction_conflict_ratio。
对 h30 optical-optical 的 100 对样本预审计结果：
gt_mutual_ratio: 0.923278
gt_direction_conflict_ratio: 0.076722
因此约 92.3% 的离散 coarse GT 是互逆的，方向独立监督有实验价值；约 7.7% 的冲突也会被明确记录。
从零开始跑，明确指定 FP32：
USE_WANDB=1 WANDB_PROJECT=SwinMatcher \
WANDB_NAME='实验2A-v2_方向独立GT_FP32_fineUnion' \
CUDA_DEVICES=1 GPUS=1 BATCH_SIZE=2 PRECISION=32 \
bash scripts/train_stage1.sh