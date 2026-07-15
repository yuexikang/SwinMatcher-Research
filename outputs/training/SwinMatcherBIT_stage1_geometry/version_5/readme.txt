实验二 A 已实现，且只改了 coarse GT 构造与其 loss/诊断。
移除双方向 conf_matrix_gt 并集。
新增 target_0to1 / valid_0to1，只监督 P(0→1) 的唯一目标。
新增 target_1to0 / valid_1to0，只监督 P(1→0) 的唯一目标。
fine GT padding 仅使用明确的 0→1 对应，不再使用双向 union。
gt_rank0/1、gt_top1_0/1 改为按各自方向的真实标签计算。
保留 Focal loss、数据、网络、Gamma 关闭和 h30 optical-optical 不变。
合成单应矩阵测试通过：没有再创建 conf_matrix_gt，两个方向各自都能正常反向传播；rank 诊断也可正常输出。
从零开始跑实验二 A，不要加载旧 checkpoint：
USE_WANDB=1 WANDB_PROJECT=SwinMatcher \
WANDB_NAME='实验2A_分离双向GT监督_h30同模态' \
CUDA_DEVICES=1 GPUS=1 BATCH_SIZE=2 PRECISION=16 \
bash scripts/train_stage1.sh



毫无疑问的失败