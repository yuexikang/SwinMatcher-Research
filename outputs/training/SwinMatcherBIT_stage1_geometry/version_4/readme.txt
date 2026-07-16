关闭几何扰动

实验一有效，而且效果非常明显。

关闭 Stage 1 的随机 Gamma 后，粗匹配已经从“基本学不出来”变成了“训练集上具有较强的候选排序能力”。仓库提交也确认，这次只增加了 APPLY_GAMMA 开关，并在 Stage 1 设置为 False，其余训练主体没有变化，因此这个对照是成立的。

但模型现在暴露出了下一层问题：

训练阶段显著收敛，验证阶段开始有输出，但错误匹配仍然很多，泛化能力较弱。

一、关闭随机扰动前后的定量对比
指标	开启随机 Gamma	关闭随机 Gamma	变化
GT confidence 0→1 均值	0.00577	0.15443	提高约 26.8倍
GT confidence 1→0 均值	0.00558	0.14325	提高约 25.7倍
GT confidence 中位数	0.00283	0.08828	提高约 31倍
GT rank 0→1 均值	234.95	13.31	下降约 94.3%
GT rank 1→0 均值	277.40	14.15	下降约 94.9%
GT rank 中位数	84 / 89	3 / 3	接近正确候选前3
GT top1 比例	3.68% / 3.10%	32.43% / 29.95%	提高约 9倍
coarse 最大置信度	0.170 / 0.165	0.943 / 0.954	已能产生高置信预测
raw coarse matches	0	2525	粗匹配链路已打通
Fine 输出数量	184	1895	提高约 10.3倍
有效子像素匹配	373	1586	提高约 4.25倍
当前 batch 总 loss	2.237	1.244	下降约 44.4%

关闭 Gamma 前的完整摘要在旧 run 中，关闭后的结果在新 run 中。

所以之前的主要问题已经可以确定：

Stage 1 两幅图各自独立进行强随机 Gamma，严重破坏了基础同模态几何匹配学习。

它不是小影响，而是粗匹配能否形成的决定性因素之一。

二、现在 coarse→fine 链路已经真正接通

以前：

raw_pred_match_count         = 0
gt_padded_for_fine_count     = 2457
fine_input_match_count       = 2457

说明 Fine 完全依赖 GT coarse 窗口。

现在：

raw_pred_match_count         = 2525
gt_padded_for_fine_count     = 200
fine_input_match_count       = 2457

这里的 200 正好接近代码设置的最小 GT padding 数量。也就是说，现在 Fine 输入已经主要来自模型自己的 coarse 预测，只保留最低限度的 GT 补充。

因此这次不再是：

GT coarse
→ Fine

而是真正开始形成：

模型 coarse prediction
→ Fine refinement
→ Subpixel refinement

这是这次实验最重要的进展。