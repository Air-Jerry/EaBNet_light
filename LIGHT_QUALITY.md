# PESQ 优先的增强模型

目标是在当前同一批 600 条语音、原 PESQ 计算方式下达到约 3.4。首个十轮实验的最好结果为新增强器第 10 轮的 2.8211451，原第 40 轮为 2.7214096；目标尚未达到。

## 模型与训练

新标识为 `cbam_refiner_v1`。从 `checkpoints_cbam_flat_projection64/model_epoch_40.pt` 严格加载原模型，冻结其全部参数和 BN 状态，始终以 FP32 推理。后接可训练的复数谱残差增强器：

- 输入为 8 路混合语音的压缩实虚谱和原模型的两路实虚谱，共 18 通道。
- 四级频率 U-Net，通道数 32、48、64、96；瓶颈包含两层双向 GRU，每方向 128 单元。
- 输出两路复数谱残差，乘以当前时频点的八通道压缩幅度均值，使修正量随输入音量调整，全静音时修正量为零。最后一层零初始化；第 0 轮逐点等于原第 40 轮。
- 冻结参数 735,070，可训练参数 1,890,018，总参数 2,625,088。新增强器使用未来上下文，整体为离线、非因果模型。checkpoint 中原 `is_causal` 描述冻结基线，`quality_refiner_causal=False` 和候选信息描述新增强器。

保持 16 kHz、FFT 512、hop 160、win 320、power 0.5、前 8 通道及原始幅度。变长样本在真实末端独立计算 STFT、模型和损失，避免批次补零进入双向上下文。推理只使用 mixture，不使用干净参考。

损失为归一化的压缩幅度/实虚谱 L1，加 0.1 倍多分辨率 STFT 损失、0.1 倍波形 L1。多分辨率 FFT 为 256、512、1024；重建与损失均使用 FP32，采用原 STFT 的精确逆压缩及真实采样长度。目标幅度仅用于监督项加权，不对输入或推理输出做参考增益匹配。每轮日志记录各损失分量。

这些设计参考了 [TF-GridNet 的后滤波方案（II-B、VI-B4）](https://arxiv.org/html/2211.12433v2) 和 [DEMUCS 的波形 L1 与多分辨率谱损失（2.3）](https://arxiv.org/html/2006.12847v1)。当前模型是工程改进方案，不是上述论文的完整实现，也不以其他数据集的分数证明本数据能达到 3.4。

## 服务器启动

```bash
cd ~/EaBNet_light
git pull --ff-only origin codex/9.16
mkdir -p ./logs_cbam_refiner_v1
CUDA_VISIBLE_DEVICES=0,1 nohup bash run_light_quality.sh \
  > ./logs_cbam_refiner_v1/launch.log 2>&1 &
```

```bash
tail -f ./logs_cbam_refiner_v1/launch.log
```

脚本先检查数据和当前代码版本，确认训练与评估对验证 CSV 解析到相同的音频文件，初始化新模型，再对完整 600 条评估第 0 轮。它将全部指标与已提交的原第 40 轮报告比较，容差为 0.001；不一致则停止。通过后双卡训练，初始学习率 0.0003，每卡 batch 1、累积 4 次，等效全局 batch 8。训练使用 FP32，关闭 AMP 和 TF32；评估沿用原脚本的 FP32 设置及后端默认值，以保持历史基线比较。本次服务器环境的评估默认值为 matmul TF32=False、cuDNN TF32=True。基线全程冻结，Adam 从空状态开始。

每训练一轮即保存并评估该轮 PESQ，共最多 10 轮。轮与轮之间恢复完整模型、Adam 和学习率调度状态；每轮启动新进程会重新设种子，因此不是单个连续进程的逐位复现。按完整 600 条 PESQ 比较，不能直接以训练损失选择最终模型。

每个实验创建独立的 `checkpoints_cbam_refiner_v1_quality_*`、`bestmodels_cbam_refiner_v1_quality_*` 和 `logs_cbam_refiner_v1/quality_*` 目录。原模型、原权重和原训练入口保留。新 checkpoint 包含完整基线与增强器，可独立加载评估。

已完成的实验可恢复完整模型、Adam 和学习率状态，继续到指定的累计轮数。例如从第 10 轮继续到第 20 轮：

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup bash continue_light_quality.sh \
  ./checkpoints_cbam_refiner_v1_quality_20261029_192913_1274588 20 \
  > ./logs_cbam_refiner_v1/quality_20261029_192913_1274588/continue_11_20_launch.log 2>&1 &
```

原实验进程应已结束。续训保留保存的批量、梯度累积和训练设置，追加原训练日志，每轮评估全部 600 条。脚本校验起始轮次的评估与权重 SHA、验证 CSV 和评分协议，并用文件锁防止同一续训入口重复启动；达到 PESQ 3.4 时停止后续轮次，等待复核。

## 后续 PESQ 监督实验

`run_light_metric.sh` 从已有质量模型开始，保持生成模型、冻结基线、8 通道与原 STFT 不变。训练时新增 [MP-SENet 的 metric discriminator](https://github.com/yxlu-0102/MP-SENet/blob/main/models/discriminator.py)，用训练集的 clean 与当前 estimate 计算真实 16 kHz WB-PESQ，标签为 `(PESQ - 1) / 3.5`。生成器损失在现有损失上增加 `0.05 * (D(clean_mag, estimate_mag) - 1)^2`。判别器输入为最终波形重建后重新 STFT 的压缩幅度；对生成器保留梯度，对真实 PESQ 标签和判别器更新所用的 estimate 断开梯度。推理只加载原生成模型。

这是作者方法的适配。生成器继续每卡 batch 1、累积 4 次，等效全局 batch 8；包含有效标签时，判别器每个微批次更新一次，跨两卡平均梯度，等效 batch 2。与作者每次生成器更新对应一次判别器更新的安排不同，有效标签覆盖完整累积窗口时为 4:1。新阶段从选定权重开始，G/D 均使用新的 Adam，学习率均为 0.0001；阶段内的断点恢复保留双方模型、优化器和更新计数。验证损失使用原物理重建损失，最终模型按完整 600 条真实 PESQ 选择。

真实 PESQ 标签只在训练集计算。无法计算的静音或短片段不伪造分数，日志记录有效与无效数量；整轮无有效标签则停止。CPU 每卡顺序计算标签，避免每个 rank 再创建 15 个并行进程。额外 CPU 同步与判别器计算的实际耗时、以及能否提高当前数据的 PESQ，须由服务器确认。

待现有实验结束，使用其 PESQ 最佳权重及对应完整报告启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup bash run_light_metric.sh \
  ./checkpoints_cbam_refiner_v1_quality_实际运行名/model_epoch_最佳轮次.pt \
  ./logs_cbam_refiner_v1/实际运行名/pesq_selection/model_epoch_最佳轮次/summary.json \
  > ./logs_cbam_metric_v1_launch.log 2>&1 &
```

启动器拒绝与本仓库现有训练任务重叠，先对完整 600 条验证新阶段第 0 轮与源模型指标一致，再双卡训练最多 6 轮，逐轮完整评估；达到 3.4 时停止后续轮次，等待复核。当前单元检查及 CPU/Gloo 训练检查不代表服务器性能已经改善。

若训练已经完成但评估被中断，只补跑评估（替换实际目录名）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run_light_quality.sh \
  --evaluate-only ./checkpoints_cbam_refiner_v1_quality_实际时间戳_PID \
  > ./logs_cbam_refiner_v1/evaluation_retry.log 2>&1 &
```

## 提交结果

```bash
cd ~/EaBNet_light
git add logs_cbam_refiner_v1/
git commit -m "Add quality refiner training and evaluation results"
git push
```

主要结果位于 `logs_cbam_refiner_v1/quality_*/pesq_selection/model_epoch_*/summary.json`。训练未结束时上传日志仅用于检查当前状态，不能据此认定整个实验已完成。
