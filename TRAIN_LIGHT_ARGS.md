# train_light.py 参数说明

`train_light.py` 用于训练轻量版 FCAE-Att-DFSMN EaBNet 多通道语音增强模型。默认配置偏向 12GB 显卡稳定训练：较小的单步 batch、梯度累积、AMP，以及保守的 DataLoader 内存策略。

## 推荐启动命令

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 train_light.py \
  --parallel-mode auto \
  --resume yes \
  --resume-reset-lr yes \
  --learning-rate 1e-3 \
  --lr-reduce-metric val \
  --train-lr-patience 3 \
  --train-lr-min-delta 1e-3 \
  --train-lr-factor 0.5 \
  --use-amp yes \
  --model-amp yes \
  --grad-clip 3.0 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 0 \
  --pin-memory no \
  --prefetch-factor 1 \
  --persistent-workers no \
  --strict-memory yes \
  --segment-seconds 4 \
  --allow-tf32 yes \
  --cudnn-benchmark yes \
  --train-dir /data/ssd1/jinrui.yang/training_set \
  --val-dir /data/ssd1/jinrui.yang/validation_set
```

多 GPU DDP 训练时，把 `--nproc_per_node` 改为 GPU 数量，并设置 `--parallel-mode ddp`。

## 数据与输出路径

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--train-dir` | `/data/ssd1/jinrui.yang/training_set` | 训练集目录，目录下需要包含 `metadata.csv`。 |
| `--val-dir` | `/data/ssd1/jinrui.yang/validation_set` | 验证集目录，目录下需要包含 `metadata.csv`。 |
| `--checkpoint-dir` | `./checkpoints` | 相对路径，用于保存 `checkpoint_latest.pt` 和周期性 checkpoint。 |
| `--best-dir` | `./bestmodels` | 相对路径，用于保存验证集最优模型。 |
| `--log-dir` | `./logs` | 相对路径，用于保存 TensorBoard 日志。 |

`metadata.csv` 至少应包含 `sample_id`、`mixture_path`、`target_path` 三列。`mixture_path` 指向多通道混合语音 wav，`target_path` 指向干净目标语音 wav。

## 训练与优化

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num-epochs` | `100` | 最大训练轮数。 |
| `--batch-size` | `1` | 每个进程每一步的 batch size。12GB 显卡建议保持为 `1`。 |
| `--grad-accum-steps` | `4` | 梯度累积步数。有效 batch 大约为 `batch-size * grad-accum-steps * world_size`。 |
| `--learning-rate` | `1e-3` | Adam 初始学习率，与论文设置一致。 |
| `--weight-decay` | `0.0` | Adam 权重衰减。 |
| `--grad-clip` | `5.0` | 梯度裁剪阈值，设置为 `<=0` 表示关闭。 |
| `--seed` | `1234` | 随机种子。 |
| `--save-every` | `10` | 每隔 N 个 epoch 额外保存一个 checkpoint。 |

## STFT 与特征压缩

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--sample-rate` | `16000` | 期望的数据采样率，混合语音和目标语音必须一致。 |
| `--win-length` | `320` | STFT 窗长。16 kHz 下对应 20 ms。 |
| `--hop-length` | `160` | STFT 帧移。16 kHz 下对应 10 ms。 |
| `--n-fft` | `512` | FFT 点数，论文使用 512。 |
| `--power` | `0.5` | 幅度压缩指数。`0.5` 表示平方根压缩。 |
| `--segment-seconds` | `4.0` | 每条样本裁剪长度，单位为秒。如果仍然 OOM，可降到 `3` 或 `2`。 |
| `--train-random-crop` | `yes` | 训练时随机裁剪；验证时使用确定性裁剪。 |

## DataLoader 与内存

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num-workers` | `4` | DataLoader worker 数量。开启 `--strict-memory yes` 后，实际值会被强制为 `0`。 |
| `--pin-memory` | `yes` | 是否启用 pinned host memory。开启 `--strict-memory yes` 后实际关闭。 |
| `--persistent-workers` | `no` | 是否在 epoch 间保留 DataLoader worker。worker 数为 `0` 时无效。 |
| `--prefetch-factor` | `2` | 每个 worker 预取的 batch 数。worker 数为 `0` 时无效。 |
| `--strict-memory` | `yes` | 保守内存模式，会强制 `num_workers=0`、关闭 pinned memory、关闭 persistent workers，并将预取设为 `1`。 |
| `--empty-cache-each-epoch` | `yes` | 每个 epoch 后调用 `torch.cuda.empty_cache()`。 |
| `--empty-cache-every-batches` | `0` | 每 N 个 batch 调用一次 `torch.cuda.empty_cache()`，`0` 表示关闭。显存碎片严重时可尝试 `20`。 |
| `--mem-log-every-batches` | `100` | 每 N 个 batch 打印一次内存信息。 |
| `--malloc-trim-every-batches` | `100` | Linux/glibc 下，每 N 个 batch 尝试释放 CPU heap 空闲页。 |
| `--gc-every-batches` | `100` | 每 N 个 batch 运行 Python 垃圾回收。 |
| `--malloc-arena-max` | `2` | Linux/glibc arena 数量的 best-effort 限制。 |
| `--log-ram-tree` | `yes` | 记录当前进程及其子进程的总 RAM。 |

脚本默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，用于缓解 CUDA allocator 碎片化。

## AMP 与后端选项

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--use-amp` | `yes` | 启用 AMP 和 GradScaler。 |
| `--model-amp` | `yes` | 模型前向使用 AMP；STFT 特征构建保持 fp32，模型内部 Fourier block 也会在 FFT 前转为 fp32。 |
| `--allow-tf32` | `yes` | 在 Ampere 或更新 GPU 上允许 TF32 matmul/cuDNN kernel。 |
| `--cudnn-benchmark` | `yes` | 对固定输入形状启用 cuDNN autotuner，以提升速度。 |

## 模型结构

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target-ref-mic` | `0` | 如果 target wav 是多通道，选择该通道作为目标。 |
| `--num-mics` | `8` | 输入麦克风通道数，必须不大于 mixture wav 的通道数。 |
| `--channels` | `64` | CRED/FCAE/FCAD 的主通道数。 |
| `--embed-dim` | `64` | embedding tensor 通道数，也是 LSTM beamforming 头的输入维度。 |
| `--kd1` | `5` | 为兼容原 EaBNet 保留，轻量版 DFSMN 的显式记忆阶数由 `--dfsmn-memory-size` 控制。 |
| `--cd1` | `64` | DFSMN hidden units。 |
| `--d-feat` | `256` | 为兼容原 EaBNet 保留，轻量版 CRED 不直接使用。 |
| `--p` | `6` | 为兼容原 EaBNet 保留，轻量版 CRED 不直接使用。 |
| `--q` | `3` | 为兼容原 EaBNet 保留，轻量版 CRED 不直接使用。 |
| `--bf-type` | `lstm` | Beamforming head 类型，可选 `lstm` 或 `cnn`。 |
| `--topo-type` | `mimo` | 输出拓扑，可选 `mimo` 或 `miso`。 |
| `--intra-connect` | `cat` | 为兼容原 EaBNet 保留。轻量版 CRED 的 skip path 使用 concat。 |
| `--norm-type` | `BN` | 归一化类型，可选 `BN`、`IN`、`cLN`。论文使用 BN。 |
| `--dfsmn-layers` | `3` | 共享 DFSMN memory layer 的重复次数，论文使用 3。 |
| `--dfsmn-memory-size` | `20` | 共享 DFSMN block 的左上下文记忆阶数，即历史帧 memory taps 数。 |
| `--is-causal` | `yes` | 是否使用因果时间建模。 |
| `--is-u2` | `yes` | 为兼容原 EaBNet 保留，轻量版 CRED 不直接使用。 |

## 并行与设备

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--gpu-ids` | 空 | 单进程 DataParallel 使用的 GPU ID，例如 `0,1`。为空表示使用所有可见 GPU。DDP 使用 `LOCAL_RANK`。 |
| `--parallel-mode` | `auto` | 可选 `auto`、`none`、`dp`、`ddp`。多 GPU 训练推荐使用 `ddp`。 |
| `--cpu-threads` | `8` | PyTorch intra-op CPU 线程数。 |
| `--interop-threads` | `1` | PyTorch inter-op CPU 线程数。 |
| `--cpu-thread-scale` | `1.0` | 作用于 `cpu-threads` 的缩放系数。 |

## 断点续训与学习率调度

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--resume` | `yes` | 如果 `checkpoint_latest.pt` 存在，则从该 checkpoint 恢复训练。 |
| `--resume-reset-lr` | `no` | 恢复训练后，是否用 `--learning-rate` 覆盖 optimizer 中保存的学习率。 |
| `--train-lr-reduce-on-plateau` | `yes` | 当选定的 loss 指标不再改善时降低学习率。 |
| `--lr-reduce-metric` | `val` | 用于 plateau 判断的指标，可选 `train` 或 `val`。 |
| `--train-lr-patience` | `2` | 连续多少个 epoch 无改善后降低学习率。 |
| `--train-lr-factor` | `0.5` | 学习率衰减倍率。 |
| `--train-lr-min-delta` | `0.0` | loss 至少下降多少才算改善。 |
| `--min-learning-rate` | `1e-6` | 学习率下限。 |

## 非有限 loss 处理

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--skip-non-finite-batches` | `yes` | 遇到 NaN/Inf loss 时跳过该 batch。DDP 下会同步各 rank 的决策。 |
| `--max-non-finite-batches-per-epoch` | `20` | 每个 epoch 允许跳过的非有限 loss batch 上限，超过后终止。 |
| `--stop-on-non-finite` | `no` | 遇到 NaN/Inf loss 时是否立即终止训练。 |
| `--non-finite-log-max-ids` | `8` | 打印非有限 loss batch 样本 ID 的最大数量。 |

## OOM 调参顺序

1. 保持 `--batch-size 1`。
2. 将 `--segment-seconds 4` 降到 `3` 或 `2`。
3. 保持 `--use-amp yes --model-amp yes`。
4. 尝试设置 `--empty-cache-every-batches 20`。
5. 如果显存仍然不足，降低模型宽度，例如 `--channels 48 --embed-dim 48 --cd1 48`。
