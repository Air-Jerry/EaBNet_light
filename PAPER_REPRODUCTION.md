# CTS-Net 论文复现核对记录

本项目的复现目标是用户提供的 **Two Heads are Better Than One: A Two-Stage Complex Spectral Mapping Approach for Monaural Speech Enhancement**，IEEE/ACM TASLP 29 (2021), 1829–1843，DOI：[10.1109/TASLP.2021.3079813](https://doi.org/10.1109/TASLP.2021.3079813)。下文页码指 PDF 页码；括号内为期刊页码。

附件 SHA-256：`095ba8c9cc5397632feb6d0cd7344b850fcf88eab0d67d17111f1ff02c9633e3`。

**目前能够核验的是论文公开的数学结构、参数表和训练要求，不能据此保证作者未披露的实现细节完全一致。训练结果未达标时，仍须区分实现问题、实验协议差异和论文方法本身的局限。单次训练失败不能直接归因于论文。**

## 1. 版本与证据来源

- 以用户附件的 TASLP 期刊版为准：第 3 页公式 (7)–(11)，第 4 页 Fig. 1–2，第 5 页 Sec. III-D/E，第 6 页 Table I、公式 (17)–(21) 与 Sec. IV，第 12–13 页基准数据和结果。
- 原 `EaBNet_light.py` 的 EaBNet 是不同的多通道模型；该文件不能通过改类名变成本文的单通道 CTS-Net。
- 论文第 7 页脚注 4 指向作者的[期刊版演示仓库](https://github.com/Andong-Li-speech/CTS_TASLP_demos)。该链接在论文中被描述为增强语音示例，并不等于已经公开、已核验的训练源码。
- 本轮检索找到了会议版仓库，但期刊演示仓库的进一步访问未成功，源码、提交版本和 LICENSE 未能取得。因此，当前实现不得标注为“作者代码逐行一致”或“作者实现已验证”。
- `Andong-Li-speech/CTS-Net` 对应先前会议工作，不能在未逐项核对时拿来替代本文期刊版；尤其不可从其他版本直接搬用幅度压缩、归一化和训练损失。

## 2. 架构核对

`CTSNet.py` 的数据流对应公式 (7)–(11)：

1. 输入 noisy RI，计算幅度 `|X|`。
2. ME-Net 仅预测幅度；用 noisy 相位形成 coarse RI。
3. CS-Net 输入依次为 noisy real、noisy imag、coarse real、coarse imag，四通道拼接。
4. CS-Net 有独立 real/imag 两个 decoder，预测复数残差；输出是 coarse RI 加 residual RI。
5. 联合训练允许梯度经 coarse RI 返回 ME-Net，不能在两阶段间 `detach()`。

| 部分 | 论文要求 | 对应证据 |
| --- | --- | --- |
| 频谱 | 161 个频点；ME 输入 1 通道，CS 输入 4 通道 | 第 5–6 页，Table I |
| Encoder | 5 个 Conv-GLU，每层输出 64 通道 | Sec. III-D.1 |
| 卷积核 | 第一层时间×频率为 `(2,5)`，后四层 `(2,3)` | Sec. III-D.1 |
| Stride | 全部 `(1,2)`，时间长度不变 | Table I |
| 频率维 | `161 → 79 → 39 → 19 → 9 → 4` | Table I |
| 瓶颈变形 | `64 × T × 4` 与 `256 × T` 间转换，不任意重排时间维 | Table I |
| S-TCM 数量 | 每阶段 3 组，每组 6 个，共 18 个；两阶段不共享参数 | Sec. III-D.2 |
| Dilation | 每组均为 `1,2,4,8,16,32` | Sec. III-D.2 |
| S-TCM 压缩 | 输入 `1×1 Conv:256→64`，输出 `1×1 Conv:64→256` | Fig. 2、Table I |
| S-TCM 分支 | main/gate 各自独立 `PReLU → Norm → 普通 dilated Conv(k=5)`；gate 接 Sigmoid | Fig. 2(b) |
| S-TCM 输出 | 两分支相乘，再 `PReLU → Norm → 1×1 Conv`，加模块输入 | Fig. 2(b) |
| Dilated Conv | `64→64` 普通卷积，不能换成 depthwise | Sec. III-C |
| Decoder | Encoder 的镜像；每层先拼接对应 skip，输入为 128 通道 | Table I |
| Decoder 频率维 | `4 → 9 → 19 → 39 → 79 → 161` | Table I |
| Decoder 输出通道 | 前四层 64，末层 1 | Table I |
| 输出投影 | 每个 decoder 后为 `Linear(161,161)` | Table I、Fig. 1(e) |
| 最终激活 | ME 用 Softplus；CS 的 real/imag 为线性输出 | Sec. III-D |

两个独立卷积与一个双倍输出通道卷积再 split 的 GLU 在参数独立时等价。该等价实现不改变公式 (13)。

## 3. 输入、STFT 与损失

- 论文指定单通道、16 kHz、20 ms Hann/Hanning 窗、50% overlap、320-point FFT，即 `win_length=320, hop_length=160, n_fft=320`。沿用其他 FFT 设置会改变 Table I 结构。
- **本文没有规定幅度 power compression。** 论文公式按原始幅度和复数频谱书写，所以本文复现配置使用 `power=1.0`；其他 power 值只能作为明确标记的扩展实验。
- `cts_features.py` 将同一 STFT/iSTFT 实现用于训练和推理；保留现有 waveform/tensor 数据接口，但单通道约束必须显式满足。
- 当前采用 PyTorch 的 periodic Hann、`center=True`、reflection padding、`normalized=False`、one-sided STFT。**这些边界和归一化选项不是论文明确披露的内容。** 需要作者实现或更完整补充材料确认；音频往返重建通过也不能证明与作者选择一致。
- 对变长 batch，应先按各自有效长度作 STFT，再补齐频谱；直接对补零后的波形统一反射延拓，会令短样本边界依赖同 batch 最长样本。
- noisy 与 clean 不能各自独立归一化，也不能在推理时利用 clean 目标做能量匹配。否则训练目标或测评分数会改变。

按公式 (17)–(21)：

```text
L_ME = ||estimated_magnitude - clean_magnitude||²
L_RI = ||estimated_real - clean_real||²
     + ||estimated_imag - clean_imag||²
L_Mag = ||sqrt(estimated_real² + estimated_imag²) - clean_magnitude||²
L_joint = 0.5 L_RI + 0.5 L_Mag + 0.1 L_ME
```

这里 RI 是 real/imag 两项之和。若对两通道一起执行默认 `mean()`，却只对幅度执行单通道 `mean()`，RI 项会少一个系数 2；不能沿用旧损失而仍声称 `alpha=0.5` 与本文一致。

当前实现对上述三项统一除以有效 `T×F` 单元数，并在计算之前屏蔽补齐位置；保留三项的相对权重。论文写的是 Frobenius 平方和，没有披露 minibatch reduction、变长样本加权和数值稳定细节，因此统一有效单元平均属于明确记录的实现约定。为了消除补齐对 InstanceNorm 的影响，CTS 训练入口将有效帧数传给模型；不同长度的样本先分别前向计算，再补齐输出用于损失。自行调用模型处理变长 batch 时，同样必须传入 `frame_lengths`，只设置损失掩码不能消除该影响。

## 4. 训练协议与“保留训练逻辑”的边界

论文的优化流程包含不可省略的两阶段：先单独训练 ME-Net 至收敛，再加载最佳 ME 权重进行 ME+CS 联合训练。一次从随机初始化开始的联合训练是论文 Table V 讨论的消融，不能替代主实验。

| 项目 | 论文描述 |
| --- | --- |
| 优化器 | Adam，`betas=(0.9,0.999)` |
| 第一阶段 LR | ME `0.001` |
| 第二阶段 LR | ME `0.0001`；CS `0.001` |
| 联合损失 | `alpha=0.5, lambda=0.1` |
| 训练 chunk | 8 秒，16 kHz 时为 128000 samples |
| Batch | utterance 级 batch size 8 |
| LR 衰减 | 连续 3 次验证损失上升后减半 |
| Early stopping | 连续 5 次验证损失上升后停止 |
| 总 epoch | 文中写共 60 个，但未明确如何分配到预训练和联合训练 |

实现新增 `train_cts.py`，直接复用原 `EnhancementDataset`、`collate_batch` 和 `run_epoch`。`train_light.py` 仅增加默认关闭的可选接口与可复用参数解析，原 EaBNet 默认优化行为保留。`EaBNet_light.py`、`evaluate_light.py` 未修改。CTS 入口单独实现两阶段调度、损失和参数配置；若把“不改变训练逻辑”解释为连原单阶段目标函数和优化流程也不能变化，这一要求与复现论文两阶段主实验无法同时满足。

CTS 默认 ME 上限 30 轮，总预算 60 轮；30 不是论文提供的阶段划分。ME 达到连续 5 次验证集最佳损失不改善后，加载最佳 ME 并开始 joint，joint 使用剩余预算。若 ME 到上限仍未满足此判据，则保存 checkpoint 并明确失败；可扩大 `--pretrain-max-epochs` 和总预算后续训，但扩大总预算属于偏离文中 60 轮的实验。`--allow-unconverged-pretrain yes` 仅是记录在案的调试覆盖，不能作为主实验完成收敛的依据。

以下细节仍需作者确认，当前实验必须在配置和日志中记录选择：

- ME “至收敛”的可执行判定；60 epochs 是跨阶段总和还是作者脚本另有定义，文中没有各阶段分配。
- “连续损失上升”相对上一次还是历史最佳；相等损失怎样计数；LR 减半是否重置停止计数；论文没有可执行伪代码。
- Adam 的 eps、weight decay、梯度裁剪、初始化和随机种子；这些未披露值不能被称为论文明确要求。
- AMP、DP/DDP、梯度累积、effective batch size 和不同设备的数值差异。`batch 8 × accumulation 4` 已改变论文的优化批量。
- 重启必须恢复阶段、优化器、两个参数组 LR、scheduler 计数和随机状态；不能把预训练 checkpoint 当作已联合训练模型。
- 划分训练/验证集不能混入同一 clean/noise 切片产生的数据泄漏；不能把跳过 NaN/Inf 批次后的训练默认为完整成功。

## 5. InstanceNorm 与因果性尚有歧义

论文 Sec. III-D.1 明确写 InstanceNorm，Fig. 2 中 temporal block 仅标 `Norm`。论文同时报告 causal 模型，但没有给出 cumulative IN、channelwise LN 或逐帧 normalization 的公式。

当前 `norm_type="IN"` 按文字使用标准 InstanceNorm。标准 IN 计算完整时间轴的统计量，即使卷积全部左侧补齐，当前帧也会依赖未来帧；加上 centered STFT 的 lookahead，**该配置不能被标注为严格的端到端流式因果实现**。

`norm_type="cIN"` 提供累积统计的研究选项，它是当前实现的明确扩展，未得到作者确认，不可用其结果无条件对照论文 causal 行。其频谱域未来扰动测试通过也不能消除 centered STFT 的窗口延迟。要复现严格 causal 行，仍需确认作者 Norm 定义和前端延迟约定。

IN 的 affine 参数、eps、PReLU 是共享参数还是逐通道，以及末层归一化细节也没有完整披露。当前选择应随 checkpoint 保存，以免不同设置混用。

## 6. 参数量基准

第 12 页 Table VII 列出 ME-Net **1.96M**，完整 CTS-Net **4.35M**；MACs 分别为 **2.01G/s、5.57G/s**。该表的文字标题称 Million，但数值列明确标 `MACs(G/s)`，引用时应保留这一单位差异说明。

参数量用于结构核对，不是实现一致性的充分证明。当前运行统计为 ME **1,968,167**，CS **2,395,150**，总计 **4,363,317** 个参数，约 4.36M，与论文表中 4.35M 存在小幅差异。PReLU 参数共享、IN affine 和 bias 的未披露设置可能解释差异，但尚未证实；不能因此宣称参数级完全一致，也不应仅为凑数而改动结构。当前卷积和 Linear 含 bias，PReLU 逐通道，IN 使用 affine、eps=1e-5、无 running statistics。

## 7. 数据与评估必须匹配目标表格

### WSJ0-SI84：Table II–III

- 7138 utterances、83 speakers（42 male/41 female）；训练 5428、验证 957，来自 77 speakers。
- seen/unseen 两套测试，每套 150 utterances、6 speakers（3 male/3 female）；seen 与训练说话人重合，unseen 不重合。论文没有提供精确 utterance ID manifest。
- 噪声来自 Interspeech 2020 DNS Challenge，随机约 20000 段、约 55 小时。每次随机切噪声并与随机 clean 混合。
- 训练 SNR `[-5,-4,-3,-2,-1,0] dB`；训练 150000 对、验证 10000 对；训练约 300 小时。
- 测试噪声为 NOISEX92 的 babble 和 factory1；SNR 为 `[-6,-3,0,3,6] dB`，每个 case 150 对。
- 报告 **NB-PESQ、ESTOI、SDR**。第 12 页脚注 6 明确 WSJ0 的 PESQ 为 narrow-band。不能用 WB-PESQ、普通 STOI 或 SI-SDR 替换同名对照数值。
- SDR 的实现及滤波/对齐约定需与参考 [62] 匹配，并记录实际评估库版本。

### DNS Challenge：Table VIII

- corpus 包含超过 500 小时 clean、2150 speakers，超过 180 小时 noise；作者生成约 3000 小时 noisy-clean 用于该基准训练。
- 官方评估分 reverb/no-reverb，各 150 clips，SNR 0–20 dB。
- 报告 WB-PESQ、NB-PESQ、STOI (%)、SI-SDR (dB)。**与 WSJ0 的 ESTOI/SDR 不同。**
- 论文 CTS 行：reverb `3.02 / 3.47 / 92.70 / 15.58`；no-reverb `2.94 / 3.42 / 96.66 / 17.99`，顺序同上一条。
- 生成器版本、精确 manifest、RIR 和 noisy-clean 合成参数未披露完整；不能用任意 DNS 数据包就称完全相同训练集。

### VoiceBank + DEMAND：Table IX

- 训练 11572 utterances，测试 824，测试 SNR `2.5,7.5,12.5,17.5 dB`。
- 报告 WB-PESQ、CSIG、CBAK、COVL；CTS 行为 `2.92 / 4.25 / 3.46 / 3.59`。
- 论文 ME 行为 `2.90 / 4.25 / 3.42 / 3.59`；该 corpus 上两阶段增益本来就很小，不应把低 SNR WSJ0 的增益直接当作本数据集期望。
- 本文没有完整描述该数据集的验证集划分及全部训练超参数是否与 WSJ0 相同。表格无法替代具体训练/验证 split。

以上三个目标表使用不同训练数据和评估指标。现有任意数据目录即使能直接接入，也不能因此与全部论文表格形成受控比较。仅有 `SI-SDR / SI-SNR` 的评估脚本不能完整验证论文结果；缺少指标实现时应明确报告缺项，不能填入另一种指标。

## 8. 接受标准

可以通过自动检查验证的事项包括：频率维逐层吻合、两分支和两阶段参数独立、residual 恒等路径、已知手算损失、padding 掩码、有限梯度、STFT/iSTFT 往返、checkpoint 恢复和推理前端一致。通过这些检查只能降低已知实现错误的风险。

要进一步排除复现实验错误，还需要固定数据 manifest/哈希、切分和 SNR 协议、全部配置、代码和环境版本、随机种子、每阶段收敛曲线、最佳 checkpoint、失败批次数，以及与原始指标相同的评估程序。重复随机种子的波动也应纳入结果判断。

当前仍未取得作者源码、权重、精确数据清单及上述未披露细节，也尚无完整原协议训练结果。因此目前只能标为 **“依据期刊正文重建并验证已公开要求”**，不能标为 **“完全复刻，未达标即可认定论文有误”**。若后续结果不足，先定位数据、前端、实现和优化差异，再讨论方法本身；未披露或访问失败不是论文结论错误的证据。

## 9. 运行与验证记录

依赖见 `requirements-cts.txt`。本轮使用项目内 `.venv`，Python 3.12.14、PyTorch 2.14.0+cpu。GPU 服务器应安装与硬件兼容的 PyTorch CUDA wheel；本轮没有验证 CUDA/NCCL。

在项目目录运行，保持现有 `metadata.csv` 三列 `sample_id,mixture_path,target_path`：

```bash
python -m pip install -r requirements-cts.txt
python train_cts.py --train-dir /data/ssd1/jinrui.yang/training_set --val-dir /data/ssd1/jinrui.yang/validation_set
python evaluate_cts.py --val-dir /path/to/wsj0_test --checkpoint ./bestmodels_cts/best_model.pt --protocol wsj0 --estimate-dir ./estimate_set_cts
python -m pytest tests -q
```

Windows 本轮测试环境可使用 `.\.venv\Scripts\python.exe` 代替 `python`。上面的 Linux 数据路径来自原脚本，需指向实际数据；本工作区没有论文规模的数据集。`train_cts.py` 不会自动生成/下载混合数据。

默认读取 mixture 首通道；即使现有 wav 有 8 通道，CTS 仍只用第 0 通道，不做 beamforming。目标单声道直接读取，目标多声道由 `--target-ref-mic` 选择；选不同参考位置时需确认声学对齐。CSV、原 Dataset 的裁剪接口和 batch 张量键不变。新增数据预检会拒绝采样率/配对长度错误、无效参考通道、重复 sample_id，以及跨训练/验证集完全相同的目标文件路径或 SHA256；哈希不同不代表说话人、切片或噪声一定无泄漏。

默认保存至独立的 `checkpoints_cts`、`bestmodels_cts`、`logs_cts`。完整数据/源码哈希清单在 `reproducibility_manifest.json`，参数在 `run_config.json`，完成/失败状态在 `training_summary.json`。每个 checkpoint 记录阶段、优化器、随机状态和清单摘要。预训练最佳文件为 `best_me.pt`，joint 最佳文件为 `best_model.pt`。默认自动续训；源码、数据、运行环境或受保护训练参数变动会明确拒绝复用旧运行。重新实验应同时选择新的 checkpoint、best、log 目录。

评估从 checkpoint 读取前端参数，拒绝将 FFT 改回 512 等不一致设置，也拒绝将 ME-only checkpoint 当作完整 CTS-Net。`--protocol dns` 使用 DNS 的四项指标；当前**没有实现 VoiceBank 的 CSIG/CBAK/COVL**，因此本项目评估入口尚不能复核 Table IX 全部指标。`--pesq-mode wb` 是 WSJ0 的显式非论文变体。输出 WAV 为 FLOAT，不做 clean 参考增益匹配；`metadata.csv` 和 `evaluation_status.json` 记录逐样本分数、完整分母及失败原因。异常指标不会被静默忽略。

新增测试覆盖 Table I 形状、独立 S-TCM、残差恒等路径、手算 RI/Mag/ME 损失、padding 与零输入推理、真实反向和小批拟合、STFT 往返、变长单独/批量一致、两阶段真实训练、差异 LR、完整 RNG 续训、数据/源码校验以及真实音频六项评分（增强和 mixture 各三项）。因果扰动测试分别确认 `cIN` 的频谱帧因果性和标准 `IN` 的未来依赖，不能把前者测试结果用于声明后者严格因果。

额外执行过双进程 CPU/Gloo DDP 两阶段训练，并对旧 `run_epoch` 默认路径与 Git 原版进行了实际对照，损失和参数更新一致。短训练的断点续训与同配置连续训练，全部模型参数逐项相等。短训练使用显式未收敛覆盖以测试阶段转换；这些测试**不构成论文效果或收敛复现**。

数值边界：整条 mixture 全静音或恒定 DC 时，多层零方差 IN 可以造成真实的 FP32 梯度溢出。训练前端对此明确报错，保留全静音 clean target（前提是 mixture 有正常信号）及零输入推理。极小幅度等其他异常若产生非有限梯度，会在更新参数前中止，不能靠跳过坏 batch、替换 NaN 或改归一化常数掩盖。该输入检查是新增实现防护，并非论文披露的预处理步骤。

最终自动测试计数和本轮可验证证据见 `VERIFICATION.md`；其中随机模型/合成波形只用于程序检查。本文中的 benchmark 分数始终是论文报告值，不是本项目训练结果。
