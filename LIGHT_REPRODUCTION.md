# FCAE-Att-DFSMN 复现核对记录

## 当前结论

**这是按已披露公式修正、可测试的实现，尚未达到作者实现等价或论文结果复现认证。不能据此把训练不达标归因于论文本身。**

依据是用户提供的 ICASSP 2023 论文 *A Lightweight Fourier Convolutional Attention Encoder for Multi-Channel Speech Enhancement*，DOI [10.1109/ICASSP49357.2023.10095716](https://doi.org/10.1109/ICASSP49357.2023.10095716)。用户确认没有额外作者代码或配置。本次检索未找到可验证的目标论文官方实现；[原始 EaBNet 仓库](https://github.com/Andong-Li-speech/EaBNet)对应参考文献 [9]，不是这篇 FCAE 论文的实现。

默认模型实测 **800,674** 个可训练参数，论文表 1 为 **约 0.74M**，仍未匹配。论文还省略了损失函数、若干结构参数及精确数据生成配置。因此，无法同时保证“训练逻辑原样保留”与“整个实验唯一还原”。本记录明确区分论文直接披露、引用方法补充、实现选择和未验证事项。

## 当前接口与历史修改范围

当前评估统一使用 `evaluate_light.py`，支持原 `metadata.csv` 和分别指定 mixture/target 路径。它从 checkpoint 恢复候选结构及训练前处理，计算 PESQ、ESTOI、SDR，并保留 STOI/SI-SNR 和带噪基线；可导出音频及逐样本 CSV。此前的独立评估入口已合并并删除，具体命令见本文“运行和验证”及 [LIGHT_STRUCTURE_EVIDENCE.md](LIGHT_STRUCTURE_EVIDENCE.md)。本次整合只修改评估，不修改当前训练逻辑或模型实现，已有训练权重无需重训。

以下是最初公式修正阶段的范围记录：

- 修改 `EaBNet_light.py`，保留 `EaBNet` 构造调用及 `forward(inpt)` 接口：输入 `(B,T,257,M,2)`，输出 `(B,2,T,257)`。
- 当时的 `train_light.py` 和 `evaluate_light.py` 逐字节保持原样，包括数据加载、裁剪、STFT、压缩、优化器、AMP/DDP、梯度累积、学习率、日志、断点流程和旧评估行为；此描述不是当前评估文件的状态。
- `com_mag_mse_loss` 原函数保持原样；没有将 RI 项从均值改成求和，也没有增加训练损失。
- 当时新增独立四指标评估，仍读取原 `metadata.csv` 接口；现已并入 `evaluate_light.py`。
- 新增针对 light 模型的测试及 `audit_light.py`，不修改、恢复或移除已有 CTS 文件状态。
- 不符合 512 点 FFT 的输入现在明确报错；解码尺寸错误不再用裁剪/补零悄悄掩盖。

历史基线文件 SHA256，仅用于追溯最初交付，不是当前文件应满足的哈希：

| 文件 | 最初公式修正阶段的 SHA256 |
|---|---|
| `train_light.py` | `608a8d7ac3dd6e615c502613e525f764fc67e9e9c4c9c3d0e167ab2a0211f391` |
| `evaluate_light.py` | `1f86987e39c011293ca774866712e12e37368db31352af030d1f2ff920d4f002` |

## 论文到代码对应

| 依据 | 本次实现与核对 | 证据等级 |
|---|---|---|
| 式 (2) | `sum(conj(W_m) * X_m)`；原实现使用 `W_m * X_m`，已修正两处实虚乘积符号 | 正文明确 |
| 图 1、3.1 节 | LayerNorm → 两层 64 单元 LSTM → 一个输出 Linear；删除原额外 Linear/ReLU | 按已披露结构对齐 |
| 图 2、4.2 节 | 5 层 FCAE + 3 次共享 DFSMN + 5 层 FCAD；跳接 CA → SA | 正文明确 |
| 图 2 | 默认等宽 CRED 最后一层 FCAD 后直接输出 embedding，去掉额外 Conv-BN-PReLU；非等宽变体仅保留通道投影 | 按图选择；简图不能排除作者另有未披露投影 |
| 式 (3) | Conv → BN → PReLU | 默认 BN 与正文一致；IN/cLN 为兼容变体 |
| 式 (4)、(6) | 只在频率轴做实 FFT，将全部 real 通道与全部 imag 通道拼接 | 正文明确 |
| 式 (5) | 沿通道平均/最大池化，分别 ReLU/Conv，相加 sigmoid 后乘输入 | 公式明确；核大小等未披露 |
| 式 (7) | 拼接后的谱域 Conv → BN → ReLU | 顺序明确；选用 1×1 卷积是实现选择 |
| 式 (8)、(9) | SA 后按通道分成实虚部，实逆 FFT 显式传入原频点长度 | 正文明确；FFT 默认规范化约定未披露 |
| 式 (10) | 逆 FFT 分支与 CA(Q) 相加，经 1×1 Conv 输出 | 正文/图 3 明确 |
| 式 (11) | TF 全局平均/最大池化拼接，经 Conv/sigmoid 后乘输入 | 公式明确；池化范围、卷积细节为实现选择 |
| 4.2 节 | 频率×时间卷积核 `(5,2),(3,2)×4`，步长 `(2,1)`；代码按时间×频率排列 | 正文明确 |
| 4.2 节 | 16 kHz、320 点 Hann 窗、160 点帧移、512 点 STFT | 现有前处理已一致；center/padding 未披露 |
| 4.2 节及第 4 页首句 | Adam、初始 LR=0.001，验证损失连续两轮未下降时 LR×0.5 | 训练默认值一致；脚本顶部示例覆盖成 patience=3，不应当作论文设置 |

式 (2) 的修正是权重定义与论文一致：无约束网络也可以学习共轭后的系数，因此原符号约定本身不证明模型表达能力不足，更不意味着这项修正必然提高指标。

该算子实际接收原训练前端提供的压缩谱。正文式 (2) 写在 STFT 域，但没有解释是否采用此压缩；本次不改变前端，因而这里仅能确认复数算子的形式一致，不能认证整个前端与作者相同。

频率维度必须精确往返：

```text
257 → 127 → 63 → 31 → 15 → 7 → 15 → 31 → 63 → 127 → 257
```

各层时间维保持不变。实 FFT 输出实际为 `floor(F/2)+1`；论文写作 `F/2` 是简略记号，不能据此丢掉 DC/Nyquist 或错误重建奇数长度。半精度 FFT 提升为 float32，float64 参考计算保留原精度。

## DFSMN 的证据和选择

目标论文仅明确三层共享权重、64 隐藏单元及时间建模，没有给出递推公式。本实现采用其参考文献 [16]，[Bi et al., §2.2–2.3，式 (1)–(5)](https://arxiv.org/pdf/1802.09194)：线性投影后计算包含可学习 `a0` 的记忆，再加入上一层记忆输出，最后经过仿射变换和 ReLU。原实现的记忆前 BN/PReLU、记忆后 BN 和层输入残差不等价于该定义，已改正。

这里的“上一层记忆”是同次前向中跨网络深度传递的状态，不是跨 utterance 缓存，也不是 LSTM 的时间隐藏状态。三个调用共用同一个模块和参数，记忆状态每次模型前向从空开始。

仍无法验证的选择：20 个历史帧 + 当前帧、步长 1、默认无未来帧、64 维投影、各频率作为独立序列但共享参数。参考文献 [17] FRCRN 还有不同的组织方式，不能把其完整配置直接认定为目标论文配置。当前实现是有文献依据的补全，**不是确认过的作者配置**。

## 参数量与尚未解决的歧义

| 部分 | 可训练参数 |
|---|---:|
| 五层 FCAE | 255,764 |
| 共享 DFSMN（重复三次只计一次） | 9,664 |
| 五组 skip CA/SA | 41,770 |
| 五层 FCAD | 425,748 |
| 两层 LSTM、LayerNorm、Linear | 67,728 |
| 合计 | **800,674** |

论文表 2 去除 skip attention 后从约 0.74M 到约 0.73M；当前 skip attention 自身占 41,770 个参数，也无法解释该消融差值。参数差异说明仍存在待核实结构细节；不应为了凑到 0.74M 随意改变通道数、共享关系或缩减 attention。

尚未披露或无法核验：

- SA 卷积核、bias、分支是否共享权重；CA 卷积形态及是否有隐含降维。
- skip 使用拼接还是相加、详细解码器通道配置、embedding 宽度和额外投影。
- DFSMN 记忆阶数/步长、投影宽度、频率展开方式和初始化。
- 卷积时间 padding、FFT normalization、PReLU 参数共享粒度及所有初始化。
- 训练损失及权重、0.5 幅度压缩、batch size、训练轮数、裁剪、梯度裁剪、AMP 和随机种子。
- 训练/验证样本清单、RIR 实现及随机种子、干净目标是直达声还是含早期反射的信号、参考麦克风定义。
- PESQ 模式/版本等指标实现。新评估显式采用 16 kHz 宽带 PESQ，但无法证明与作者的工具链相同。

`is_causal=yes` 仅约束 DFSMN 记忆。SA 对称时间卷积、CA 全时间池化和训练态 BN 都可能使用未来帧，整个模型不能宣称严格流式因果。论文所提原始 EaBNet 的 causal 属性不能自动继承。

## 保留训练流程带来的已知限制

下面问题属于原脚本；按用户“不改变已有训练逻辑”的约束，本次没有悄悄修正：

1. `run_epoch` 默认每次 loss 除以完整 `grad_accum_steps`；最后一个不足该长度的累积窗口梯度会偏小。默认累积 4 时，要核对实际每 rank 的 DataLoader 批次数是否整除 4，或自行选择已有 `--grad-accum-steps 1` 参数绕开该情况。
2. `main` 设置 torch seed，但随机裁剪使用 Python `random`。默认严格内存配置使 workers=0，torch seed 不足以复现裁剪；断点也没有保存全部 RNG 状态。
3. 变长 batch 先对波形补零后 STFT，短样本末端与单独 reflect-padding STFT 不相同；全时间 attention 也可能受补零长度影响。默认 batch size=1 可避免跨样本补零问题，损失掩码无法消除模型内部 padding 影响。
4. DDP 验证采样可能补重复样本，epoch loss 默认按 batch 聚合；不能将该 loss 等同于每条测试 utterance 的均值指标。
5. 默认跳过非有限 loss，某些失败样本可能不进入训练统计。排错时可使用已有 `--stop-on-non-finite yes --skip-non-finite-batches no`。
6. 当前训练和验证都默认裁剪到 4 秒；测试时完整 utterance 的统计不等同于裁剪验证损失。

因此“保留现有训练流程”和“排除全部复现过程误差”在目前条件下不能同时成立。以上不是论文失败的证据。

## 数据与结果的比较范围

论文 4.1 节的数据要求：AISHELL-1、AISHELL-3、VCTK、LibriSpeech train-clean-360；筛选 SNR>15 dB 的干净语音；MUSAN/Audioset 噪声；8 麦线阵、5 cm 间距；房间约 3×3×3 到 8×8×3.5 m；RT60 0.1–1.1 s；声源距离 0.5–5 m；混合 SNR -5–25 dB；超过 5,000 个 RIR；约 60,000 个训练样本、1,600 个验证样本。

论文表 1 使用 **ConferencingSpeech2021 development test set**，不是任意本地 validation_set。其参考值为：

| 方法 | PESQ | STOI | E-STOI | SI-SNR (dB) |
|---|---:|---:|---:|---:|
| Noisy | 1.514 | 0.825 | 0.694 | 4.567 |
| Proposed | 2.359 | 0.926 | 0.847 | 11.10 |

最初本地核对阶段没有该训练/测试集，也没有进行完整训练；随机权重和合成 WAV 只用于检查代码闭环。用户后来提供的服务器日志显示候选已完成到第30轮训练，但日志中的验证 loss 不能代替上述增强指标，也不能据此认定使用了论文同一测试集。MACs 6.42 G/s 尚未用相同计数口径核验。

历史旧评估默认根据 target 调整估计增益，输出 PESQ、E-STOI 百分比和 BSS-eval SDR；这些数值不能直接替代论文四项指标。当前统一的 `evaluate_light.py` 同时计算 PESQ、STOI、ESTOI、去均值 SI-SNR 和 BSS-eval SDR，以及相同指标的 noisy baseline。SDR 使用 `mir_eval` 的 BSS-eval 定义（512抽头失真滤波器），不等于 SI-SNR。

当前 CSV 的 `estoi_pct` / `estoi_mix_pct` 为百分数；`enhanced_estoi` / `noisy_estoi`、STOI 字段及 JSON 的 `mean.enhanced` / `mean.noisy` 中 `estoi` / `stoi` 保持原始小数值。与论文表格比较时使用同一单位，并分别识别 `si_snr_db` 和 `sdr_db`。

当前默认 `--match-estimate-level no`，评估原始模型输出。显式设置 `yes` 才会在推理后使用干净参考计算一个受 `--max-level-gain-db` 限制的缩放系数；这属于依赖参考的后处理，报告会标注，不能与未启用时的结果混为同一评估协议。任何模式都不会在模型输入前对波形做峰值、RMS 或标准差归一化。

## 运行和验证

Windows 当前环境没有 PATH 中的 `python`，使用工作区虚拟环境：

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_light_model.py tests/test_light_pipeline.py tests/test_light_paper_evaluation.py -q -ra
& .\.venv\Scripts\python.exe audit_light.py --output output/light_audit.json
& .\.venv\Scripts\python.exe audit_light.py --require-exact --output output/light_audit.json
```

最后一条应返回非零退出码 2，因为仍有未解决的论文复现条件；这与实现测试是否通过是两回事。普通审计命令的退出码 0 只表示审计执行成功，报告中的 `exact_reproduction_verified` 仍为 false。

当前已有 `tests/test_cts_*.py` 对应的 CTS 源文件在本次任务前已被删除，本次不会恢复；请使用上面的定向测试命令，不要把 CTS 导入失败混入 light 验证结果。

启动修改后的模型时继续使用原训练入口和原 metadata。请从头训练，并使用新目录，避免默认自动恢复旧模型：

```powershell
& .\.venv\Scripts\python.exe train_light.py --train-dir "你的training_set目录" --val-dir "你的validation_set目录" --resume no --checkpoint-dir ./checkpoints_fcae --best-dir ./bestmodels_fcae --log-dir ./logs_fcae --learning-rate 0.001 --lr-reduce-metric val --train-lr-patience 2 --train-lr-factor 0.5 --train-lr-min-delta 0
```

这条命令不是作者未公布训练配方的替代证明；其余值继续使用原脚本默认配置。旧 checkpoint 包含不同的 head、DFSMN 和 projection，不能直接续训；保持 `strict=True` 加载，不用 `strict=False` 掩盖未加载权重。

统一评估入口（下面使用当前已训练候选的 checkpoint）：

```powershell
& .\.venv\Scripts\python.exe evaluate_light.py --val-dir "你的development_test目录" --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt --estimate-dir ./estimate_set_cbam_flat_projection64 --max-samples 0 --save-samples yes --match-estimate-level no --save-csv ./logs_cbam_flat_projection64/metrics.csv --save-json ./logs_cbam_flat_projection64/metrics.json
```

前处理从 checkpoint 的 `args` 读取，CLI 前处理参数默认不覆盖；显式值必须与已保存值相同。候选 checkpoint 缺失必需身份/配置时拒绝加载；普通旧格式缺少前处理字段时要求显式给出。当前已训练候选使用16 kHz、512点 FFT、320点 periodic Hann 窗、160点帧移、centered reflect padding、`normalized=False`。压缩指数沿用 checkpoint（当前 `power=0.5`），公式与训练一致为 `Z * abs(Z).clamp_min(1e-8) ** (power - 1)`，零谱保持零；逆变换对应处理接近零的分支。取 checkpoint 指定的前 `num_mics` 个输入通道；当前候选为8通道。多通道干净参考取 checkpoint 的 `target_ref_mic`。

评估逐条处理完整语音，不沿用训练阶段随机裁剪或 batch 补零；`--max-samples 0` 表示全部，正整数表示前 N 条。默认保存单通道参考麦克风 mixture、estimate、target 为 FLOAT WAV，保持幅度且不做写盘削波；输入原始多通道文件不会改写。始终生成输出目录下的 `metadata.csv`，默认汇总为该目录的 `summary.json`，额外结果副本可用 `--save-csv` / `--save-json` 指定。`--save-samples no` 可只计算指标。

单条指标失败会中止并写入失败 JSON，绝不把失败样本悄悄从均值中去掉。JSON 中保留 checkpoint/metadata 哈希、工具版本、指标定义和增益匹配标志；元数据与路径清单摘要不能证明音频内容与论文数据相同。

验证覆盖和实际执行结果见本文件后续的验证记录，以及 `output/light_audit.json`。

## 最初公式修正阶段的验证结果（历史记录）

合并运行：**36 passed，1 skipped，1 xfailed**，共 38 个测试用例。

- 数值参考测试：用显式复数算术、循环卷积和 NumPy FFT 核对式 (2)、(5)、(11) 及傅里叶路径；独立核对引用 [16] 的 DFSMN 递推。
- 结构与学习：五级频率尺寸严格往返，三次 DFSMN 同参数并传递记忆；完整默认模型所有参数梯度存在且有限，主要模块均获得非零梯度。
- 固定合成 batch、默认 64 通道模型、原始 loss、Adam 10 步：loss 从 **0.447233 降到 0.077332**。这仅排查基本可训练性，不表示真实数据收敛或达到论文效果。
- CPU bfloat16 autocast 的奇数频点 FFT 和反向通过。CUDA float16 测试因 `torch.cuda.is_available() == False` 跳过，**GPU AMP/DDP 未验证**。
- 原 metadata 加载、参考通道选择、裁剪、collate、零输入 STFT、0.5/1.0 压缩重建、完整累积窗口训练和验证、带 `module.` 前缀 checkpoint 往返通过。
- 四指标计算、SI-SNR 的解析投影和尺度/偏置检查、原始 noisy baseline、坏输入中止、checkpoint 前处理冲突拒绝通过。
- 唯一 xfail 明确复现原脚本尾窗累积错误：只有 1 个 batch、累积设为 4 的简单 SGD 示例，期望更新为 2.0，实际为 0.5。它是已知保留问题，**不是通过项**。
- 原 `train_light.py` CLI 用默认 64 通道模型，在一条合成 8 通道 WAV 上跑完 epoch 1、保存 checkpoint，再启动进程成功恢复 epoch 2，并使用产生的 checkpoint 跑通新四指标评估 CLI。此检查的训练/验证使用同一合成文件，裁剪 0.1 秒、累积 1，仅验证执行闭环，不能用于泛化或性能评价。
- 原训练/评估文件 SHA256 与任务开始时一致；原 loss 函数源码逐字一致；`git diff --check` 通过。
- 普通审计写出报告；`--require-exact` 实测返回 2；所有已披露默认配置检查匹配，但参数总量和消融参数差异仍未匹配。

在该阶段，完整基准训练、真实测试集分数、GPU 数值稳定性、多卡训练和论文 MACs 均未完成。测试只缩小已检查实现发生错误的可能性，不能证明不存在其他复现错误。

测试环境：Python/平台见 `output/light_audit.json`；PyTorch 2.14.0+cpu，NumPy 2.5.3，SciPy 1.18.1，SoundFile 0.14.0，PESQ 0.0.4，pystoi 0.4.1，mir_eval 0.8.2，pytest 9.1.1。`requirements-light.txt` 列出依赖而未声称它们是作者版本。

## 后续结构排查：24 个候选已实际执行

当时新增 `light_structure_variants.py` 和 `analyze_light_structure.py`。在该阶段，默认模型文件、训练文件及旧评估文件保持原 SHA256；默认候选同种子下的全部权重、state_dict 键和输出与原模型逐位相同。

限定的假设空间为：CA groups=1/2/4，skip fusion=cat/add，五层 skip attention 独立/共享，DFSMN 按频率独立/按 C×F 展平，共 24 种。每个候选均实例化完整模型和真正移除 skip attention 的消融模型，共执行 48 次 100 帧前向。统计唯一参数、各层形状、共享调用及两种范围明确的矩阵 MAC 估算。

**执行结果：24 种均运行成功，0 种同时满足论文完整模型和消融模型的参数区间；甚至所有无跳接模型都没有落入 725,000–734,999。** 因此本轮没有选择候选替换默认模型，也没有启动候选训练；未创建没有合格候选可用的训练适配入口。

最接近的 `ca_groups=2 / cat / 独立skip / per_frequency` 为 739,234 / 717,944 参数。完整模型能四舍五入为 0.74M，但消融模型为 0.72M，距离 0.73M 允许区间下界仍少 7,056。不能仅以完整参数量吻合宣布解决差异。

本轮选项是诊断假设，不是作者配置：分组 CA 不重排 avg/max 拼接，部分输出因此只看一种池化；跨 skip 权重共享没有正文依据；展平版本为 448→64→448，其中 64 是投影维，无法据此证明满足论文的“64 hidden units”。这些限制随每个候选写入 JSON。

默认模型在 100 帧下 Conv/Linear/LSTM 矩阵 MAC 小计为 4.679433280G；把转置卷积按输出网格的稠密代理统计时为 6.366870592G，后者包含上采样零位上的假想乘法。两者都没有计入 FFT、手写 memory、BN、池化和其他逐元素操作，不能当成作者 6.42 G/s 的同口径认证。100 帧对应 10 ms 帧移下的名义 1 秒，现有 centered STFT 对真实 1 秒波形通常输出 101 帧。

输出文件：

- `output/baseline.json`：本轮开始时的默认基线。
- `output/structure_candidates.csv`：24 个候选的计数、通过状态和具体淘汰原因。
- `output/structure_candidates.json`：逐层参数、形状、共享关系、MAC 范围、假设和源码哈希。
- `output/structure_screen_summary.md`：简明结论。
- `output/structure_verification.json`：本轮验证记录。

可复跑命令：

```powershell
& .\.venv\Scripts\python.exe analyze_light_structure.py --output-dir output --frames 100
& .\.venv\Scripts\python.exe analyze_light_structure.py --output-dir output --frames 100 --require-match
& .\.venv\Scripts\python.exe -m pytest tests/test_light_model.py tests/test_light_pipeline.py tests/test_light_paper_evaluation.py tests/test_light_structure.py tests/test_light_structure_analysis.py -q -ra
```

普通分析返回 0 表示执行成功；`--require-match` 在没有双约束合格候选时返回 2。发生候选运行异常则返回 1，不能把运行失败当成正常淘汰。

本轮合并测试为 **100 passed、1 skipped、1 xfailed**，新增 64 项通过。新增测试包括 48 个完整/消融模型的独立参数公式与前向、默认等价性、展平索引数值 oracle、代表配置全部参数反向、共享参数去重、计数区间边界、双约束淘汰规则、MAC 独立核算与完整分析 CLI。原 CUDA 跳过和训练尾窗 xfail 保留。

本轮只排除了上述 24 个具体组合；没有证明其他结构不存在，也没有证明论文表格错误。继续修改默认模型需要新的结构证据，不能把无依据的参数搜索当作论文还原。

## 进一步进展：引用实现与第三组消融约束

后续已核查 CBAM、原 EaBNet TCN 和 FRCRN 公开实现，新增五个有明确来源标签的候选。`cbam_flat_projection64` 的完整/去skip参数为 **735,070 / 731,680**，按原 TCN 结构替换推算为 **2,455,710**，三项都可舍入为论文的0.74M/0.73M/2.46M。它已完成10步合成优化，并通过原训练主函数的保存、独立进程续训和四指标评估。

这是计数匹配候选：其 CBAM 算子仍偏离目标式(11)，且64指投影宽度、不能证明符合原文隐藏维度。没有替换默认模型，也没有宣称作者等价或真实基准达标。训练使用 `train_light_candidate.py`，评估现统一使用 `evaluate_light.py`；保留原训练与数据接口，以候选身份和模型实现源码哈希管理断点。具体证据、结果及可直接运行的命令见 [LIGHT_STRUCTURE_EVIDENCE.md](LIGHT_STRUCTURE_EVIDENCE.md)。
