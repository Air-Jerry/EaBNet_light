# FCAE-Att-DFSMN 复现核对记录

## 当前结论

**这是按已披露公式修正、可测试的实现，尚未达到作者实现等价或论文结果复现认证。不能据此把训练不达标归因于论文本身。**

依据是用户提供的 ICASSP 2023 论文 *A Lightweight Fourier Convolutional Attention Encoder for Multi-Channel Speech Enhancement*，DOI [10.1109/ICASSP49357.2023.10095716](https://doi.org/10.1109/ICASSP49357.2023.10095716)。用户确认没有额外作者代码或配置。本次检索未找到可验证的目标论文官方实现；[原始 EaBNet 仓库](https://github.com/Andong-Li-speech/EaBNet)对应参考文献 [9]，不是这篇 FCAE 论文的实现。

默认模型实测 **800,674** 个可训练参数，论文表 1 为 **约 0.74M**，仍未匹配。论文还省略了损失函数、若干结构参数及精确数据生成配置。因此，无法同时保证“训练逻辑原样保留”与“整个实验唯一还原”。本记录明确区分论文直接披露、引用方法补充、实现选择和未验证事项。

## 修改范围和接口

- 修改 `EaBNet_light.py`，保留 `EaBNet` 构造调用及 `forward(inpt)` 接口：输入 `(B,T,257,M,2)`，输出 `(B,2,T,257)`。
- `train_light.py` 和 `evaluate_light.py` **逐字节保持原样**，包括数据加载、裁剪、STFT、压缩、优化器、AMP/DDP、梯度累积、学习率、日志、断点流程和旧评估行为。
- `com_mag_mse_loss` 原函数保持原样；没有将 RI 项从均值改成求和，也没有增加训练损失。
- 新增独立四指标评估 `evaluate_light_paper.py`，仍读取原 `metadata.csv` 接口。
- 新增针对 light 模型的测试及 `audit_light.py`，不修改、恢复或移除已有 CTS 文件状态。
- 不符合 512 点 FFT 的输入现在明确报错；解码尺寸错误不再用裁剪/补零悄悄掩盖。

原文件 SHA256：

| 文件 | 修改前和交付时均应为 |
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

本次没有该训练/测试集，也没有进行完整训练；没有产生可以与表 1 比较的增强指标。随机权重和合成 WAV 只用于检查代码闭环。MACs 6.42 G/s 尚未用相同计数口径核验。

旧 `evaluate_light.py` 默认根据 target 调整估计增益，输出的是 PESQ、E-STOI 百分比和 BSS-eval SDR；这些数值不能直接替代论文四项指标。独立 `evaluate_light_paper.py` 对原始估计计算 PESQ、STOI、E-STOI（均非百分比）和去均值 SI-SNR，并同时计算 noisy baseline；不使用 target 调整输出幅度。

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

独立评估入口：

```powershell
& .\.venv\Scripts\python.exe evaluate_light_paper.py --val-dir "你的development_test目录" --checkpoint ./bestmodels_fcae/best_model.pt --save-csv ./logs_fcae/paper_metrics.csv --save-json ./logs_fcae/paper_metrics.json
```

前处理从 checkpoint 的 `args` 读取，CLI 覆盖值必须与其相同；若旧格式缺字段，脚本要求显式给出。单条指标失败会中止并写入失败 JSON，绝不把失败样本悄悄从均值中去掉。JSON 中保留 checkpoint/metadata 哈希、工具版本和指标定义。

验证覆盖和实际执行结果见本文件后续的验证记录，以及 `output/light_audit.json`。

## 本次实际验证结果

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

完整基准训练、真实测试集分数、GPU 数值稳定性、多卡训练和论文 MACs 均未完成。测试只缩小已检查实现发生错误的可能性，不能证明不存在其他复现错误。

测试环境：Python/平台见 `output/light_audit.json`；PyTorch 2.14.0+cpu，NumPy 2.5.3，SciPy 1.18.1，SoundFile 0.14.0，PESQ 0.0.4，pystoi 0.4.1，mir_eval 0.8.2，pytest 9.1.1。`requirements-light.txt` 列出依赖而未声称它们是作者版本。
