# 结构差异的进一步定位与可运行候选

已找到一个同时符合论文三组参数量的候选，并跑通原训练主函数、保存、独立进程续训和指标评估。不过，它仍有两项与目标文字不一致的解释，不能将它标为作者结构。当前评估入口统一为 `evaluate_light.py`，保留此前的输入路径功能并补齐 SDR、音频导出和可选幅度匹配；这次整合不修改当前模型实现或训练逻辑。

## 新证据如何缩小范围

1. 引用[9]的[原 EaBNet 作者代码](https://github.com/Andong-Li-speech/EaBNet/blob/main/EaBNet.py)把 C×F 展平成时序特征。按当前 512 点 FFT 的瓶颈几何，输入宽度为 64×7=448。保留原代码 3 组、每组 6 个 TCM，64 内部通道、时间核5，其 TCN 有 **1,779,840** 参数。
2. 论文表2的 2.46M（TCN）与 0.74M（DFSMN）要求二者差额约 1.72M。逐频点 DFSMN 的替换差额为 1,770,176，不符；展平投影64版本差额为 **1,720,640**，符合。这是以保留原 TCN 层数和当前频率几何为条件的推断，不是目标论文补充材料。
3. 引用[13]的 [CBAM 作者固定版本](https://github.com/Jongchan/attention-module/blob/459efad0e05ee7dde50c41ca10a3d0800bc3792a/MODELS/cbam.py)使用共享完整 MLP，64→4→64、ReLU、两个有 bias 的 Linear。平均和最大池化各调用完整 MLP 再相加，每个 CA 有 **580** 参数。只替换 CA，保留目标式(5)的 SA。
4. 引用[17]的 [FRCRN 仓库](https://github.com/alibabasglab/FRCRN)指向其[维护中的记忆单元源码](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/clearvoice/models/frcrn_se/complex_nn.py)。它区分隐藏64与投影448，构成 448→64（ReLU）→448→记忆与输入残差，有 **66,368** 参数；20抽头包含当前及过去19帧。其公开整网已与 FRCRN 论文有变动，这里只实现该单元作为对照。

原 TCN 源码 SHA256：`fcca4f828765a82e2c4605b1ef1fcc630cfb30f97c70e45449eac6f62fe7e0e8`。CBAM 源码 SHA256：`6a2115f71541c77ab09c666e4cf32ab9928520f9c2bba9c1ceab8edc8fdd8ac5`。FRCRN 单元源码 SHA256：`3dcda8502c6d588493a59dcb0910624a088be3e1c8b82b9d4b9408e1c5f3b5cb`。来源和公式差异也随候选写入审计报告及训练 checkpoint。

## 五个候选的实际结果

完整/去跳接模型均实际构建并执行100帧前向；TCN列是上述已核验计数的替换推算，没有执行该 TCN 消融训练。

| 候选 ID | 完整参数 | 去 skip 参数 | TCN 替换推算 | 三项表格舍入值均符合 |
|---|---:|---:|---:|---|
| literal_per_frequency | 800,674 | 758,904 | 2,570,850 | 否 |
| literal_flat_projection64 | 850,210 | 808,440 | 2,570,850 | 否 |
| cbam_per_frequency | 685,534 | 682,144 | 2,455,710 | 否 |
| cbam_flat_projection64 | **735,070** | **731,680** | **2,455,710** | **是** |
| cbam_flat_hidden64 | 742,238 | 738,848 | 2,455,710 | 否 |

`cbam_flat_projection64` 是值得实验的计数匹配候选，仍有两项关键差异：

- CBAM 是两支共享非线性 MLP 后相加；目标论文式(11)写的是池化拼接后单个 Conv，二者不等价。
- 该 DFSMN 的64是投影宽度，输出隐藏激活宽448；不能证明符合目标论文“64 hidden units”的本意。显式采用隐藏64的 FRCRN 对照，在无 skip 参数上又不符合表格。

因此没有候选同时满足当前字面公式解释、隐藏维度解释及三组参数表。数字相符缩小了排查范围，但没有消除作者结构歧义。

五个模型都用原 loss、Adam 1e-3 在固定合成谱 batch 上运行10步；`cbam_flat_projection64` 的 loss 为 **0.424305→0.070809**。所有参数梯度存在且有限。这只是基本可训练性检查，不是实际增强效果或真实数据收敛证据。

## 直接运行

检查五个候选及三组计数：

```powershell
& .\.venv\Scripts\python.exe analyze_light_reference.py --output-dir output --frames 100 --steps 10
```

在已有数据上训练计数匹配候选，参数沿用原训练脚本；把目录替换成现有 metadata.csv 所在目录即可：

```powershell
& .\.venv\Scripts\python.exe train_light_candidate.py --candidate cbam_flat_projection64 --train-dir "你的training_set目录" --val-dir "你的validation_set目录" --resume no --learning-rate 0.001 --lr-reduce-metric val --train-lr-patience 2 --train-lr-factor 0.5 --train-lr-min-delta 0
```

默认自动使用 `checkpoints_cbam_flat_projection64`、`bestmodels_cbam_flat_projection64` 和 `logs_cbam_flat_projection64`，也可以继续使用原来的目录参数指定独立目录。续训使用相同命令与配置、改为 `--resume yes`。适配入口只临时替换模型构造函数，调用原 `train_light.main()`；没有复制或修改训练循环、优化器步骤、loss、STFT、Dataset、collate 和 metadata 接口。

统一评估入口会从 checkpoint 自动恢复候选身份，不需要人工重新选择结构；现有已训练权重无需重训。默认数据目录为 `./validation_set`，默认 checkpoint 为 `./bestmodels_cbam_flat_projection64/best_model.pt`；可显式覆盖路径：

```powershell
& .\.venv\Scripts\python.exe evaluate_light.py --val-dir "你的development_test目录" --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt --estimate-dir ./estimate_set_cbam_flat_projection64 --max-samples 0 --save-samples yes --match-estimate-level no --save-csv ./logs_cbam_flat_projection64/metrics.csv --save-json ./logs_cbam_flat_projection64/metrics.json
```

checkpoint 记录候选名、结构来源、公式差异、实现源码哈希与原训练 args。错误候选、源码变动、缺失必要配置会拒绝加载；续训前还核对前处理和模型配置。权重始终严格加载，支持原来的 `module.` 前缀。不同结构不会因为参数形状恰好相同而被当成同一模型。

### 与当前训练一致的前处理及评估输出

STFT 参数与幅度压缩从 checkpoint 的 `args` 读取，命令行默认不覆盖；显式指定冲突参数会报错。当前候选配置为16 kHz、`n_fft=512`、`win_length=320`、`hop_length=160`、periodic Hann 窗、`center=True`、reflect padding、`normalized=False`、`power=0.5`。压缩公式与训练一致为 `Z * abs(Z).clamp_min(1e-8) ** (power - 1)`；逆变换解开同一压缩（包括接近零的分支）并使用相同窗口/STFT 参数。波形直接使用读取得到的幅度，不增加峰值、RMS 或标准差归一化。

输入按 checkpoint 的麦克风数取前 `num_mics` 个通道（当前为8），干净目标若为多通道则取 checkpoint 的 `target_ref_mic`。`--mixture-ref-mic` 只选择 noisy 对照与导出的 mixture 通道，不改变送入模型的多通道输入。完整语音逐条推理，不使用训练时的随机裁剪；`--max-samples 0` 评估全部，正整数评估前 N 条。

逐样本计算增强语音和 noisy 的 PESQ（16 kHz wideband）、ESTOI、BSS-eval SDR；同时保留 STOI 与去均值 SI-SNR 供论文对照。SDR 使用 `mir_eval` BSS-eval 的512抽头失真滤波器定义，不能把它作为论文 SI-SNR。

CSV 的 `estoi_pct` / `estoi_mix_pct` 为百分数；`enhanced_estoi` / `noisy_estoi`、STOI 字段与 JSON 均值中的 `estoi` / `stoi` 保持原始小数值，比较时不要混淆单位。JSON 的 `mean.enhanced` 与 `mean.noisy` 分别含 `pesq`、`stoi`、`estoi`、`si_snr_db`、`sdr_db`；后两者单位为 dB。

`--match-estimate-level no` 为默认值。设置 `yes` 时，在模型推理后利用干净参考估计单个缩放系数，再计算增强指标并写出增强音频；`--max-level-gain-db 20` 为默认幅度调整上限。该处理依赖参考，报告会标记，比较不同实验时必须使用同一设置。

`--estimate-dir` 默认 `./estimate_set_cbam_flat_projection64`。默认 `--save-samples yes` 保存每条单通道参考麦克风 mixture、estimate、target 为 FLOAT WAV，保留幅度、不按整数 WAV 范围削波；不保存全部8个 mixture 通道。`--save-samples no` 只关闭 WAV 导出。输出目录始终包含逐样本 `metadata.csv`，汇总默认写到 `summary.json`；`--save-csv` 可另外写一份逐样本 CSV，`--save-json` 可指定汇总位置。CSV 的 `mixture_path` / `target_path` 保留原始文件位置，导出位置另记为 `mixture_saved_path` / `estimate_saved_path` / `target_saved_path`。输入 metadata 和原始音频不会被覆盖；目录配对模式下，输出目录也不能置于任一输入音频目录之内。

### 分别指定带噪输入和干净目标的位置

评估时可以用 `--mixture-path` 和 `--target-path` 替代 `--val-dir`，无需创建或修改 metadata.csv。前者是带噪多通道语音，后者是与它对应的干净参考语音。两个参数可以都是单个 WAV/FLAC 文件，也可以都是目录；目录递归按相同相对路径和文件名主体配对，忽略扩展名。例如 `noisy/speaker1/001.wav` 对应 `clean/speaker1/001.flac`。缺少配对、重复键、大小写冲突会在加载模型前报错，不会按目录枚举顺序强行配对。

Linux 服务器示例（替换前两个路径）：

```bash
MIXTURE_PATH="/实际路径/noisy"
TARGET_PATH="/实际路径/clean"
mkdir -p ./logs_cbam_flat_projection64
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  evaluate_light.py \
  --candidate cbam_flat_projection64 \
  --mixture-path "$MIXTURE_PATH" \
  --target-path "$TARGET_PATH" \
  --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt \
  --device cuda \
  --max-samples 0 \
  --estimate-dir ./estimate_set_cbam_flat_projection64 \
  --save-samples yes \
  --match-estimate-level no \
  --save-csv ./logs_cbam_flat_projection64/custom_metrics.csv \
  --save-json ./logs_cbam_flat_projection64/custom_metrics.json \
  > ./logs_cbam_flat_projection64/custom_eval.log 2>&1 &
```

`--mixture-dir` / `--target-dir` 是这两个路径参数的别名。目录内若是 `001_mix.wav` 与 `001_clean.wav`，再加 `--mixture-suffix _mix --target-suffix _clean`；后缀不包括扩展名，指定后要求各目录内每个音频文件都符合该后缀。单文件模式直接指定两个文件，允许文件名不同，不使用后缀参数。不规则配对仍可用原 `--val-dir` / metadata.csv 明确指定。

当前候选要求输入至少8通道，取前8通道；输入和参考均为16 kHz，多通道参考沿用 checkpoint 的 `target_ref_mic`。参考必须与输入来自同一句且时间对齐；评估沿用共同长度截断，不会自动校正延时。默认不使用参考做增益匹配。CSV 写出每对实际路径；JSON 的 `input_source` 记录模式、配对规则和路径清单摘要，`selected_pairs_sha256` 记录实际选中清单的摘要（两者均不是音频内容哈希）。输出路径不可覆盖输入音频、checkpoint 或当前 metadata。

当前只保留 `evaluate_light.py` 作为评估启动脚本，冗余的两个独立评估入口与旧备份已移除；`light_eval_inputs.py` 是统一入口调用的路径解析模块，不是额外验证脚本。原训练脚本、模型实现与 checkpoint 身份校验不因这次评估整合而改变，已有训练权重可直接评估。

本次统一评估入口的 CPU 定向回归共88项通过、1项已有预期失败（原训练脚本尾窗梯度累积）。检查包括真实候选模型的独立进程训练、续训与五指标评估；正常及极小幅度 STFT 与训练逐元素相等；WAV 幅度保持、目标/麦克风通道选择、参考增益开关、输出保护与前 N 条限制。这些检查不替代服务器真实测试集的效果评估。

这五个名称固定对应8麦、64通道/embedding、3层共享记忆、memory设置20、BN和MIMO LSTM头；新入口会拒绝改变这些结构维度，避免记录的来源说明与真实模型不一致。自定义结构继续使用原入口。

候选间比较需要使用相同数据划分和预算，只用验证集作选择；不能选出测试分数最接近论文的结构，再反推它就是作者配置。

## 已完成与仍缺失

已完成源码核查、五候选前向/反向、独立数值公式、三组参数约束、10步合成优化，以及合成音频上的原训练→保存→进程重启续训→四指标评估。完整实验产物为 `output/reference_candidates.json`、CSV 和 `reference_screen_summary.md`；验证记录在 `output/reference_verification.json`。

候选结构阶段合并七个 light 测试文件的历史结果为 **155 passed、1 skipped、1 xfailed**，不代表本次评估整合后的测试数量。其中跳过项是 CUDA 不可用；xfail 是原脚本尾窗累积问题，不能计入通过项。该阶段新增测试还核对错误结构断点拒绝、源码和证据元数据一致性、配置字段不可省略及兼容 best checkpoint 的覆盖保护。

当前本地环境仅 CPU，未提供论文训练与测试数据，不能在此核验 GPU AMP/DDP 或论文性能基准。用户提供的服务器日志显示 `cbam_flat_projection64` 双卡训练已完成到第30轮，最后一轮验证 loss 为0.023551；这是训练日志记录，尚不能代替真实测试集的增强指标或论文复现认证。原训练尾窗梯度累积等已知保留问题仍见 `LIGHT_REPRODUCTION.md`，它们不会因为新增候选自动消失。

[IEEE 官方报告页](https://resourcecenter.ieee.org/conferences/icassp-2023/spsicassp23vid0845)和[UWA 论文记录](https://research-repository.uwa.edu.au/en/publications/a-lightweight-fourier-convolutional-attention-encoder-for-multi-c/)都确认0.74M；本轮未找到可确认的目标作者源码或补充配置。先前检索片段的0.72M没有核实为真实版本，未用于改动约束。没有联系作者、购买或绕过受限材料。

## 同一数据集 PESQ 超过3.4的性能目标

用户提供的600条评估截图中，当前候选的 PESQ 为2.689355，原始 mixture 为1.088096；用户确认对照使用相同训练集，3.4来自同一批测试数据、同一参考和评估方式下的成绩。差距约0.711，不能由论文表1的2.359是否达到来替代判断，也不能保证增加轮数或修改一项损失就能消除差距。按用户要求，不以取得对照模型名称作为继续工作的前提。

当前 `best_model.pt` 按验证集的压缩复数/幅度 MSE 保存，最低频谱损失不保证最高 PESQ。统一评估入口新增 `--checkpoint-dir`，可在**独立验证集**上比较同一实验目录中的 `model_epoch_*.pt`、`checkpoint_latest.pt` 和 `best_model.pt`，按平均 PESQ 排序，不改训练循环、不改权重，也不复制或覆盖训练目录里的最佳模型。600条最终测试音频用于确认最终结果，不参与 checkpoint 或损失超参数选择。

```bash
mkdir -p ./logs_cbam_flat_projection64
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  evaluate_light.py \
  --checkpoint-dir ./checkpoints_cbam_flat_projection64 \
  --candidate cbam_flat_projection64 \
  --val-dir /data/ssd1/jinrui.yang/validation_set \
  --device cuda \
  --max-samples 0 \
  --save-samples no \
  --match-estimate-level no \
  --estimate-dir ./estimate_set_cbam_flat_projection64_selection \
  --save-csv ./logs_cbam_flat_projection64/checkpoint_ranking.csv \
  --save-json ./logs_cbam_flat_projection64/checkpoint_ranking.json \
  > ./logs_cbam_flat_projection64/checkpoint_selection.log 2>&1 &
```

每个 checkpoint 的逐样本结果与汇总分开放在 `--estimate-dir` 下，顶层 JSON 的 `best_checkpoint` 给出验证集 PESQ 最高的权重路径。参与比较的音频清单、前处理和指标协议必须一致；任何 checkpoint 评估失败时，不给出最佳模型结论。该比较需分别推理每个 checkpoint，耗时会随 checkpoint 数量增加。初步排错可用 `--max-samples`，最终选择应使用完整验证集，避免前 N 条的排列偏差。

后续按证据推进：

1. 先保留当前第30轮权重，完成验证集 PESQ 比较，确认当前所谓“最佳”是否实际最优。
2. 服务器日志显示第30轮仍更新最低验证 loss，尚不能断言已经收敛。可沿用上次实际双卡配置，将 `--num-epochs` 改为60、`--save-every` 改为1，并保持 `--resume yes --resume-reset-lr no`。这表示从第31轮继续到第60轮，恢复优化器及学习率；不是从头训练，也不把学习率重置到1e-3。是否保留续训结果由独立验证集 PESQ 决定。
3. 如果验证集 PESQ 平台持续存在，再检查训练集与目标条件的噪声、SNR、混响、目标定义和阵列几何；训练/验证都偏低与仅验证偏低需要不同处理。新增样本须使用独立语音和噪声片段，避免测试数据泄漏。
4. 确认数据与训练预算后，再把重建波形损失或感知指标代理训练作为独立实验，与原始 MSE 配方对照。此类改动属于性能改进，不自动等同于论文复现；损失权重、结构容量与能否达到3.4都必须通过实测确定。

全部比较保持 `--match-estimate-level no` 和同一 PESQ 模式；参考增益匹配只改变评估协议，不能作为模型本身达到目标的依据。

该选模扩展与原单模型评估的定向回归共109项通过；另有2项真实符号链接保护测试因本地 Windows 不允许创建符号链接而跳过。检查了 checkpoint 排序、协议/样本清单一致性、失败中止、输入覆盖保护、真实候选权重的独立进程五指标评估。原评估文件的12条注释及6段文档字符串逐项保持，训练和模型源码未改。

## 第31至40轮结果与固定训练小集诊断

2026-09-22收到的 `continue_31_40.log` 显示从第30轮成功续训至第40轮，仍为53600条训练、2680条验证、双卡DDP、每卡batch 4，非有限loss跳过次数为0。学习率在第32轮结束时由0.00025降到0.000125，第37轮结束时降到0.0000625。第40轮train loss为0.023099、val loss为0.023354；最低val loss为第39轮的0.023336。

同批600条babble对照音频的21个已保存checkpoint全部完成评估，输入路径清单摘要和noisy指标一致。第30轮PESQ为2.689355，第40轮为2.721410，十轮增加0.032054，距离3.4仍差0.678590。按训练val loss保存的 `best_model.pt` 对应第39轮，PESQ为2.714079；第40轮在这批音频上略高。因此选模差异只解释约0.007331，不能解释与3.4的完整差距。第40轮仍有改善，不能断言绝对收敛，也不能推断继续增加轮次必然达标。

这600条音频已经参与多轮比较，是当前诊断对照集；JSON中通用的 `selection_scope` 字样不会使它自动成为独立验证集。最终泛化报告仍需要未参与调整的评估数据。

新增 `diagnose_light_overfit.py` 负责在独立目录完成固定训练小集的拟合实验：从原训练metadata确定性抽取8条有效语音，各截取固定6秒；保持原幅度、麦克风顺序、目标通道、采样率和训练前端，保存FLOAT WAV及来源记录。短音频或静音片段的排除会记入清单，不通过补零凑足长度。拟合前后都在这同一批固定片段上，以eval模式计算原频谱损失，并复用 `evaluate_light.py` 计算五项增强/带噪指标。

诊断复制第40轮完整模型状态，包括BN运行统计，建立独立的epoch 0起点与全新Adam状态。原始checkpoint不改。原 `train_light_candidate.py` 执行全部训练：单卡、FP32、batch 2、梯度累积1、固定学习率0.0001、禁用plateau、禁用再次裁剪，200轮对应800次更新。同一小集同时传入train-dir和val-dir，只用于检查已见样本的拟合能力。BN统计仍按原模型更新，训练后同时报告最新和最低小集val loss权重的结果。

服务器命令如下，输出目录必须尚不存在；日志放在其外部：

```bash
mkdir -p ./logs_cbam_flat_projection64
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  diagnose_light_overfit.py \
  --checkpoint ./checkpoints_cbam_flat_projection64/model_epoch_40.pt \
  --train-dir /data/ssd1/jinrui.yang/training_set \
  --output-dir ./logs_cbam_flat_projection64/overfit8_epoch40 \
  --num-samples 8 \
  --segment-seconds 6 \
  --epochs 200 \
  --batch-size 2 \
  --learning-rate 0.0001 \
  --device cuda \
  > ./logs_cbam_flat_projection64/overfit8_epoch40.log 2>&1 &
```

完成后读取输出目录中的 `diagnostic_summary.json`，比较同一固定小集的初始/最终eval loss与PESQ。这8条裁剪音频的PESQ不能直接与原600条的2.721410比较。若小集损失和PESQ明显改善，表明当前实现能在该小集继续拟合，后续检查完整训练的优化、损失与泛化；若损失改善但PESQ不改善，再检查输出失真和损失目标；若小集也难拟合，继续检查梯度、BN统计、学习率和结构。一次成功或失败都不能单独证明完全复现正确或论文本身有问题。该工具没有改动原训练循环、模型、评估协议或已有注释，也没有在此阶段引入新的正式训练损失。

本次诊断工具的15项定向测试全部通过，包含真实CPU候选模型的固定片段基线评估、调用原训练入口、最新/最佳模型复评；逐张量确认初始模型与BN状态保持、Adam历史清空及实际更新次数；验证FLOAT幅度大于1时不削波、前8麦克风顺序与多通道目标选择、固定偏移及输入文件保护。子进程返回成功但未完成更新会被判失败。测试使用合成音频，不替代服务器上这8条真实训练语音的诊断，也没有在本地验证GPU表现。

2026-09-24收到的首份真实小集报告为 `status=failed`：只完成初始eval loss 0.0252563、PESQ 2.287191，`after_latest` 和 `after_best` 均为空，`steps=800`只是计划次数。补充训练日志确认第1轮第1批在第二层LSTM前向中CUDA显存不足，尚未完成第一次参数更新。报错时GPU 0总容量11.60 GiB、空闲232.75 MiB，本训练进程约5.03 GiB，另一个进程约6.07 GiB；不能仅凭显存量判断另一进程的身份。该报告不能用于推断拟合能力或与600条对照集的2.721410比较。恢复时先确认有足够空闲显存的GPU，使用新输出目录并保留原8条固定选择及训练配置。

诊断汇总现包含 `training_log` 路径、已取得的 `training_returncode`，以及失败时日志末尾最多100行/12000字符的 `training_log_tail`，保留真实子进程异常，避免只有包装层退出码。该错误记录修复不改变训练参数或原训练/模型文件。更新后的17项定向测试通过，包括模拟首批OOM的完整异常保留和日志摘录长度限制；实际GPU显存峰值仍需服务器确认。

后续 `overfit8_epoch40_v2` 的日志完成了第25轮（train loss 0.023397、val loss 0.023349），停在第26轮首批；用户的进程查询为空，说明当时已没有诊断或训练进程。日志末尾无Traceback，不能据此认定退出原因。JSON中的 `running` 只是最后写入的阶段，不代表当前进程仍存活。

同一入口新增 `--resume-run`，用于已停止的诊断：验证固定音频与metadata哈希、原始checkpoint、初始基线、最新权重/Adam步数及原训练配置，随后从最后完整保存轮次继续到原定轮数，保留原 `before` 结果和已训练的优化器状态。恢复命令只允许另外指定设备，不能借此改变样本数、batch、学习率或总轮数。日志追加，已完成的诊断不会重复训练；训练已完成而评估未完成时，可只补复评。损坏或不一致的checkpoint会明确拒绝，不会静默退回初始模型。报告写入改用同目录临时文件后原子替换，并用操作系统文件锁阻止同一新版诊断同时运行；Linux子进程继承该锁，父进程退出但子进程还在时仍会保持互斥。

```bash
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  diagnose_light_overfit.py \
  --resume-run ./logs_cbam_flat_projection64/overfit8_epoch40_v2 \
  --device cuda \
  > ./logs_cbam_flat_projection64/overfit8_epoch40_v2_resume.log 2>&1 &
```

本次若最新checkpoint保存于第25轮，则恢复100次已完成的Adam更新，从第26轮继续至第200轮，总计800次更新。第26轮未保存的更新会重做。原trainer未保存RNG状态，重启会重新设种子并排列batch，因此这里只承诺恢复保存的模型、BN、优化器和轮次，不声称与不中断训练逐位等价。新测试使用真实CPU子进程在第1轮后中断再恢复，检查Adam累计步数、基线/数据不变、日志追加、完成后不重复运行、配置篡改拒绝和运行锁。

恢复功能与原诊断流程的组合测试为27项全部通过，包含报告原子替换失败时保留上次完整JSON。原训练、模型和评估文件保持不变；服务器上从第25轮继续至第200轮的真实GPU诊断仍需执行。
