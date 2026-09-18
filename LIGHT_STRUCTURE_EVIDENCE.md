# 结构差异的进一步定位与可运行候选

已找到一个同时符合论文三组参数量的候选，并跑通原训练主函数、保存、独立进程续训和四指标评估。不过，它仍有两项与目标文字不一致的解释，不能将它标为作者结构。原 `EaBNet_light.py`、`train_light.py`、`evaluate_light.py` 保持本轮开始时的内容。

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

独立四指标评估会从 checkpoint 自动恢复候选身份，不需要人工重新选择结构：

```powershell
& .\.venv\Scripts\python.exe evaluate_light_candidate.py --val-dir "你的development_test目录" --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt --save-csv ./logs_cbam_flat_projection64/paper_metrics.csv --save-json ./logs_cbam_flat_projection64/paper_metrics.json
```

checkpoint 记录候选名、结构来源、公式差异、实现源码哈希与原训练 args。错误候选、源码变动、缺失必要配置会拒绝加载；续训前还核对前处理和模型配置。权重始终严格加载，支持原来的 `module.` 前缀。不同结构不会因为参数形状恰好相同而被当成同一模型。

### 分别指定带噪输入和干净目标的位置

评估时可以用 `--mixture-path` 和 `--target-path` 替代 `--val-dir`，无需创建或修改 metadata.csv。前者是带噪多通道语音，后者是与它对应的干净参考语音。两个参数可以都是单个 WAV/FLAC 文件，也可以都是目录；目录递归按相同相对路径和文件名主体配对，忽略扩展名。例如 `noisy/speaker1/001.wav` 对应 `clean/speaker1/001.flac`。缺少配对、重复键、大小写冲突会在加载模型前报错，不会按目录枚举顺序强行配对。

Linux 服务器示例（替换前两个路径）：

```bash
MIXTURE_PATH="/实际路径/noisy"
TARGET_PATH="/实际路径/clean"
mkdir -p ./logs_cbam_flat_projection64
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  evaluate_light_candidate.py \
  --candidate cbam_flat_projection64 \
  --mixture-path "$MIXTURE_PATH" \
  --target-path "$TARGET_PATH" \
  --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt \
  --device cuda \
  --max-samples 0 \
  --save-csv ./logs_cbam_flat_projection64/custom_metrics.csv \
  --save-json ./logs_cbam_flat_projection64/custom_metrics.json \
  > ./logs_cbam_flat_projection64/custom_eval.log 2>&1 &
```

`--mixture-dir` / `--target-dir` 是这两个路径参数的别名。目录内若是 `001_mix.wav` 与 `001_clean.wav`，再加 `--mixture-suffix _mix --target-suffix _clean`；后缀不包括扩展名，指定后要求各目录内每个音频文件都符合该后缀。单文件模式直接指定两个文件，允许文件名不同，不使用后缀参数。不规则配对仍可用原 `--val-dir` / metadata.csv 明确指定。

当前候选要求输入至少8通道，取前8通道；输入和参考均为16 kHz，多通道参考沿用 checkpoint 的 `target_ref_mic`。参考必须与输入来自同一句且时间对齐；评估沿用共同长度截断，不会自动校正延时，也不使用参考做增益匹配。评估逐条处理完整语音，保留原四指标与带噪基线。CSV 写出每对实际路径；JSON 的 `input_source` 记录模式、配对规则和路径清单摘要，`selected_pairs_sha256` 记录实际选中清单的摘要（两者均不是音频内容哈希）。输出路径不可覆盖输入音频、checkpoint 或当前 metadata。

这些选项仅扩展评估输入；原训练脚本、模型实现与 checkpoint 身份校验不变，已有训练权重可直接评估。

这五个名称固定对应8麦、64通道/embedding、3层共享记忆、memory设置20、BN和MIMO LSTM头；新入口会拒绝改变这些结构维度，避免记录的来源说明与真实模型不一致。自定义结构继续使用原入口。

候选间比较需要使用相同数据划分和预算，只用验证集作选择；不能选出测试分数最接近论文的结构，再反推它就是作者配置。

## 已完成与仍缺失

已完成源码核查、五候选前向/反向、独立数值公式、三组参数约束、10步合成优化，以及合成音频上的原训练→保存→进程重启续训→四指标评估。完整实验产物为 `output/reference_candidates.json`、CSV 和 `reference_screen_summary.md`；验证记录在 `output/reference_verification.json`。

合并七个 light 测试文件的最终结果为 **155 passed、1 skipped、1 xfailed**。其中跳过项是 CUDA 不可用；xfail 是原脚本尾窗累积问题，不能计入通过项。新增测试还核对错误结构断点拒绝、源码和证据元数据一致性、配置字段不可省略及兼容 best checkpoint 的覆盖保护。

当前环境仅 CPU，未提供论文训练与测试数据，尚未执行真实完整训练、GPU AMP/DDP 或论文性能基准。原训练尾窗梯度累积等已知保留问题仍见 `LIGHT_REPRODUCTION.md`。它们不会因为新增候选自动消失。

[IEEE 官方报告页](https://resourcecenter.ieee.org/conferences/icassp-2023/spsicassp23vid0845)和[UWA 论文记录](https://research-repository.uwa.edu.au/en/publications/a-lightweight-fourier-convolutional-attention-encoder-for-multi-c/)都确认0.74M；本轮未找到可确认的目标作者源码或补充配置。先前检索片段的0.72M没有核实为真实版本，未用于改动约束。没有联系作者、购买或绕过受限材料。
