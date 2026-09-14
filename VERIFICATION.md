# CTS-Net 实现验证记录

日期：2026-09-14。结论：已验证公开架构/公式对应实现和程序链路；尚未验证论文数据规模下的训练效果，不能声称完全排除复现差异。

## 环境与最终检查

- Windows，本地 Python 3.12.14，PyTorch 2.14.0+cpu；其他直接依赖固定在 `requirements-cts.txt`。
- `python -m pytest tests -q --junitxml=tmp/cts-tests.xml`：**44 passed，28.77 秒**。
- 7 个新旧 Python 入口/模块均通过 `py_compile`。
- `git diff --check` 通过。
- `train_cts.py --help`、`evaluate_cts.py --help` 正常退出。

| 验证内容 | 证据/结果 |
| --- | --- |
| Table I 编码维度、36 个独立 S-TCM、3 个 decoder | `tests/test_cts_model.py`，通过 |
| noisy-phase 重构、全局残差、Softplus、两阶段梯度流 | 同上，通过 |
| 公式 (17)–(21) 手算、RI 权重和 padding 梯度 | 同上，通过；手算例 joint loss=25.9 |
| 小批量学习能力 | 同上，12 次 Adam 更新后 loss < 初始的 65%；不代表基准效果 |
| 标准 IN 与 cIN 因果性区别 | 同上，分别确认未来依赖和频谱帧因果性 |
| Hann/320 FFT、变长端点、原始幅度/压缩变体往返 | `tests/test_cts_features.py`，通过 |
| 全静音 mixture 明确失败，静音 clean target 合法 | 同上，通过 |
| ME→joint、分组 LR、最佳权重切换、严格数据/checkpoint 检查 | `tests/test_cts_training.py`，通过 |
| 续训对照 | 两条变长多通道音频通过原 CSV 接口取首通道；2+1 轮续训与连续 3 轮所有参数逐项相等 |
| 梯度累积末尾不足完整窗口 | 与独立手算的有效帧加权 SGD 更新一致 |
| 非有限反向梯度 | 在 optimizer 更新前明确中止；不会用有限 loss 掩盖坏梯度 |
| 真正的音频评估链路 | `tests/test_cts_evaluation.py`，随机 CTS-Net checkpoint、3 秒 WAV、实际前后端和 NB-PESQ/ESTOI/BSS-Eval；增强及混合六项指标均有限 |
| 数据/源码/指标失败 | 明确拒绝配置漂移、错误样本、无效分数，不静默更改分母 |
| 双进程训练 | 额外运行 CPU/Gloo DDP 的 ME→joint，变长输入与梯度累积，通过；`tmp/ddp_smoke` 为临时证据 |
| 原训练默认行为 | 与 Git `4948c10eb3a33fed6133107d71416c58de419007` 原版 `run_epoch` 实测对照，3 批、accumulation=2：双方 loss=4.5178178151448565，参数值=0.6981123089790344 |

## 已知边界

1. 本工作区没有原论文完整数据集；没有执行 300 小时 WSJ0/3000 小时 DNS 训练、发表分数重现、多随机种子统计或主观 AB 测试。
2. 未验证 CUDA、NCCL、多卡 GPU 或 8 秒×batch 8 的 GPU 内存需求。CPU/Gloo 验证不等同于 GPU 验证。
3. 作者源码、训练权重、精确数据清单及未披露细节尚未取得。IN、STFT 边界、初始化、loss reduction、两阶段轮次分配等约定详见 `PAPER_REPRODUCTION.md`。
4. 标准 IN 配置包含未来统计量，不能宣传为严格流式因果模型。`cIN` 为明确标记的扩展。
5. 本实现总参数 4,363,317，论文表 VII 为 4.35M；小幅差异原因未获作者确认。
6. 当前评估支持 WSJ0 和 DNS 指标。VoiceBank 的 CSIG/CBAK/COVL 尚未实现。
7. 短训练测试显式允许未收敛 ME 切换以覆盖完整程序流程；不能据此声称 ME 已收敛。正式训练默认禁止这一跳过。

这些检查能够定位并防止一批具体复现错误，不能证明没有任何实现错误，更不能把未来的指标不足自动归因于论文。
