# 训练和诊断记录

`logs*/` 是运行目录，受仓库 `.gitignore` 规则忽略。需要共享的日志和结果复制到本目录后，可以正常提交到 Git。

## 已归档记录

以下文件于 2026-09-24 从用户提供的文件原样复制。文件名和日志内容保留不变；`.gitattributes` 禁止 Git 自动转换归档日志和 JSON 的换行符。

| 文件（相对于 `cbam_flat_projection64/`） | 内容 |
| --- | --- |
| `pilot.log` | 试运行日志 |
| `full_train_2gpu.log` | 双卡训练日志，从第 7 轮检查点恢复，记录第 8–30 轮 |
| `continue_31_40.log` | 第 31–40 轮继续训练日志 |
| `checkpoint_ranking.json` | 21 个检查点在同一批 600 条数据上的评估排名 |
| `overfit8_epoch40_v2/diagnostic_summary.json` | 从原始第 40 轮权重开始，8 条固定样本、200 轮、800 次更新的拟合诊断结果 |

8 条训练样本上的 PESQ 从 2.2872 提高至 2.8692，仅说明该子集的拟合有所改善。它不是 600 条数据的测试成绩，也不能证明实现与作者完全一致。600 条数据已被多次用于比较和选取检查点，不应视为一次性独立测试集。

## 从服务器提交下一次诊断结果

在 `~/EaBNet_light` 中执行。将 `run_dir` 替换为实际实验目录，待该实验完成后归档。每个实验使用独立目录，保留之前的记录。

```bash
git pull --ff-only

run_dir="./logs_cbam_flat_projection64/overfit8_lr3e4_实际时间戳"
archive_dir="./experiment_logs/cbam_flat_projection64/$(basename "$run_dir")"
mkdir -p "$archive_dir"

cp "$run_dir/training.log" "$archive_dir/"
cp "$run_dir/diagnostic_summary.json" "$archive_dir/"
cp "$run_dir/subset_manifest.json" "$archive_dir/"
cp "$run_dir/training_command.json" "$archive_dir/"

git add -- "$archive_dir"
git commit -m "Archive completed fitting diagnostic"
git push
```

`subset_manifest.json` 和 `training_command.json` 用于核对样本、裁剪位置及训练配置。以上命令只复制指定的日志与 JSON；检查点、音频和 TensorBoard 运行文件继续留在服务器的运行目录。
