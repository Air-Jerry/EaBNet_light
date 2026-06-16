# train_light.py Arguments

`train_light.py` trains the lightweight FCAE-Att-DFSMN EaBNet model for multi-channel speech enhancement. The default settings are conservative for 12 GB GPUs: small per-step batch size, gradient accumulation, AMP, and memory-safe DataLoader behavior.

## Recommended Command

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

For multi-GPU DDP, set `--nproc_per_node` to the number of GPUs and use `--parallel-mode ddp`.

## Data And Output Paths

| Argument | Default | Meaning |
| --- | --- | --- |
| `--train-dir` | `/data/ssd1/jinrui.yang/training_set` | Training set directory. It must contain `metadata.csv`. |
| `--val-dir` | `/data/ssd1/jinrui.yang/validation_set` | Validation set directory. It must contain `metadata.csv`. |
| `--checkpoint-dir` | `./checkpoints` | Relative directory for `checkpoint_latest.pt` and periodic checkpoints. |
| `--best-dir` | `./bestmodels` | Relative directory for the best validation checkpoint. |
| `--log-dir` | `./logs` | Relative TensorBoard log directory. |

`metadata.csv` should include `sample_id`, `mixture_path`, and `target_path`. `mixture_path` points to the multi-channel mixture wav, while `target_path` points to the clean target wav.

## Optimization

| Argument | Default | Meaning |
| --- | --- | --- |
| `--num-epochs` | `100` | Maximum number of training epochs. |
| `--batch-size` | `1` | Per-step batch size per process. Keep this at `1` on 12 GB GPUs. |
| `--grad-accum-steps` | `4` | Number of gradient accumulation steps. Effective batch size is roughly `batch-size * grad-accum-steps * world_size`. |
| `--learning-rate` | `1e-3` | Adam initial learning rate, matching the paper setup. |
| `--weight-decay` | `0.0` | Adam weight decay. |
| `--grad-clip` | `5.0` | Gradient clipping threshold. Set `<=0` to disable. |
| `--seed` | `1234` | Random seed. |
| `--save-every` | `10` | Save an extra checkpoint every N epochs. |

## STFT And Feature Compression

| Argument | Default | Meaning |
| --- | --- | --- |
| `--sample-rate` | `16000` | Expected sample rate for mixture and target wavs. |
| `--win-length` | `320` | STFT window length. At 16 kHz, this is 20 ms. |
| `--hop-length` | `160` | STFT hop length. At 16 kHz, this is 10 ms. |
| `--n-fft` | `512` | FFT size used by the paper. |
| `--power` | `0.5` | Magnitude compression exponent. `0.5` means square-root compression. |
| `--segment-seconds` | `4.0` | Crop length in seconds. Lower this to `3` or `2` if CUDA OOM persists. |
| `--train-random-crop` | `yes` | Use random crop during training. Validation always uses a deterministic crop. |

## DataLoader And Memory

| Argument | Default | Meaning |
| --- | --- | --- |
| `--num-workers` | `4` | DataLoader worker count. With `--strict-memory yes`, the effective value becomes `0`. |
| `--pin-memory` | `yes` | Enable pinned host memory. With `--strict-memory yes`, this is effectively disabled. |
| `--persistent-workers` | `no` | Keep DataLoader workers alive between epochs. Ignored when worker count is `0`. |
| `--prefetch-factor` | `2` | Number of prefetched batches per worker. Ignored when worker count is `0`. |
| `--strict-memory` | `yes` | Conservative memory mode. It forces worker count to `0`, disables pinned memory and persistent workers, and uses prefetch factor `1`. |
| `--empty-cache-each-epoch` | `yes` | Call `torch.cuda.empty_cache()` after each epoch. |
| `--empty-cache-every-batches` | `0` | Call `torch.cuda.empty_cache()` every N batches. `0` disables it. Try `20` when fragmentation is severe. |
| `--mem-log-every-batches` | `100` | Print memory stats every N batches. |
| `--malloc-trim-every-batches` | `100` | On Linux/glibc, try to return free CPU heap pages every N batches. |
| `--gc-every-batches` | `100` | Run Python garbage collection every N batches. |
| `--malloc-arena-max` | `2` | Best-effort Linux/glibc arena limit. |
| `--log-ram-tree` | `yes` | Log RAM for the current process and descendants. |

The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce CUDA allocator fragmentation.

## AMP And Backend Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--use-amp` | `yes` | Enable AMP and GradScaler. |
| `--model-amp` | `yes` | Run the model forward pass under AMP. STFT construction stays fp32, and Fourier blocks cast FFT inputs to fp32 internally. |
| `--allow-tf32` | `yes` | Allow TF32 matmul/cuDNN kernels on Ampere or newer GPUs. |
| `--cudnn-benchmark` | `yes` | Enable cuDNN autotuner for fixed-shape workloads. |

## Model Structure

| Argument | Default | Meaning |
| --- | --- | --- |
| `--target-ref-mic` | `0` | If target wav is multi-channel, select this channel as the target. |
| `--num-mics` | `8` | Number of input microphone channels. Must be no larger than the mixture wav channel count. |
| `--channels` | `64` | Main CRED/FCAE/FCAD channel count. |
| `--embed-dim` | `64` | Embedding tensor channel count and LSTM beamforming input size. |
| `--kd1` | `5` | Kept for EaBNet compatibility. The lightweight DFSMN memory size is fixed at 5 internally. |
| `--cd1` | `64` | DFSMN hidden units. |
| `--d-feat` | `256` | Kept for EaBNet compatibility. The lightweight CRED does not directly use it. |
| `--p` | `6` | Kept for EaBNet compatibility. The lightweight CRED does not directly use it. |
| `--q` | `3` | Kept for EaBNet compatibility. The lightweight CRED does not directly use it. |
| `--bf-type` | `lstm` | Beamforming head type. Choices: `lstm`, `cnn`. |
| `--topo-type` | `mimo` | Output topology. Choices: `mimo`, `miso`. |
| `--intra-connect` | `cat` | Kept for EaBNet compatibility. The lightweight CRED skip path uses concatenation. |
| `--norm-type` | `BN` | Normalization type. Choices: `BN`, `IN`, `cLN`. The paper uses BN. |
| `--dfsmn-layers` | `3` | Number of repeated shared DFSMN memory layers. The paper uses 3. |
| `--is-causal` | `yes` | Use causal temporal modeling. |
| `--is-u2` | `yes` | Kept for EaBNet compatibility. The lightweight CRED does not directly use it. |

## Parallelism And Devices

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gpu-ids` | empty | GPU IDs for single-process DataParallel, for example `0,1`. Empty means all visible GPUs. DDP uses `LOCAL_RANK`. |
| `--parallel-mode` | `auto` | Choices: `auto`, `none`, `dp`, `ddp`. Use `ddp` for multi-GPU training. |
| `--cpu-threads` | `8` | PyTorch intra-op CPU thread count. |
| `--interop-threads` | `1` | PyTorch inter-op CPU thread count. |
| `--cpu-thread-scale` | `1.0` | Scale factor applied to `cpu-threads`. |

## Resume And Learning-Rate Scheduling

| Argument | Default | Meaning |
| --- | --- | --- |
| `--resume` | `yes` | Resume from `checkpoint_latest.pt` if it exists. |
| `--resume-reset-lr` | `no` | After resume, overwrite optimizer LR with `--learning-rate`. |
| `--train-lr-reduce-on-plateau` | `yes` | Reduce learning rate when the selected loss metric plateaus. |
| `--lr-reduce-metric` | `val` | Metric used for plateau detection. Choices: `train`, `val`. |
| `--train-lr-patience` | `2` | Number of non-improving epochs before LR reduction. |
| `--train-lr-factor` | `0.5` | Multiplicative LR decay factor. |
| `--train-lr-min-delta` | `0.0` | Minimum loss decrease required to count as improvement. |
| `--min-learning-rate` | `1e-6` | Lower bound for learning rate. |

## Non-Finite Loss Handling

| Argument | Default | Meaning |
| --- | --- | --- |
| `--skip-non-finite-batches` | `yes` | Skip NaN/Inf loss batches. The decision is synchronized across ranks in DDP. |
| `--max-non-finite-batches-per-epoch` | `20` | Abort if skipped non-finite batches exceed this count in one epoch. |
| `--stop-on-non-finite` | `no` | Stop immediately when NaN/Inf loss is encountered. |
| `--non-finite-log-max-ids` | `8` | Maximum number of sample IDs printed for a non-finite batch. |

## OOM Tuning Order

1. Keep `--batch-size 1`.
2. Reduce `--segment-seconds 4` to `3` or `2`.
3. Keep `--use-amp yes --model-amp yes`.
4. Try `--empty-cache-every-batches 20`.
5. If memory is still insufficient, reduce model width with `--channels 48 --embed-dim 48 --cd1 48`.
