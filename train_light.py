"""
~/miniconda3/envs/EaBNet/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 train_light.py \
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
  --empty-cache-every-batches 0 \
  --allow-tf32 yes \
  --cudnn-benchmark yes \
  --train-dir /data/ssd1/jinrui.yang/training_set \
  --val-dir /data/ssd1/jinrui.yang/validation_set
"""
"""
tensorboard --logdir /data/ssd1/jinrui.yang/logs --port 6006 --host 0.0.0.0
"""

import argparse
import csv
import ctypes
import gc
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Reduce CUDA allocator fragmentation on 12GB-class GPUs. Users can still
# override this before launching if they need a different allocator policy.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import soundfile as sf
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from EaBNet_light import EaBNet, com_mag_mse_loss


# Global caches/utilities used across mini-batches to reduce overhead.
_WINDOW_CACHE: Dict[Tuple[int, str, int], torch.Tensor] = {}
_LIBC = None
try:
    _LIBC = ctypes.CDLL('libc.so.6')
except Exception:
    _LIBC = None


def get_process_ram_mb() -> float:
    """Return current process RSS in MB (Linux /proc)."""
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS: value is in kB
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return float('nan')


def _get_pid_ram_mb(pid: int) -> float:
    try:
        with open(f'/proc/{pid}/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return 0.0


def _get_child_pids(pid: int) -> List[int]:
    try:
        with open(f'/proc/{pid}/task/{pid}/children', 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception:
        return []
    if not text:
        return []
    child_pids: List[int] = []
    for part in text.split():
        try:
            child_pids.append(int(part))
        except ValueError:
            continue
    return child_pids


def get_process_tree_ram_mb() -> float:
    """Return RSS of current process + descendants in MB (Linux /proc)."""
    root_pid = os.getpid()
    total = 0.0
    queue = [root_pid]
    visited = set()
    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)
        total += _get_pid_ram_mb(pid)
        queue.extend(_get_child_pids(pid))
    return total


def format_gpu_mem_line(gpu_ids: List[int]) -> str:
    if not torch.cuda.is_available() or not gpu_ids:
        return 'GPU mem: N/A'
    parts = []
    for gid in gpu_ids:
        allocated = torch.cuda.memory_allocated(gid) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(gid) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(gid) / (1024 ** 2)
        parts.append(f'gpu{gid} alloc={allocated:.1f}MB resv={reserved:.1f}MB peak={peak:.1f}MB')
    return 'GPU mem: ' + ' | '.join(parts)


def maybe_malloc_trim() -> None:
    """Ask glibc to return free heap pages back to OS (Linux/glibc only)."""
    if _LIBC is None:
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def maybe_set_allocator_policy(arena_max: int) -> None:
    """Best-effort glibc mallopt tuning; most effective when env is set before process start."""
    if _LIBC is None or arena_max <= 0:
        return
    # glibc mallopt constants:
    # M_ARENA_MAX = -8
    try:
        _LIBC.mallopt(-8, int(arena_max))
    except Exception:
        pass


@dataclass
class SampleRecord:
    sample_id: int
    mixture_path: Path
    target_path: Path


# ---------------------------
# Dataset and input pipeline
# ---------------------------
def resolve_metadata_path(dataset_dir: Path) -> Path:
    metadata_path = dataset_dir / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f'metadata.csv not found under {dataset_dir}')
    return metadata_path


def resolve_record_path(root_dir: Path, relative_path: str) -> Path:
    normalized = relative_path.replace('\\', '/').strip()
    search_roots = [root_dir, *root_dir.parents[:3]]
    seen_roots = set()
    for base_dir in search_roots:
        if base_dir in seen_roots:
            continue
        seen_roots.add(base_dir)
        candidate = (base_dir / normalized).resolve()
        if candidate.exists():
            return candidate
    fallback = (root_dir / Path(normalized).name).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'cannot resolve path from metadata: {relative_path}')


def scan_rir_targets(records: List[SampleRecord], sample_count: int = 3) -> None:
    for record in records[:sample_count]:
        if record.target_path.name.endswith('_rir.wav'):
            raise ValueError(
                'Current dataset target files are RIR labels (*_rir.wav), but train1.py is built for EaBNet speech enhancement. '
                'Please regenerate training_set and validation_set with generate_dataset.py --target-type speech.'
            )


class EnhancementDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        sample_rate: int,
        target_ref_mic: int = 0,
        num_mics: int = 8,
        segment_samples: int = 0,
        random_crop: bool = False,
    ):
        self.dataset_dir = dataset_dir.resolve()
        self.sample_rate = sample_rate
        self.target_ref_mic = target_ref_mic
        self.num_mics = max(1, int(num_mics))
        self.segment_samples = max(0, int(segment_samples))
        self.random_crop = random_crop
        self.records = self._load_records()
        if not self.records:
            raise RuntimeError(f'no valid samples found in {self.dataset_dir}')
        scan_rir_targets(self.records)

    def _load_records(self) -> List[SampleRecord]:
        metadata_path = resolve_metadata_path(self.dataset_dir)
        metadata_dir = metadata_path.parent
        records: List[SampleRecord] = []
        with metadata_path.open('r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # Resolve relative paths against metadata location to support entries like ../training_set/...
                mixture_path = resolve_record_path(metadata_dir, row['mixture_path'])
                target_path = resolve_record_path(metadata_dir, row['target_path'])
                if not mixture_path.exists() or not target_path.exists():
                    continue
                records.append(
                    SampleRecord(
                        sample_id=int(row['sample_id']),
                        mixture_path=mixture_path,
                        target_path=target_path,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Load one multichannel mixture and one target waveform for a sample.
        record = self.records[index]
        # Read directly as float32 to avoid float64->float32 conversions and extra memory copies.
        mixture, mix_sr = sf.read(record.mixture_path, dtype='float32', always_2d=True)
        target, tgt_sr = sf.read(record.target_path, dtype='float32', always_2d=False)

        if mix_sr != self.sample_rate or tgt_sr != self.sample_rate:
            raise ValueError(
                f'sample rate mismatch for sample {record.sample_id}: mixture={mix_sr}, target={tgt_sr}, expected={self.sample_rate}'
            )

        mixture_tensor = torch.from_numpy(mixture)
        if mixture_tensor.shape[1] < self.num_mics:
            raise ValueError(
                f'mixture channels ({mixture_tensor.shape[1]}) < num_mics ({self.num_mics}) '
                f'for sample {record.sample_id}'
            )
        # Use only the first N channels to match model input mic count.
        mixture_tensor = mixture_tensor[:, :self.num_mics]

        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        if target_tensor.ndim == 2:
            if target_tensor.shape[1] <= self.target_ref_mic:
                raise ValueError(
                    f'target_ref_mic={self.target_ref_mic} out of range for sample {record.sample_id} with shape {tuple(target_tensor.shape)}'
                )
            target_tensor = target_tensor[:, self.target_ref_mic]

        target_tensor = target_tensor.view(-1)
        valid_length = min(mixture_tensor.shape[0], target_tensor.shape[0])

        if self.segment_samples > 0 and valid_length > self.segment_samples:
            # Optional fixed-length crop to control memory/throughput.
            if self.random_crop:
                start = random.randint(0, valid_length - self.segment_samples)
            else:
                start = 0
            end = start + self.segment_samples
            mixture_tensor = mixture_tensor[start:end]
            target_tensor = target_tensor[start:end]
            valid_length = self.segment_samples
        else:
            mixture_tensor = mixture_tensor[:valid_length]
            target_tensor = target_tensor[:valid_length]

        return {
            'mixture': mixture_tensor,
            'target': target_tensor,
            'num_samples': valid_length,
            'sample_id': record.sample_id,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Build a padded batch (B, T, M) and keep original lengths for frame masking.
    mixtures = [item['mixture'] for item in batch]
    targets = [item['target'] for item in batch]
    lengths = torch.tensor([item['num_samples'] for item in batch], dtype=torch.long)
    sample_ids = torch.tensor([item['sample_id'] for item in batch], dtype=torch.long)

    max_len = int(lengths.max().item())
    num_mics = mixtures[0].shape[1]

    # Fast path: if all samples are same length, stack directly to reduce CPU work.
    if bool(torch.all(lengths == lengths[0])):
        mixture_batch = torch.stack(mixtures, dim=0)
        target_batch = torch.stack(targets, dim=0)
        return {
            'mixture': mixture_batch,
            'target': target_batch,
            'lengths': lengths,
            'sample_ids': sample_ids,
        }

    mixture_batch = torch.zeros(len(batch), max_len, num_mics, dtype=torch.float32)
    target_batch = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for idx, (mixture, target) in enumerate(zip(mixtures, targets)):
        cur_len = mixture.shape[0]
        mixture_batch[idx, :cur_len] = mixture
        target_batch[idx, :cur_len] = target

    return {
        'mixture': mixture_batch,
        'target': target_batch,
        'lengths': lengths,
        'sample_ids': sample_ids,
    }


def waveform_lengths_to_frames(lengths: torch.Tensor, hop_length: int) -> List[int]:
    # Convert waveform lengths to STFT frame counts for loss masking.
    return [(int(length.item()) // hop_length) + 1 for length in lengths]


def build_stft_batch(
    mixture: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Front-end feature construction expected by EaBNet:
    # mixture -> (B, T, F, M, 2), target -> (B, 2, T, F), with power compression.
    batch_size, _, num_mics = mixture.shape
    cache_key = (win_length, device.type, device.index or 0)
    window = _WINDOW_CACHE.get(cache_key)
    if window is None:
        window = torch.hann_window(win_length, device=device)
        _WINDOW_CACHE[cache_key] = window

    mixture = mixture.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    mixture_flat = mixture.transpose(1, 2).reshape(batch_size * num_mics, -1)
    mixture_stft = torch.stft(
        mixture_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    freq_bins, frames = mixture_stft.shape[-2], mixture_stft.shape[-1]
    # Compress only magnitude and keep phase, matching EaBNet paper setup.
    # Avoid angle() + exp(1j*phase), which creates large temporary tensors.
    mixture_scale = torch.abs(mixture_stft).clamp_min(1e-8).pow(power - 1.0)
    mixture_stft = torch.stack([mixture_stft.real * mixture_scale, mixture_stft.imag * mixture_scale], dim=-1)
    mixture_stft = mixture_stft.view(batch_size, num_mics, freq_bins, frames, 2).permute(0, 3, 2, 1, 4).contiguous()

    target_stft = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    target_scale = torch.abs(target_stft).clamp_min(1e-8).pow(power - 1.0)
    target_stft = torch.stack([target_stft.real * target_scale, target_stft.imag * target_scale], dim=1)
    target_stft = target_stft.permute(0, 1, 3, 2).contiguous()

    return mixture_stft, target_stft


def create_model(args: argparse.Namespace, device: torch.device) -> EaBNet:
    # Instantiate lightweight FCAE-Att-DFSMN EaBNet backbone.
    model = EaBNet(
        k1=(2, 3),
        k2=(1, 3),
        c=args.channels,
        M=args.num_mics,
        embed_dim=args.embed_dim,
        kd1=args.kd1,
        cd1=args.cd1,
        d_feat=args.d_feat,
        p=args.p,
        q=args.q,
        is_causal=args.is_causal == 'yes',
        is_u2=args.is_u2 == 'yes',
        bf_type=args.bf_type,
        topo_type=args.topo_type,
        intra_connect=args.intra_connect,
        norm_type=args.norm_type,
        dfsmn_layers=args.dfsmn_layers,
        dfsmn_memory_size=args.dfsmn_memory_size,
    ).to(device)
    return model


def parse_gpu_ids(gpu_ids_text: str) -> List[int]:
    if not gpu_ids_text.strip():
        return []
    ids: List[int] = []
    for part in gpu_ids_text.split(','):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def model_state_dict_for_save(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module.state_dict()
    return model.state_dict()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            stripped[key[len('module.'):]] = value
        else:
            stripped[key] = value
    return stripped


def load_model_state_flexible(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    target = model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model
    target.load_state_dict(_strip_module_prefix(state_dict), strict=True)


def save_checkpoint(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def maybe_copy_best_for_infer(best_path: Path, checkpoint_dir: Path) -> None:
    compat_path = checkpoint_dir / 'best_model.pt'
    compat_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, compat_path)


def run_epoch(
    model: EaBNet,
    loader: DataLoader,
    optimizer: Adam,
    scaler: GradScaler,
    device: torch.device,
    gpu_ids: List[int],
    args: argparse.Namespace,
    training: bool,
) -> float:
    # Unified train/val loop.
    # - train mode: backward + optimizer step
    # - val mode: forward only
    model.train(mode=training)
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    amp_enabled = device.type == 'cuda' and args.use_amp == 'yes'
    model_amp_enabled = amp_enabled and args.model_amp == 'yes'
    accum_steps = max(1, int(args.grad_accum_steps))
    progress = tqdm(loader, desc='train' if training else 'val', leave=False, disable=not is_main_process())
    epoch_start_time = time.perf_counter()
    prev_iter_end = epoch_start_time
    total_data_wait = 0.0
    total_compute = 0.0
    non_finite_batches = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(progress, start=1):
        iter_start = time.perf_counter()
        total_data_wait += max(0.0, iter_start - prev_iter_end)

        mixture = batch['mixture']
        target = batch['target']
        lengths = batch['lengths']
        frame_list = waveform_lengths_to_frames(lengths, args.hop_length)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=amp_enabled):
                # Build STFT-domain inputs/labels on the fly from waveform batches.
                mixture_stft, target_stft = build_stft_batch(
                    mixture=mixture,
                    target=target,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    power=args.power,
                    device=device,
                )

            # Keep FFT feature construction in fp32, but allow conv/LSTM activations
            # inside the model to use AMP. Fourier blocks cast their FFT inputs back
            # to fp32 internally for numerical compatibility.
            with autocast(device_type=device.type, enabled=model_amp_enabled):
                estimate = model(mixture_stft)
            raw_loss = com_mag_mse_loss(estimate.float(), target_stft.float(), frame_list)

            # In DDP, require finite loss on all ranks before stepping.
            local_is_finite = bool(torch.isfinite(raw_loss.detach()).item())
            global_is_finite = local_is_finite
            if is_distributed():
                finite_flag = torch.tensor([1 if local_is_finite else 0], dtype=torch.int32, device=device)
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                global_is_finite = bool(finite_flag.item() == 1)

            if not global_is_finite:
                non_finite_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)

                if is_main_process():
                    sid_text = ''
                    sample_ids = batch.get('sample_ids')
                    if isinstance(sample_ids, torch.Tensor):
                        max_ids = max(1, int(args.non_finite_log_max_ids))
                        sid_vals = sample_ids[:max_ids].tolist()
                        sid_text = f', sample_ids={sid_vals}'
                    tqdm.write(
                        f'[WARN][{"train" if training else "val"}] Non-finite loss at batch={batch_idx}{sid_text}. '
                        f'skip={args.skip_non_finite_batches}, count={non_finite_batches}'
                    )

                iter_end = time.perf_counter()
                total_compute += max(0.0, iter_end - iter_start)
                prev_iter_end = iter_end

                del mixture, target, lengths, frame_list, mixture_stft, target_stft, estimate, raw_loss, batch

                if args.stop_on_non_finite == 'yes':
                    raise RuntimeError('Encountered non-finite loss and --stop-on-non-finite=yes')
                if non_finite_batches > max(0, int(args.max_non_finite_batches_per_epoch)):
                    raise RuntimeError(
                        f'Non-finite loss batch count exceeded max_non_finite_batches_per_epoch='
                        f'{args.max_non_finite_batches_per_epoch}'
                    )
                if args.skip_non_finite_batches == 'yes':
                    continue
                raise RuntimeError('Encountered non-finite loss and skipping is disabled')

            if training:
                # Support gradient accumulation to emulate larger batch sizes.
                loss = raw_loss / accum_steps
                scaler.scale(loss).backward()
                should_step = (batch_idx % accum_steps == 0) or (batch_idx == len(loader))
                if should_step:
                    if args.grad_clip > 0:
                        # Clip after unscale so threshold applies to true gradients.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss = raw_loss

        total_loss += float(raw_loss.detach().item())
        total_batches += 1
        total_samples += int(mixture.shape[0])
        progress.set_postfix(loss=f'{raw_loss.detach().item():.4f}')

        if args.mem_log_every_batches > 0 and (batch_idx % args.mem_log_every_batches == 0):
            ram_self_mb = get_process_ram_mb()
            ram_tree_mb = get_process_tree_ram_mb() if args.log_ram_tree == 'yes' else float('nan')
            gpu_line = format_gpu_mem_line(gpu_ids)
            phase = 'train' if training else 'val'
            if args.log_ram_tree == 'yes':
                if is_main_process():
                    tqdm.write(
                        f'[MEM][{phase}] batch={batch_idx} RAM(self/tree)={ram_self_mb:.1f}/{ram_tree_mb:.1f}MB | {gpu_line}'
                    )
            else:
                if is_main_process():
                    tqdm.write(f'[MEM][{phase}] batch={batch_idx} RAM={ram_self_mb:.1f}MB | {gpu_line}')

        if args.malloc_trim_every_batches > 0 and (batch_idx % args.malloc_trim_every_batches == 0):
            maybe_malloc_trim()

        if args.gc_every_batches > 0 and (batch_idx % args.gc_every_batches == 0):
            gc.collect()

        if device.type == 'cuda' and args.empty_cache_every_batches > 0 and (batch_idx % args.empty_cache_every_batches == 0):
            torch.cuda.empty_cache()

        iter_end = time.perf_counter()
        total_compute += max(0.0, iter_end - iter_start)
        prev_iter_end = iter_end

        # Release references early to reduce peak host/GPU memory pressure.
        del mixture, target, lengths, frame_list, mixture_stft, target_stft, estimate, raw_loss, loss, batch

    progress.close()
    if total_batches == 0:
        return math.nan

    elapsed = max(1e-9, time.perf_counter() - epoch_start_time)
    sps = total_samples / elapsed
    ms_per_batch = (elapsed / total_batches) * 1000.0
    data_wait_pct = (total_data_wait / elapsed) * 100.0
    compute_pct = (total_compute / elapsed) * 100.0
    phase = 'train' if training else 'val'

    global_samples = total_samples
    global_batches = total_batches
    global_elapsed = elapsed
    global_sps = sps
    if is_distributed():
        stats = torch.tensor([float(total_samples), float(total_batches)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_samples = int(stats[0].item())
        global_batches = int(stats[1].item())

        elapsed_t = torch.tensor([elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
        global_elapsed = max(1e-9, elapsed_t.item())
        global_sps = global_samples / global_elapsed

    if is_main_process() and args.log_perf == 'yes':
        print(
            f'[PERF][{phase}] samples(local/global)={total_samples}/{global_samples}, '
            f'batches(local/global)={total_batches}/{global_batches}, '
            f'samples/s(local/global)={sps:.1f}/{global_sps:.1f}, ms/batch(local)={ms_per_batch:.1f}, '
            f'data_wait={data_wait_pct:.1f}%, compute={compute_pct:.1f}%, non_finite_skips(local)={non_finite_batches}'
        )

    avg_loss = total_loss / total_batches
    if is_distributed():
        # Report globally averaged loss across ranks.
        t = torch.tensor([avg_loss], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg_loss = (t.item() / dist.get_world_size())
    return avg_loss


def auto_resume_if_available(
    model: EaBNet,
    optimizer: Adam,
    scaler: GradScaler,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, float, Dict[str, float]]:
    # Resume full training state (model/optimizer/scaler/LR-reducer metadata).
    if not checkpoint_path.exists():
        return 1, float('inf'), {}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state_flexible(model, checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler_state = checkpoint.get('scaler_state_dict')
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    next_epoch = int(checkpoint['epoch']) + 1
    best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
    print(f'Resumed from {checkpoint_path} at epoch {checkpoint["epoch"]}, best_val_loss={best_val_loss:.6f}')
    lr_reducer_state = checkpoint.get('lr_reducer_state', {})
    return next_epoch, best_val_loss, lr_reducer_state


def parse_args() -> argparse.Namespace:
    # CLI config surface for data, model, optimization, memory, and parallelism.
    parser = argparse.ArgumentParser(description='Train lightweight FCAE-Att-DFSMN EaBNet for multichannel speech enhancement.')
    parser.add_argument('--train-dir', default='/data/ssd1/jinrui.yang/training_set')
    parser.add_argument('--val-dir', default='/data/ssd1/jinrui.yang/validation_set')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--best-dir', default='./bestmodels')
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--grad-accum-steps', type=int, default=4, help='Number of steps to accumulate gradients before optimizer step')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--win-length', type=int, default=320)
    parser.add_argument('--hop-length', type=int, default=160)
    parser.add_argument('--n-fft', type=int, default=512)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', choices=['yes', 'no'], default='yes')
    parser.add_argument('--use-amp', choices=['yes', 'no'], default='yes')
    parser.add_argument('--model-amp', choices=['yes', 'no'], default='yes', help='Run model forward under AMP while keeping FFT feature construction in fp32')
    parser.add_argument('--target-ref-mic', type=int, default=0)
    parser.add_argument('--num-mics', type=int, default=8)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--kd1', type=int, default=5)
    parser.add_argument('--cd1', type=int, default=64)
    parser.add_argument('--d-feat', type=int, default=256)
    parser.add_argument('--p', type=int, default=6)
    parser.add_argument('--q', type=int, default=3)
    parser.add_argument('--bf-type', choices=['lstm', 'cnn'], default='lstm')
    parser.add_argument('--topo-type', choices=['mimo', 'miso'], default='mimo')
    parser.add_argument('--intra-connect', choices=['cat', 'add'], default='cat')
    parser.add_argument('--norm-type', default='BN')
    parser.add_argument('--dfsmn-layers', type=int, default=3, help='Number of repeated shared DFSMN memory layers in CRED')
    parser.add_argument('--dfsmn-memory-size', type=int, default=20, help='Number of left-context memory taps in the shared DFSMN block')
    parser.add_argument('--is-causal', choices=['yes', 'no'], default='yes')
    parser.add_argument('--is-u2', choices=['yes', 'no'], default='yes')
    parser.add_argument('--gpu-ids', default='', help='Comma-separated GPU IDs, e.g. "0,1". Empty = use all visible GPUs.')
    parser.add_argument('--cpu-threads', type=int, default=8, help='Max intra-op CPU threads for PyTorch')
    parser.add_argument('--interop-threads', type=int, default=1, help='Max inter-op CPU threads for PyTorch')
    parser.add_argument('--cpu-thread-scale', type=float, default=1.0, help='Scale factor for cpu-threads, e.g. 2.0 to double')
    parser.add_argument('--pin-memory', choices=['yes', 'no'], default='yes')
    parser.add_argument('--persistent-workers', choices=['yes', 'no'], default='no')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor (only when num_workers > 0)')
    parser.add_argument('--strict-memory', choices=['yes', 'no'], default='yes', help='Use conservative DataLoader memory settings')
    parser.add_argument('--log-ram-tree', choices=['yes', 'no'], default='yes', help='Log RAM for process tree (main + workers)')
    parser.add_argument('--empty-cache-each-epoch', choices=['yes', 'no'], default='yes')
    parser.add_argument('--empty-cache-every-batches', type=int, default=0, help='Call torch.cuda.empty_cache every N batches (0 disables)')
    parser.add_argument('--mem-log-every-batches', type=int, default=100, help='Print memory usage every N batches')
    parser.add_argument('--malloc-trim-every-batches', type=int, default=100, help='Call malloc_trim every N batches (0 disables)')
    parser.add_argument('--gc-every-batches', type=int, default=100, help='Call gc.collect every N batches (0 disables)')
    parser.add_argument('--malloc-arena-max', type=int, default=2, help='Best-effort glibc M_ARENA_MAX via mallopt at startup (<=0 disables)')
    parser.add_argument('--segment-seconds', type=float, default=4.0, help='Crop each sample to this duration in seconds (<=0 disables)')
    parser.add_argument('--train-random-crop', choices=['yes', 'no'], default='yes', help='Use random crop for training when segment-seconds > 0')
    parser.add_argument('--parallel-mode', choices=['auto', 'none', 'dp', 'ddp'], default='auto', help='Multi-GPU mode: dp or ddp (recommended).')
    parser.add_argument('--log-perf', choices=['yes', 'no'], default='yes', help='Print epoch-level throughput and bottleneck breakdown')
    parser.add_argument('--allow-tf32', choices=['yes', 'no'], default='yes', help='Allow TF32 matmul/cuDNN kernels on Ampere+ GPUs for higher throughput')
    parser.add_argument('--cudnn-benchmark', choices=['yes', 'no'], default='yes', help='Enable cuDNN autotuner for fixed-shape workloads')
    parser.add_argument('--train-lr-reduce-on-plateau', choices=['yes', 'no'], default='yes', help='Halve LR when selected metric does not improve for N consecutive epochs')
    parser.add_argument('--lr-reduce-metric', choices=['train', 'val'], default='val', help='Metric used for LR plateau check')
    parser.add_argument('--train-lr-patience', type=int, default=2, help='Number of consecutive non-improving epochs before reducing LR')
    parser.add_argument('--train-lr-factor', type=float, default=0.5, help='Multiplicative LR decay factor when plateau is reached')
    parser.add_argument('--train-lr-min-delta', type=float, default=0.0, help='Minimum decrease in selected metric to count as an improvement')
    parser.add_argument('--min-learning-rate', type=float, default=1e-6, help='Lower bound for learning rate when reducing on plateau')
    parser.add_argument('--resume-reset-lr', choices=['yes', 'no'], default='no', help='After resume, override optimizer LR with --learning-rate')
    parser.add_argument('--skip-non-finite-batches', choices=['yes', 'no'], default='yes', help='Skip batches with NaN/Inf loss (DDP-synchronized decision)')
    parser.add_argument('--max-non-finite-batches-per-epoch', type=int, default=20, help='Abort epoch when skipped non-finite batches exceed this number')
    parser.add_argument('--stop-on-non-finite', choices=['yes', 'no'], default='no', help='Abort immediately when NaN/Inf loss is encountered')
    parser.add_argument('--non-finite-log-max-ids', type=int, default=8, help='Maximum sample IDs to print when non-finite loss occurs')
    return parser.parse_args()


def main() -> None:
    # 1) Parse config and initialize runtime (DDP/threads/seeds/backends).
    args = parse_args()
    maybe_set_allocator_policy(args.malloc_arena_max)

    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    env_rank = int(os.environ.get('RANK', '0'))
    env_local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_ddp = args.parallel_mode == 'ddp' or (args.parallel_mode == 'auto' and env_world_size > 1)

    if use_ddp and env_world_size > 1:
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://')

    if args.cpu_threads > 0:
        scaled_threads = max(1, int(round(args.cpu_threads * args.cpu_thread_scale)))
        cpu_count = os.cpu_count() or scaled_threads
        scaled_threads = min(scaled_threads, cpu_count)
        torch.set_num_threads(scaled_threads)
        args.cpu_threads = scaled_threads
    if args.interop_threads > 0:
        torch.set_num_interop_threads(args.interop_threads)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = (args.allow_tf32 == 'yes')
        torch.backends.cudnn.allow_tf32 = (args.allow_tf32 == 'yes')
        torch.backends.cudnn.benchmark = (args.cudnn_benchmark == 'yes')
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if args.allow_tf32 == 'yes' else 'highest')

    train_dir = Path(args.train_dir).expanduser()
    val_dir = Path(args.val_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    best_dir = Path(args.best_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser()
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    best_path = best_dir / 'best_model.pt'

    gpu_ids: List[int] = []
    if torch.cuda.is_available():
        if use_ddp and env_world_size > 1:
            torch.cuda.set_device(env_local_rank)
            device = torch.device(f'cuda:{env_local_rank}')
            gpu_ids = [env_local_rank]
        else:
            requested_ids = parse_gpu_ids(args.gpu_ids)
            if requested_ids:
                gpu_ids = requested_ids
            else:
                gpu_ids = list(range(torch.cuda.device_count()))
            if not gpu_ids:
                raise RuntimeError('CUDA is available but no GPU IDs selected')
            torch.cuda.set_device(gpu_ids[0])
            device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')

    if is_main_process():
        print(f'Device: {device}')
        if use_ddp and env_world_size > 1:
            print(f'DDP world_size/rank/local_rank: {env_world_size}/{env_rank}/{env_local_rank}')
        if gpu_ids:
            print(f'GPU IDs: {gpu_ids}')
        print(f'CPU threads (intra/inter-op): {torch.get_num_threads()}/{torch.get_num_interop_threads()}')
        print(f'TF32/cudnn_benchmark: {args.allow_tf32}/{args.cudnn_benchmark}')
        print(f'MALLOC_ARENA_MAX env: {os.environ.get("MALLOC_ARENA_MAX", "<unset>")} | mallopt target: {args.malloc_arena_max}')
        print(f'Train dir: {train_dir}')
        print(f'Val dir: {val_dir}')
        print(f'Checkpoint dir: {checkpoint_dir}')
        print(f'Best model dir: {best_dir}')
        print(f'TensorBoard log dir: {log_dir}')

    # 2) Build datasets/loaders.
    segment_samples = int(round(args.segment_seconds * args.sample_rate)) if args.segment_seconds > 0 else 0
    train_dataset = EnhancementDataset(
        train_dir,
        sample_rate=args.sample_rate,
        target_ref_mic=args.target_ref_mic,
        num_mics=args.num_mics,
        segment_samples=segment_samples,
        random_crop=(args.train_random_crop == 'yes'),
    )
    val_dataset = EnhancementDataset(
        val_dir,
        sample_rate=args.sample_rate,
        target_ref_mic=args.target_ref_mic,
        num_mics=args.num_mics,
        segment_samples=segment_samples,
        random_crop=False,
    )

    effective_num_workers = args.num_workers
    effective_pin_memory = (device.type == 'cuda' and args.pin_memory == 'yes')
    effective_persistent_workers = (args.num_workers > 0 and args.persistent_workers == 'yes')
    effective_prefetch_factor = max(1, int(args.prefetch_factor))

    if args.strict_memory == 'yes':
        # Conservative defaults to avoid host RAM growth from pinned-memory queues and deep prefetch.
        effective_pin_memory = False
        effective_persistent_workers = False
        effective_prefetch_factor = 1
        effective_num_workers = 0

    if is_main_process():
        print(
            'DataLoader config: '
            f'workers={effective_num_workers}, pin_memory={effective_pin_memory}, '
            f'persistent_workers={effective_persistent_workers}, prefetch_factor={effective_prefetch_factor if effective_num_workers > 0 else "N/A"}, '
            f'strict_memory={args.strict_memory}, segment_samples={segment_samples}, parallel_mode={args.parallel_mode}'
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=(effective_num_workers > 0 and effective_persistent_workers),
        collate_fn=collate_batch,
    )
    if effective_num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = effective_prefetch_factor

    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=(effective_num_workers > 0 and effective_persistent_workers),
        collate_fn=collate_batch,
    )
    if effective_num_workers > 0:
        val_loader_kwargs['prefetch_factor'] = effective_prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    # 3) Build model + optimizer + AMP scaler.
    model = create_model(args, device)
    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, output_device=device.index if device.type == 'cuda' else None)
        if is_main_process():
            print('Enabled DistributedDataParallel')
    elif args.parallel_mode in ('auto', 'dp') and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        if is_main_process():
            print(f'Enabled DataParallel on {len(gpu_ids)} GPUs')

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(device='cuda', enabled=device.type == 'cuda' and args.use_amp == 'yes')
    writer = SummaryWriter(log_dir=str(log_dir)) if is_main_process() else None

    # 4) Optionally resume from latest checkpoint.
    start_epoch = 1
    best_val_loss = float('inf')
    best_plateau_loss_for_lr = float('inf')
    stale_train_epochs = 0
    if args.resume == 'yes':
        start_epoch, best_val_loss, lr_reducer_state = auto_resume_if_available(model, optimizer, scaler, latest_path, device)
        if args.resume_reset_lr == 'yes':
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(args.learning_rate)
            if is_main_process():
                print(f'[LR] resume-reset-lr enabled. Forced LR={optimizer.param_groups[0]["lr"]:.8f}')
        if lr_reducer_state:
            best_plateau_loss_for_lr = float(
                lr_reducer_state.get(
                    'best_plateau_loss_for_lr',
                    lr_reducer_state.get('best_train_loss_for_lr', best_plateau_loss_for_lr),
                )
            )
            stale_train_epochs = int(lr_reducer_state.get('stale_train_epochs', stale_train_epochs))

    if is_main_process():
        print(f'Training samples: {len(train_dataset)}')
        print(f'Validation samples: {len(val_dataset)}')

    # 5) Epoch loop: train -> validate -> LR update -> checkpoint.
    for epoch in range(start_epoch, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_ram_before = get_process_ram_mb()
        if torch.cuda.is_available() and gpu_ids:
            for gid in gpu_ids:
                torch.cuda.reset_peak_memory_stats(gid)

        if is_main_process():
            print(f'\nEpoch {epoch}/{args.num_epochs}')
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, gpu_ids, args, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, scaler, device, gpu_ids, args, training=False)

        if args.train_lr_reduce_on_plateau == 'yes':
            # Manual ReduceLROnPlateau logic to keep behavior explicit and checkpoint-friendly.
            metric_value = float(val_loss if args.lr_reduce_metric == 'val' else train_loss)
            improved = (
                math.isfinite(metric_value)
                and metric_value < (best_plateau_loss_for_lr - args.train_lr_min_delta)
            )
            if improved:
                best_plateau_loss_for_lr = metric_value
                stale_train_epochs = 0
            else:
                stale_train_epochs += 1

            if stale_train_epochs >= max(1, int(args.train_lr_patience)):
                lr_changed = False
                for param_group in optimizer.param_groups:
                    old_lr = float(param_group['lr'])
                    new_lr = max(float(args.min_learning_rate), old_lr * float(args.train_lr_factor))
                    if new_lr < old_lr:
                        lr_changed = True
                    param_group['lr'] = new_lr
                stale_train_epochs = 0
                if is_main_process() and lr_changed:
                    print(
                        f'[LR] {args.lr_reduce_metric}-loss plateau reached. '
                        f'New LR={optimizer.param_groups[0]["lr"]:.8f} '
                        f'(factor={args.train_lr_factor}, patience={args.train_lr_patience})'
                    )

        epoch_ram_after = get_process_ram_mb()
        mem_line = format_gpu_mem_line(gpu_ids)
        if is_main_process():
            print(
                f'[MEM] epoch={epoch} RAM(before/after)={epoch_ram_before:.1f}/{epoch_ram_after:.1f}MB | {mem_line}'
            )

        # Help long runs keep memory stable.
        gc.collect()
        if torch.cuda.is_available() and args.empty_cache_each_epoch == 'yes':
            torch.cuda.empty_cache()

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()

        if is_main_process():
            print(
                f'Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, '
                f'lr={optimizer.param_groups[0]["lr"]:.8f}'
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                print(f'Updated best model: {best_path} (val_loss={best_val_loss:.6f})')

        checkpoint_state = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'model_state_dict': model_state_dict_for_save(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'lr_reducer_state': {
                'best_plateau_loss_for_lr': float(best_plateau_loss_for_lr),
                'stale_train_epochs': int(stale_train_epochs),
                'metric': args.lr_reduce_metric,
            },
            'args': vars(args),
        }
        if is_main_process():
            save_checkpoint(latest_path, checkpoint_state)

        if is_main_process() and epoch % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f'model_epoch_{epoch}.pt', checkpoint_state)

        if is_main_process() and val_loss == best_val_loss:
            save_checkpoint(best_path, checkpoint_state)
            maybe_copy_best_for_infer(best_path, checkpoint_dir)

    # 6) Final cleanup.
    if writer is not None:
        writer.close()
    if is_main_process():
        print('\nTraining finished.')

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
