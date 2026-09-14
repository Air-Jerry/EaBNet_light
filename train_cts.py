"""CTS-Net training with the existing metadata.csv, Dataset and epoch loop.

The paper specifies 60 total epochs but does not allocate them between stages.
The configurable 30-epoch ME cap below is an explicit implementation assumption.
ME must reach the validation stopping criterion before joint training starts.
"""

import argparse
import functools
import gc
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import train_light as existing
from CTSNet import CTSNet, cts_loss
from cts_features import build_stft_batch


PAPER_DEFAULTS = {
    'num_epochs': 60, 'batch_size': 8, 'grad_accum_steps': 1,
    'sample_rate': 16000, 'n_fft': 320, 'win_length': 320,
    'hop_length': 160, 'power': 1.0, 'num_mics': 1,
    'segment_seconds': 8.0, 'learning_rate': 1e-3,
    'weight_decay': 0.0, 'grad_clip': 0.0,
    'norm_type': 'IN', 'is_causal': 'yes', 'parallel_mode': 'none',
    'use_amp': 'no', 'model_amp': 'no', 'allow_tf32': 'no',
    'cudnn_benchmark': 'no', 'train_lr_patience': 3,
    'train_lr_factor': 0.5, 'train_lr_min_delta': 0.0,
    'lr_reduce_metric': 'val', 'skip_non_finite_batches': 'no',
    'stop_on_non_finite': 'yes', 'resume_reset_lr': 'no',
    'checkpoint_dir': './checkpoints_cts', 'best_dir': './bestmodels_cts',
    'log_dir': './logs_cts',
}

# Options accepted by the original launcher but irrelevant to this architecture.
LEGACY_MODEL_FLAGS = (
    '--channels', '--embed-dim', '--kd1', '--cd1', '--d-feat', '--p', '--q',
    '--bf-type', '--topo-type', '--intra-connect', '--dfsmn-layers',
    '--dfsmn-memory-size', '--is-u2',
)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--pretrain-max-epochs', type=int, default=30,
                        help='ME epoch cap, an allocation assumption within --num-epochs total')
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--joint-me-learning-rate', type=float, default=1e-4)
    parser.add_argument('--joint-cs-learning-rate', type=float, default=1e-3)
    parser.add_argument('--allow-unconverged-pretrain', choices=['yes', 'no'], default='no',
                        help='Debug override: allow joint training before ME convergence; records a deviation')
    parser.add_argument('--hash-audio', choices=['yes', 'no'], default='yes',
                        help='Hash input audio contents for resume/data provenance (one initial scan)')
    parser.add_argument('--deterministic', choices=['yes', 'no'], default='yes')
    if '--help' in argv or '-h' in argv:
        print('CTS-Net specific options (all original data/runtime options follow):')
        print(parser.format_help())
        print('Paper defaults:', json.dumps(PAPER_DEFAULTS, sort_keys=True))
    cts_args, remaining = parser.parse_known_args(argv)
    for flag in LEGACY_MODEL_FLAGS:
        if any(token == flag or token.startswith(flag + '=') for token in remaining):
            parser.error(f'{flag} configures EaBNet, not CTS-Net; CTS-Net topology is fixed by the paper')
    args = existing.parse_args(remaining, defaults=PAPER_DEFAULTS)
    for key, value in vars(cts_args).items():
        setattr(args, key, value)
    validate_args(args)
    return args


def validate_args(args):
    required = {'sample_rate': 16000, 'n_fft': 320, 'win_length': 320,
                'hop_length': 160, 'power': 1.0, 'num_mics': 1}
    for name, value in required.items():
        if getattr(args, name) != value:
            raise ValueError(f'CTS-Net paper configuration requires --{name.replace("_", "-")}={value}')
    if args.target_ref_mic < 0:
        raise ValueError('--target-ref-mic must be nonnegative')
    if args.batch_size < 1 or args.grad_accum_steps < 1 or args.num_epochs < 2:
        raise ValueError('batch/accumulation must be positive; total epochs must leave room for both stages')
    if args.pretrain_max_epochs < 1 or args.pretrain_max_epochs >= args.num_epochs:
        raise ValueError('--pretrain-max-epochs must lie between 1 and --num-epochs minus 1')
    if args.early_stop_patience < 1 or args.train_lr_patience < 1:
        raise ValueError('LR and stopping patience must be positive')
    if args.lr_reduce_metric != 'val' or args.train_lr_reduce_on_plateau != 'yes':
        raise ValueError('CTS-Net uses validation loss for LR reduction')
    if args.resume_reset_lr != 'no':
        raise ValueError('CTS-Net resume preserves per-stage optimizer LR; --resume-reset-lr must be no')
    if args.skip_non_finite_batches != 'no' or args.stop_on_non_finite != 'yes':
        raise ValueError('CTS-Net reproduction requires failure on non-finite loss')
    if args.save_every < 1 or args.segment_seconds <= 0:
        raise ValueError('--save-every and --segment-seconds must be positive')
    if min(args.learning_rate, args.joint_me_learning_rate, args.joint_cs_learning_rate) <= 0:
        raise ValueError('Learning rates must be positive')
    if not 0 < args.train_lr_factor < 1:
        raise ValueError('--train-lr-factor must be between zero and one')
    if args.deterministic == 'yes' and args.cudnn_benchmark == 'yes':
        raise ValueError('Deterministic runs require --cudnn-benchmark no')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp_path, path)


def make_manifest(args, datasets, world_size):
    audio_cache = {}
    splits = {}
    target_keys = {}
    for name, dataset in datasets.items():
        records = []
        seen_ids = set()
        target_keys[name] = set()
        for record in dataset.records:
            if record.sample_id in seen_ids:
                raise ValueError(f'Duplicate sample_id={record.sample_id} in {name} metadata')
            seen_ids.add(record.sample_id)
            if record.mixture_path == record.target_path:
                raise ValueError(f'Mixture and target are the same file for {name} sample {record.sample_id}')
            item = {'sample_id': record.sample_id}
            for field in ('mixture_path', 'target_path'):
                path = getattr(record, field)
                key = str(path)
                if key not in audio_cache:
                    stat = path.stat()
                    info = sf.info(path)
                    audio_cache[key] = {'path': key, 'bytes': stat.st_size,
                                        'sample_rate': info.samplerate, 'frames': info.frames,
                                        'channels': info.channels,
                                        'sha256': sha256_file(path) if args.hash_audio == 'yes' else None}
                item[field] = audio_cache[key]
                audio = item[field]
                if audio['sample_rate'] != args.sample_rate:
                    raise ValueError(f'Sample rate mismatch in {path}: {audio["sample_rate"]}')
                if audio['frames'] <= args.n_fft // 2:
                    raise ValueError(f'Audio too short for centered reflection STFT: {path}')
            mixture_info, target_info = item['mixture_path'], item['target_path']
            if mixture_info['frames'] != target_info['frames']:
                raise ValueError(f'Paired audio lengths differ for {name} sample {record.sample_id}; '
                                 'refusing silent truncation')
            if mixture_info['channels'] < args.num_mics:
                raise ValueError(f'Insufficient mixture channels for {name} sample {record.sample_id}')
            if target_info['channels'] <= args.target_ref_mic:
                raise ValueError(f'--target-ref-mic out of range for {name} sample {record.sample_id}')
            target_keys[name].add('path:' + target_info['path'])
            if target_info['sha256']:
                target_keys[name].add('sha256:' + target_info['sha256'])
            records.append(item)
        splits[name] = {
            'metadata_path': str(dataset.dataset_dir / 'metadata.csv'),
            'metadata_sha256': sha256_file(dataset.dataset_dir / 'metadata.csv'),
            'records': records,
        }
    overlap = target_keys['train'] & target_keys['val']
    if overlap:
        raise ValueError(f'Training/validation clean targets overlap ({len(overlap)} matching path/hash keys); '
                         'paper uses disjoint clean utterances for these splits')
    root = Path(__file__).resolve().parent
    code = {name: sha256_file(root / name) for name in
            ('CTSNet.py', 'cts_features.py', 'train_cts.py', 'train_light.py')}
    payload = {'schema_version': 1, 'model_name': 'CTSNet', 'splits': splits,
               'code_sha256': code, 'torch_version': str(torch.__version__),
               'numpy_version': np.__version__, 'cuda_version': torch.version.cuda,
               'world_size': world_size, 'hash_audio': args.hash_audio}
    payload['digest'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng(loader_generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader_generator': loader_generator.get_state()}


def restore_rng(state, loader_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([entry.cpu() for entry in state['cuda']])
    loader_generator.set_state(state['loader_generator'].cpu())


def unwrap(model):
    return model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model


def make_optimizer(model, stage, args):
    core = unwrap(model)
    core.set_stage(stage)
    groups = [{'params': core.me_net.parameters(), 'lr': args.learning_rate, 'name': 'me_net'}]
    if stage == 'joint':
        groups[0]['lr'] = args.joint_me_learning_rate
        groups.append({'params': core.cs_net.parameters(), 'lr': args.joint_cs_learning_rate,
                       'name': 'cs_net'})
    return Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)


def wrap_model(core, device, gpu_ids, args):
    if existing.is_distributed():
        # Rebuild this wrapper when trainable parameters change between stages.
        return DDP(core, device_ids=[device.index] if device.type == 'cuda' else None,
                   output_device=device.index if device.type == 'cuda' else None)
    if args.parallel_mode in ('auto', 'dp') and len(gpu_ids) > 1:
        return torch.nn.DataParallel(core, device_ids=gpu_ids, output_device=gpu_ids[0])
    return core


def build_loaders(args, generator):
    segment_samples = round(args.segment_seconds * args.sample_rate)
    datasets = {
        split: existing.EnhancementDataset(
            Path(getattr(args, split + '_dir')).expanduser(), args.sample_rate,
            target_ref_mic=args.target_ref_mic, num_mics=args.num_mics,
            segment_samples=segment_samples,
            random_crop=(split == 'train' and args.train_random_crop == 'yes'))
        for split in ('train', 'val')
    }
    workers = 0 if args.strict_memory == 'yes' else args.num_workers
    if workers < 0:
        raise ValueError('--num-workers must be nonnegative')
    if args.persistent_workers == 'yes' and workers:
        raise ValueError('Exact epoch-boundary RNG resume requires --persistent-workers no')
    sampler = DistributedSampler(datasets['train'], shuffle=True, seed=args.seed) if existing.is_distributed() else None
    loaders = {}
    for split, dataset in datasets.items():
        kwargs = dict(dataset=dataset, batch_size=args.batch_size,
                      shuffle=(split == 'train' and sampler is None),
                      sampler=sampler if split == 'train' else None,
                      num_workers=workers, collate_fn=existing.collate_batch,
                      pin_memory=(args.strict_memory != 'yes' and args.pin_memory == 'yes'),
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=generator)
        if workers:
            kwargs['prefetch_factor'] = max(1, args.prefetch_factor)
        # Each distributed rank evaluates the complete validation set. This costs
        # extra compute but avoids padded DistributedSampler samples biasing loss.
        loaders[split] = DataLoader(**kwargs)
    return datasets, loaders, sampler


def setup_runtime(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1 and args.parallel_mode not in ('auto', 'ddp'):
        raise ValueError('torchrun WORLD_SIZE>1 requires --parallel-mode ddp or auto')
    use_cuda = args.device != 'cpu' and torch.cuda.is_available()
    if args.device == 'cuda' and not use_cuda:
        raise RuntimeError('CUDA requested but unavailable')
    gpu_ids = []
    if use_cuda:
        gpu_ids = ([local_rank] if world_size > 1 else
                   (existing.parse_gpu_ids(args.gpu_ids) or [0]))
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device('cuda', gpu_ids[0])
    else:
        device = torch.device('cpu')
    if world_size > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo', init_method='env://')
    if args.cpu_threads > 0:
        torch.set_num_threads(min(os.cpu_count() or 1,
                                 max(1, round(args.cpu_threads * args.cpu_thread_scale))))
    if args.interop_threads > 0:
        torch.set_num_interop_threads(args.interop_threads)
    existing.maybe_set_allocator_policy(args.malloc_arena_max)
    seed_everything(args.seed, args.deterministic == 'yes')
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32 == 'yes'
    torch.backends.cudnn.allow_tf32 = args.allow_tf32 == 'yes'
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == 'yes'
    return device, gpu_ids, world_size, rank


def check_resume(checkpoint, args, manifest):
    if checkpoint.get('model_name') != 'CTSNet':
        raise ValueError('Checkpoint is not a CTSNet training checkpoint')
    if checkpoint.get('reproducibility_manifest', {}).get('digest') != manifest['digest']:
        raise ValueError('Resume data/code/runtime manifest changed; use a new run directory and --resume no')
    # Epoch budget and output locations may change, but the actual optimization,
    # input representation and random-data schedule must match the saved run.
    mutable = {'num_epochs', 'pretrain_max_epochs', 'allow_unconverged_pretrain',
               'checkpoint_dir', 'best_dir', 'log_dir', 'resume', 'save_every',
               'mem_log_every_batches', 'log_perf', 'log_ram_tree',
               'empty_cache_each_epoch', 'empty_cache_every_batches',
               'malloc_trim_every_batches', 'gc_every_batches'}
    saved = checkpoint['args']
    changed = [key for key, value in vars(args).items()
               if key not in mutable and saved.get(key) != value]
    if changed:
        raise ValueError('Resume configuration differs: ' + ', '.join(changed))


def load_verified_best(path, state, args, manifest, device='cpu'):
    if not Path(path).exists():
        raise FileNotFoundError(f'Best checkpoint required by the saved training state is missing: {path}')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    check_resume(checkpoint, args, manifest)
    if checkpoint.get('stage') != state['stage']:
        raise ValueError(f'Best checkpoint stage does not match saved state: {path}')
    if checkpoint.get('val_loss') != state['best_val_loss']:
        raise ValueError(f'Best checkpoint validation loss does not match saved state: {path}; '
                         'a different run or incomplete checkpoint write may have replaced it')
    return checkpoint


def save_state(path, model, optimizer, scaler, state, args, manifest, generator):
    local_rng = capture_rng(generator)
    rng_states = [None] * dist.get_world_size() if existing.is_distributed() else [local_rng]
    if existing.is_distributed():
        dist.all_gather_object(rng_states, local_rng)
    # The complete (potentially very large) per-audio audit is kept in the JSON
    # sidecar once. Checkpoints retain its digest and compact provenance so that
    # training does not serialize 160,000 metadata records after every epoch.
    manifest_reference = {key: value for key, value in manifest.items() if key != 'splits'}
    manifest_reference['splits'] = {
        name: {'metadata_path': split['metadata_path'],
               'metadata_sha256': split['metadata_sha256'],
               'sample_count': len(split['records'])}
        for name, split in manifest['splits'].items()
    }
    checkpoint = dict(state, model_name='CTSNet',
                      model_state_dict=existing.model_state_dict_for_save(model),
                      optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                      args=vars(args).copy(), reproducibility_manifest=manifest_reference,
                      rng_states=rng_states)
    if existing.is_main_process():
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(checkpoint, temp)
        os.replace(temp, path)
    return checkpoint


def initial_stage_state(stage='me', epoch=0):
    return {'stage': stage, 'epoch': epoch, 'stage_epoch': 0,
            'best_val_loss': float('inf'), 'train_loss': None, 'val_loss': None,
            'lr_reducer_state': {'stale_lr_epochs': 0, 'stale_stop_epochs': 0},
            'me_converged': False}


def deviations(args, world_size):
    fields = ('batch_size', 'grad_accum_steps', 'segment_seconds', 'learning_rate',
              'grad_clip', 'weight_decay', 'norm_type', 'power', 'train_lr_patience',
              'train_lr_factor', 'train_lr_min_delta')
    result = [f'{name}={getattr(args, name)} (paper preset: {PAPER_DEFAULTS[name]})'
              for name in fields if getattr(args, name) != PAPER_DEFAULTS[name]]
    if args.early_stop_patience != 5:
        result.append(f'early_stop_patience={args.early_stop_patience} (paper: 5)')
    if args.joint_me_learning_rate != 1e-4 or args.joint_cs_learning_rate != 1e-3:
        result.append('Joint learning rates differ from paper')
    if world_size != 1:
        result.append(f'Distributed training world_size={world_size}; paper uses one GPU')
    if args.use_amp == 'yes' or args.model_amp == 'yes' or args.allow_tf32 == 'yes':
        result.append('Reduced precision enabled; numerical departure from FP32 reference')
    if args.hash_audio != 'yes':
        result.append('Audio contents not hashed; provenance only covers paths/size/metadata')
    if args.target_ref_mic != 0:
        result.append(f'target_ref_mic={args.target_ref_mic} paired with mixture channel zero; verify acoustic alignment')
    return result


def main(argv=None):
    args = parse_args(argv)
    device, gpu_ids, world_size, rank = setup_runtime(args)
    generator = torch.Generator().manual_seed(args.seed + rank)
    datasets, loaders, sampler = build_loaders(args, generator)
    if existing.is_main_process():
        print('Auditing data and code hashes before training...', flush=True)
    manifest = make_manifest(args, datasets, world_size)
    checkpoint_dir, best_dir = Path(args.checkpoint_dir), Path(args.best_dir)
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    best_me_path = best_dir / 'best_me.pt'
    best_joint_path = best_dir / 'best_model.pt'
    if args.resume == 'no' and latest_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {latest_path}; choose a new --checkpoint-dir')
    checkpoint = None
    if args.resume == 'yes' and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        check_resume(checkpoint, args, manifest)
        if checkpoint['stage_epoch'] > 0:
            required_best = best_me_path if checkpoint['stage'] == 'me' else best_joint_path
            load_verified_best(required_best, checkpoint, args, manifest)
    if checkpoint is None and any(path.exists() for path in (best_me_path, best_joint_path, checkpoint_dir / 'best_model.pt')):
        raise FileExistsError('Existing best checkpoint found without a matching resumable run; choose new output directories')
    if existing.is_main_process():
        atomic_json(checkpoint_dir / 'reproducibility_manifest.json', manifest)
        atomic_json(checkpoint_dir / 'run_config.json', vars(args))
        print(f'CTSNet: device={device}, samples train/val={len(datasets["train"])}/{len(datasets["val"])}, '
              f'effective batch={args.batch_size * args.grad_accum_steps * world_size}')
        print(f'Total budget={args.num_epochs}; ME cap={args.pretrain_max_epochs} '
              '(stage allocation is not specified in the paper)')
        for item in deviations(args, world_size):
            print('[PROTOCOL DEVIATION] ' + item)
    core = CTSNet(is_causal=args.is_causal == 'yes', norm_type=args.norm_type,
                  return_auxiliary=True).to(device)
    state = initial_stage_state()
    if checkpoint:
        state = {key: checkpoint[key] for key in state}
        existing.load_model_state_flexible(core, checkpoint['model_state_dict'])
    optimizer = make_optimizer(core, state['stage'], args)
    model = wrap_model(core, device, gpu_ids, args)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda' and args.use_amp == 'yes')
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        restore_rng(checkpoint['rng_states'][rank], generator)
    else:
        # Identical initial model on ranks; independent deterministic crop streams.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
    writer = SummaryWriter(args.log_dir) if existing.is_main_process() else None
    summary = {'model_name': 'CTSNet', 'manifest_digest': manifest['digest'],
               'protocol_deviations': deviations(args, world_size),
               'stage_budget_assumption': 'ME cap configurable; remaining total budget goes to joint stage',
               'validation_stopping_interpretation': 'consecutive non-improvements against best stage loss',
               'validation_extent': 'training validation loss uses leading segment_seconds; evaluate_cts scores complete utterances',
               'normalization_note': 'IN is literal paper instance normalization; causal convolutions do not establish streaming causality',
               'status': 'running'}
    try:
        while state['epoch'] < args.num_epochs:
            if (state['stage'] == 'joint' and
                    state['lr_reducer_state']['stale_stop_epochs'] >= args.early_stop_patience):
                break
            # Handle stage boundary before consuming another epoch, including a
            # resumed run that previously reached the configured ME cap.
            if state['stage'] == 'me':
                converged = state['lr_reducer_state']['stale_stop_epochs'] >= args.early_stop_patience
                exhausted = state['stage_epoch'] >= args.pretrain_max_epochs
                if converged or exhausted:
                    if not converged and args.allow_unconverged_pretrain != 'yes':
                        raise RuntimeError('ME pretraining reached its epoch cap without the validation stopping criterion. '
                                           'Resume with a larger --pretrain-max-epochs and total budget, or use '
                                           '--allow-unconverged-pretrain yes only for a declared debugging run.')
                    if existing.is_distributed():
                        dist.barrier()
                    best_me = load_verified_best(best_me_path, state, args, manifest, device)
                    existing.load_model_state_flexible(core, best_me['model_state_dict'])
                    state = initial_stage_state('joint', state['epoch'])
                    state['me_converged'] = converged
                    optimizer = make_optimizer(core, 'joint', args)
                    model = wrap_model(core, device, gpu_ids, args)
                    scaler = GradScaler('cuda', enabled=device.type == 'cuda' and args.use_amp == 'yes')
                    save_state(latest_path, model, optimizer, scaler, state, args, manifest, generator)
                    if existing.is_main_process():
                        print(f'Starting joint training from best ME checkpoint; ME converged={converged}', flush=True)
            state['epoch'] += 1
            state['stage_epoch'] += 1
            if sampler is not None:
                sampler.set_epoch(state['epoch'])
            stage = state['stage']
            loss_fn = functools.partial(cts_loss, stage=stage, alpha=0.5, lambda_me=0.1)
            loop_kwargs = dict(model=model, optimizer=optimizer, scaler=scaler,
                               device=device, gpu_ids=gpu_ids, args=args, loss_fn=loss_fn,
                               feature_builder=build_stft_batch, correct_accumulation=True,
                               loss_weighting='frames', fail_nonfinite_grad=True,
                               forward_with_lengths=True)
            if existing.is_main_process():
                print(f'\nEpoch {state["epoch"]}/{args.num_epochs}, stage={stage}, '
                      f'stage epoch={state["stage_epoch"]}', flush=True)
            train_loss = existing.run_epoch(loader=loaders['train'], training=True, **loop_kwargs)
            val_loss = existing.run_epoch(loader=loaders['val'], training=False, **loop_kwargs)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError('Epoch loss is non-finite; no valid checkpoint produced')
            state['train_loss'], state['val_loss'] = train_loss, val_loss
            improved = val_loss < state['best_val_loss'] - args.train_lr_min_delta
            reducer = state['lr_reducer_state']
            if improved:
                state['best_val_loss'] = val_loss
                reducer['stale_lr_epochs'] = reducer['stale_stop_epochs'] = 0
            else:
                reducer['stale_lr_epochs'] += 1
                reducer['stale_stop_epochs'] += 1
            if reducer['stale_lr_epochs'] >= args.train_lr_patience:
                for group in optimizer.param_groups:
                    group['lr'] = max(args.min_learning_rate, group['lr'] * args.train_lr_factor)
                reducer['stale_lr_epochs'] = 0
            if improved:
                best_path = best_me_path if stage == 'me' else best_joint_path
                save_state(best_path, model, optimizer, scaler, state, args, manifest, generator)
                if stage == 'joint' and existing.is_main_process():
                    existing.maybe_copy_best_for_infer(best_path, checkpoint_dir)
            save_state(latest_path, model, optimizer, scaler, state, args, manifest, generator)
            if state['epoch'] % args.save_every == 0:
                save_state(checkpoint_dir / f'model_epoch_{state["epoch"]}.pt',
                           model, optimizer, scaler, state, args, manifest, generator)
            if writer:
                writer.add_scalar(f'{stage}/train_loss', train_loss, state['epoch'])
                writer.add_scalar(f'{stage}/val_loss', val_loss, state['epoch'])
                for group in optimizer.param_groups:
                    writer.add_scalar(f'{stage}/lr_{group["name"]}', group['lr'], state['epoch'])
                writer.flush()
            if existing.is_main_process():
                print(f'{stage}: train={train_loss:.8f} val={val_loss:.8f} '
                      f'best={state["best_val_loss"]:.8f} lr={[g["lr"] for g in optimizer.param_groups]}')
                atomic_json(checkpoint_dir / 'training_status.json', dict(summary, **state))
            gc.collect()
            if device.type == 'cuda' and args.empty_cache_each_epoch == 'yes':
                torch.cuda.empty_cache()
            if stage == 'joint' and reducer['stale_stop_epochs'] >= args.early_stop_patience:
                break
        if state['stage'] != 'joint' or state['stage_epoch'] < 1:
            raise RuntimeError('Total epoch budget exhausted before joint training; reproduction is incomplete')
        summary.update(status='finished', epoch=state['epoch'], stage=state['stage'],
                       me_converged=state['me_converged'], best_joint_val_loss=state['best_val_loss'],
                       best_checkpoint=str(best_joint_path.resolve()))
        if not state['me_converged']:
            summary['protocol_deviations'].append('Joint stage started before ME convergence by explicit debug override')
        if existing.is_main_process():
            atomic_json(checkpoint_dir / 'training_summary.json', summary)
            print(f'Training finished. Best joint checkpoint: {best_joint_path.resolve()}')
    except Exception as error:
        if existing.is_main_process():
            atomic_json(checkpoint_dir / 'training_summary.json',
                        dict(summary, status='failed', error=str(error), epoch=state['epoch'], stage=state['stage']))
        raise
    finally:
        if writer:
            writer.close()
        if existing.is_distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
