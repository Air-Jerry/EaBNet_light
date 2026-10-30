"""Short, isolated two-GPU timing run; never writes a training checkpoint."""

import argparse
import cProfile
from datetime import datetime
import json
import os
from pathlib import Path
import pstats
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

import train_light as trainer
from light_candidate_runtime import select_model_factory
from light_metric_loss import MetricObjective
from light_quality_model import QUALITY_CANDIDATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--performance", choices=("standard", "overlap"), default="standard")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    selected = parser.parse_args()
    if selected.steps < 4 or selected.steps % 4:
        raise ValueError("Steps must be a positive multiple of four")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(selected.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**state["args"])
    args.batch_size = selected.batch_size
    args.grad_accum_steps = 4 // selected.batch_size
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "yes"
    torch.manual_seed(args.seed)
    with select_model_factory(trainer, QUALITY_CANDIDATE):
        model = trainer.create_model(args, device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    model = DistributedDataParallel(model, device_ids=[rank])
    kwargs = {} if selected.performance == "standard" else {"accelerated": True}
    objective = MetricObjective(device, weight=args.metric_weight,
                                discriminator_lr=args.metric_discriminator_lr, **kwargs)
    objective.load_state_dict(state["metric_state"])
    objective.training = True
    dataset = trainer.EnhancementDataset(Path(args.train_dir), sample_rate=args.sample_rate,
        target_ref_mic=args.target_ref_mic, num_mics=args.num_mics,
        segment_samples=96000, random_crop=False)
    ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(819))
    ids = ids[:(selected.steps + 16) * dist.get_world_size()]
    ids = ids[dist.get_rank()::dist.get_world_size()].tolist()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    initial_g = int(next(iter(optimizer.state.values()))["step"])
    initial_d = objective.updates
    rows = []
    for phase, indices in (("warmup", ids[:16]), ("timed", ids[16:])):
        loader = DataLoader(Subset(dataset, indices), batch_size=selected.batch_size, num_workers=4,
            pin_memory=True, collate_fn=trainer.collate_batch)
        objective.reset_stats()
        objective.reset_metric_stats()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        profile = cProfile.Profile()
        started = datetime.now().isoformat()
        start = time.perf_counter()
        profile.enable()
        extra = {} if selected.performance == "standard" else {"defer_ddp_sync": True}
        trainer.run_epoch(model, loader, optimizer, scaler, device, [rank], args,
            training=True, loss_fn=objective, feature_builder=objective.build_features,
            correct_accumulation=True, loss_weighting="frames", fail_nonfinite_grad=True,
            forward_with_lengths=True, **extra)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        finished = datetime.now().isoformat()
        profile.disable()
        stats = pstats.Stats(profile)
        top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:35]
        rows.append(dict(phase=phase, started_local=started, finished_local=finished,
            seconds=elapsed, samples_per_second_local=len(indices)/elapsed,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            profile=[dict(function=f"{key[0]}:{key[1]}:{key[2]}", calls=value[1],
                          self_seconds=value[2], cumulative_seconds=value[3]) for key,value in top]))
    assert objective.updates - initial_d == selected.steps + 16
    assert int(next(iter(optimizer.state.values()))["step"]) - initial_g == (selected.steps + 16)//4
    for value in objective.discriminator.state_dict().values():
        other = value.clone()
        dist.broadcast(other, src=0)
        assert torch.equal(value, other), "Discriminator ranks diverged"
    for key, value in model.module.state_dict().items():
        assert torch.isfinite(value).all(), key
        if key.startswith("base."):
            assert torch.equal(value.cpu(), state["model_state_dict"][key]), key
    result = dict(rank=rank, performance=selected.performance, measurements=rows,
        source_epoch=state["epoch"], source_checkpoint=str(Path(selected.checkpoint).resolve()),
        batch_size_per_gpu=selected.batch_size,
        sample_ids=ids, generator_updates=(selected.steps + 16)//4,
        discriminator_updates=selected.steps + 16, effective_global_batch=8,
        discriminator_ranks_identical=True, frozen_backbone_unchanged=True,
        metric_stats=objective.metric_stats())
    output = Path(selected.output)
    output.mkdir(parents=True, exist_ok=True)
    tag = selected.performance if selected.batch_size == 1 else f"{selected.performance}_b{selected.batch_size}"
    (output/f"{tag}_rank{rank}.json").write_text(json.dumps(result, indent=2))
    if rank == 0:
        print(json.dumps(rows[-1], indent=2), flush=True)
    if hasattr(objective, "close"):
        objective.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
