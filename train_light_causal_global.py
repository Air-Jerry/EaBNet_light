"""Warm-start a causal global-frequency model through the existing training loop."""

import argparse
from pathlib import Path
import sys

import torch

import train_light_causal as causal
from light_candidate_runtime import read_candidate_checkpoint,select_model_factory
from light_causal_fast_model import FAST_CANDIDATE
from light_causal_global_model import GLOBAL_CANDIDATE


WARM_FIELDS=('global_warmstart_checkpoint','global_warmstart_sha256','global_warmstart_epoch','global_sample_epoch_offset')


def aligned_sampler_type(base,offset):
    class AlignedSampler(base):
        def set_epoch(self,epoch):
            super().set_epoch(epoch+offset if self.shuffle else epoch)
    return AlignedSampler


def parse_args(argv=None):
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--warmstart')
    selected,remaining=selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    args=causal.parse_args(remaining,candidate_id=GLOBAL_CANDIDATE,
        defaults={'batch_size':4,'grad_accum_steps':1,'learning_rate':1e-4,'num_epochs':6},entry_file=__file__)
    latest=Path(args.checkpoint_dir).expanduser()/'checkpoint_latest.pt'
    existing=None
    if latest.exists():
        existing,_=read_candidate_checkpoint(latest,GLOBAL_CANDIDATE,args)
        saved=existing['args']
        if not all(name in saved for name in WARM_FIELDS):raise ValueError('Global checkpoint lacks warm-start provenance')
        source_name=saved['global_warmstart_checkpoint']
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve()!=Path(source_name):
            raise ValueError('Global warm-start differs from the saved run')
    else:
        if not selected.warmstart:raise ValueError('A new global model requires --warmstart from a completed causal fast model')
        source_name=selected.warmstart
    path=Path(source_name).expanduser().resolve()
    source,_=read_candidate_checkpoint(path,FAST_CANDIDATE)
    if int(source['epoch'])<1:raise ValueError('Warm-start must contain completed real training')
    for name in causal.quality.SOURCE_FIELDS:
        if getattr(args,name)!=source['args'][name]:raise ValueError(f'Global frontend/model setting differs from source: {name}')
    args.global_warmstart_checkpoint=str(path)
    args.global_warmstart_sha256=causal.quality.sha256_file(path)
    args.global_warmstart_epoch=int(source['epoch'])
    args.global_sample_epoch_offset=int(source['epoch'])
    if existing is not None:
        for name in WARM_FIELDS:
            if existing['args'][name]!=getattr(args,name):raise ValueError(f'Global resume {name} differs')
    for output in (args.checkpoint_dir,args.best_dir,args.log_dir):
        directory=Path(output).expanduser().resolve()
        if directory==path.parent or directory in path.parents or path.parent in directory.parents:
            raise ValueError('Global outputs must be separate from the source checkpoint')
    return args,source


def main(argv=None):
    args,source=parse_args(argv)
    parser,initializer=causal.parse_args,causal.initialize_checkpoint
    sampler=causal.trainer.DistributedSampler

    def initialize(settings,unused=None):
        directory=Path(settings.checkpoint_dir)
        latest=directory/'checkpoint_latest.pt'
        if latest.exists():return
        for output in (directory,Path(settings.best_dir)):
            if output.exists() and any(output.iterdir()):raise FileExistsError(f'New global model requires empty output: {output}')
        torch.manual_seed(settings.seed)
        with select_model_factory(causal.trainer,GLOBAL_CANDIDATE):
            model=causal.trainer.create_model(settings,torch.device('cpu'))
        target=model.state_dict();warm=source['model_state_dict']
        extra=set(target)-set(warm)
        assert extra and all(name.startswith('layers.8.') for name in extra),extra
        assert not set(warm)-set(target)
        for name,value in warm.items():
            assert target[name].shape==value.shape,name
            target[name]=value
        model.load_state_dict(target,strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
        state=dict(epoch=0,best_val_loss=float('inf'),model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),scaler_state_dict={},lr_reducer_state={},args=vars(settings).copy())
        for name in ('model_epoch_0.pt','checkpoint_latest.pt'):
            causal.trainer.save_checkpoint(directory/name,state)
        print(f'Causal global warm-start epoch {settings.global_warmstart_epoch}; zero new residual; fresh Adam',flush=True)

    causal.parse_args=lambda *unused,**keywords:args
    causal.initialize_checkpoint=initialize
    causal.trainer.DistributedSampler=aligned_sampler_type(sampler,args.global_sample_epoch_offset)
    try:return causal.main([],candidate_id=GLOBAL_CANDIDATE,entry_file=__file__)
    finally:
        causal.parse_args,causal.initialize_checkpoint=parser,initializer
        causal.trainer.DistributedSampler=sampler


if __name__=='__main__':raise SystemExit(main())
