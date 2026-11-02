"""Train causal prefix covariances through the original frontend, loss and loop."""
import argparse
from pathlib import Path
import sys

import torch

import train_light_causal_fullband as fullband
from experiments.compiled_training_norm import compiled_training_norms
from experiments.compiled_training_retention import compiled_training_retention
from experiments.batched_quality_training import batched_quality_training
from light_candidate_runtime import (COVARIANCE_CANDIDATE,WIDE_CONDITIONED_CANDIDATE,
    candidate_metadata,read_candidate_checkpoint,select_model_factory)

ROOT=Path(__file__).resolve().parent
WARM_FIELDS=('covariance_warmstart_checkpoint','covariance_warmstart_sha256',
    'covariance_warmstart_epoch','covariance_sample_epoch_offset','pmsqe_weight')


def parse_args(argv=None):
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--warmstart')
    selected,remaining=selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    causal=fullband.causal
    args=causal.trainer.parse_args(remaining,defaults=dict(
        checkpoint_dir=f'./checkpoints_{COVARIANCE_CANDIDATE}',best_dir=f'./bestmodels_{COVARIANCE_CANDIDATE}',
        log_dir=f'./logs_{COVARIANCE_CANDIDATE}',num_epochs=30,learning_rate=.001,batch_size=2,
        grad_accum_steps=2,segment_seconds=6,save_every=1,train_lr_patience=3,grad_clip=5.,norm_type='LN',
        use_amp='no',model_amp='no',allow_tf32='no',stop_on_non_finite='yes',skip_non_finite_batches='no'))
    if (args.sample_rate,args.n_fft,args.hop_length,args.win_length,args.power,args.num_mics,args.target_ref_mic)!=(16000,512,160,320,.5,8,0):
        raise ValueError('Keep the original16kHz eight-microphone reference0 frontend')
    if args.is_causal!='yes' or args.norm_type!='LN' or args.parallel_mode=='dp':
        raise ValueError('Use causal LayerNorm with none or ddp')
    if any(value!='no' for value in (args.use_amp,args.model_amp,args.allow_tf32)):
        raise ValueError('Keep FP32 parameters,frontend,GRUs,loss,inference and complex64 covariances')
    if args.stop_on_non_finite!='yes' or args.skip_non_finite_batches!='no':
        raise ValueError('Nonfinite updates must fail instead of skipping records')
    latest=Path(args.checkpoint_dir).expanduser()/'checkpoint_latest.pt';existing=None
    if latest.exists():
        if args.resume!='yes':raise FileExistsError('Use --resume yes or new output directories')
        existing,_=read_candidate_checkpoint(latest,COVARIANCE_CANDIDATE)
        source_name=existing['args'].get('covariance_warmstart_checkpoint')
        if not source_name:raise ValueError('Resume lacks its complete causal source')
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve()!=Path(source_name):
            raise ValueError('Covariance warm-start differs from the saved run')
    else:
        if not selected.warmstart:raise ValueError('A new covariance run requires a completed wide --warmstart')
        source_name=selected.warmstart
    path=Path(source_name).expanduser().resolve();source,_=read_candidate_checkpoint(path,WIDE_CONDITIONED_CANDIDATE)
    offset=source['args'].get('capacity_sample_epoch_offset')
    if (int(source['epoch'])<1 or source['args']['seed']!=args.seed or source.get('diagnostic_only',False)
            or source['args'].get('diagnostic_only',False) or isinstance(offset,bool) or not isinstance(offset,int) or offset<0):
        raise ValueError('Use a completed wide checkpoint with matching seed and sampler provenance')
    for name in causal.quality.SOURCE_FIELDS:
        if getattr(args,name)!=source['args'][name]:raise ValueError(f'Covariance warm-start setting differs: {name}')
    for name,value in candidate_metadata(COVARIANCE_CANDIDATE).items():setattr(args,name,value)
    args.covariance_warmstart_checkpoint=str(path);args.covariance_warmstart_sha256=causal.quality.sha256_file(path)
    args.covariance_warmstart_epoch=int(source['epoch']);args.covariance_sample_epoch_offset=offset+int(source['epoch'])
    args.pmsqe_weight=.1;args.causal_training_objective='spectral_mrstft_waveform_pmsqe_v1'
    args.causal_training_precision='FP32 parameters,frontend,GRUs,loss,inference; frozen causal source FP32 eval; complex64 prefix covariances'
    files=('train_light_causal_covariance.py','light_candidate_runtime.py','train_light_causal_fullband.py',
        'train_light_causal.py','train_light.py','train_light_causal_pmsqe.py','light_quality_loss.py',
        'experiments/causal_pmsqe_objective.py','experiments/compiled_training_norm.py','experiments/compiled_training_retention.py',
        'experiments/batched_quality_training.py','third_party/asteroid_pmsqe/pmsqe.py',
        'third_party/asteroid_pmsqe/bark_matrix_16k.mat','third_party/asteroid_pmsqe/PROVENANCE.json')
    args.causal_training_implementation_sha256={name:causal.quality.sha256_file(ROOT/name) for name in files}
    if existing:
        for name in (*WARM_FIELDS,'causal_training_objective','causal_training_precision','causal_training_implementation_sha256'):
            if existing['args'].get(name)!=getattr(args,name):raise ValueError(f'Covariance resume {name} differs')
    causal.quality._validate_output_paths(args,path)
    return args,{name:source[name] for name in ('epoch','args','model_state_dict')}


def main(argv=None):
    args,source=parse_args(argv);causal=fullband.causal
    parser,initializer,sampler=fullband.parse_args,causal.initialize_checkpoint,causal.trainer.DistributedSampler
    def initialize(settings,unused=None):
        directory=Path(settings.checkpoint_dir)
        if (directory/'checkpoint_latest.pt').exists():return
        for path in (directory,Path(settings.best_dir)):
            if path.exists() and any(path.iterdir()):raise FileExistsError(f'New covariance outputs must be empty: {path}')
        torch.manual_seed(settings.seed)
        with select_model_factory(causal.trainer,settings.structure_candidate):
            model=causal.trainer.create_model(settings,torch.device('cpu'))
        if not all(torch.isfinite(value).all() for value in source['model_state_dict'].values()):
            raise ValueError('The complete causal source contains nonfinite weights')
        model.base.load_state_dict(source['model_state_dict'],strict=True)
        assert all(not p.requires_grad for p in model.base.parameters())
        assert all(p.requires_grad for name,p in model.named_parameters() if not name.startswith('base.'))
        assert all(p.dtype==torch.float32 for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
        state=dict(epoch=0,best_val_loss=float('inf'),model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),scaler_state_dict={},lr_reducer_state={},args=vars(settings).copy())
        for name in ('model_epoch_0.pt','checkpoint_latest.pt'):causal.trainer.save_checkpoint(directory/name,state)
    class AlignedSampler(sampler):
        def set_epoch(self,epoch):super().set_epoch(epoch+args.covariance_sample_epoch_offset if self.shuffle else epoch)
    fullband.parse_args=lambda *unused,**kwargs:args
    causal.initialize_checkpoint,causal.trainer.DistributedSampler=initialize,AlignedSampler
    try:
        with compiled_training_norms(),compiled_training_retention(),batched_quality_training():
            return fullband.main([])
    finally:
        fullband.parse_args,causal.initialize_checkpoint,causal.trainer.DistributedSampler=parser,initializer,sampler


if __name__=='__main__':raise SystemExit(main())
