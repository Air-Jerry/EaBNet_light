"""Fixed real eight-sample CPU fitting comparison, never a generalization score."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
from torch import nn
from pesq import pesq

from light_causal_fast_model import (CausalUNetSpatial, CausalFrequencyConv,
    ActivationPReLU, activation_norms)
from light_causal_spatial_model import CausalSpatialLayer
from light_quality_loss import QualityObjective


class ResolutionSpatial(CausalUNetSpatial):
    def __init__(self):
        nn.Module.__init__(self)
        self.M, self.context = 8, 128
        self.stem = CausalFrequencyConv(16, 32, frequency_kernel=5)
        widths = (32, 48, 96)
        self.encoders = nn.ModuleList([CausalFrequencyConv(a, b, 2, 5)
            for a, b in zip(widths[:-1], widths[1:])])
        self.layers = nn.ModuleList()
        full = None
        for _ in range(4):
            layer = CausalSpatialLayer(96, 192, 65, 4, 128, full=full)
            layer.fconv1[-1] = ActivationPReLU(96)
            layer.fconv2[-1] = ActivationPReLU(96)
            activation_norms(layer)
            self.layers.append(layer)
            full = layer.full
        reverse = tuple(reversed(widths))
        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(a, b, (1, 4), stride=(1, 2), padding=(0, 1))
            for a, b in zip(reverse[:-1], reverse[1:])])
        self.decoders = nn.ModuleList([CausalFrequencyConv(2*b, b) for b in reverse[1:]])
        self.beamformer = nn.Conv2d(32, 16, 1)
        nn.init.zeros_(self.beamformer.weight)
        nn.init.zeros_(self.beamformer.bias)
        self.use_bfloat16_training = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--warmstart', help='Fit only the current17-group model from this validated checkpoint; fresh Adam')
    parser.add_argument('--refiner',action='store_true',help='Screen a causal GRU branch with cached, frozen causal backbone')
    parser.add_argument('--band',action='store_true',help='Compare the fully trainable global model and full-frequency recurrent beamforming')
    parser.add_argument('--global-capacity',action='store_true',help='Compare fully trainable recurrent65 models with and without a larger global temporal block')
    parser.add_argument('--multi-frame',action='store_true',help='Compare a fully trainable recurrent65 model with four strictly past beamforming taps')
    parser.add_argument('--metric-objective',action='store_true',help='Compare physical loss and training-only real WB-PESQ supervision on the same recurrent65 model')
    parser.add_argument('--tcn-mapping',action='store_true',help='Compare recurrent65 and a trainable full-frequency causal TCN mapping branch')
    parser.add_argument('--spatial-cues',action='store_true',help='Compare recurrent65 with a trainable current-frame IPD and magnitude-ratio stem')
    parser.add_argument('--frequency-fourier',action='store_true',help='Compare the trained TCN with global frame-local frequency Fourier mixing')
    parser.add_argument('--learning-rate',type=float,default=.001)
    args = parser.parse_args()
    if args.updates < 1:raise ValueError('Use a positive update budget')
    if args.refiner and not args.warmstart:raise ValueError('The refiner probe requires a trained causal warm-start')
    if args.band and (not args.warmstart or args.refiner):raise ValueError('Band comparison requires a global warm-start and no frozen refiner')
    if args.global_capacity and (not args.warmstart or args.band or args.refiner):raise ValueError('Global capacity comparison requires a recurrent65 warm-start and no other branch')
    if args.multi_frame and (not args.warmstart or args.band or args.refiner or args.global_capacity):raise ValueError('Multi-frame comparison requires a recurrent65 warm-start and no other branch')
    if args.metric_objective and (not args.warmstart or args.band or args.refiner or args.global_capacity or args.multi_frame):raise ValueError('Metric comparison requires a recurrent65 warm-start and no other branch')
    if args.tcn_mapping and (not args.warmstart or args.band or args.refiner or args.global_capacity or args.multi_frame or args.metric_objective):raise ValueError('TCN comparison requires a recurrent65 warm-start and no other branch')
    if args.spatial_cues and (not args.warmstart or args.band or args.refiner or args.global_capacity or args.multi_frame or args.metric_objective or args.tcn_mapping):raise ValueError('Spatial-cue comparison requires a recurrent65 warm-start and no other branch')
    if args.frequency_fourier and (not args.warmstart or args.band or args.refiner or args.global_capacity or args.multi_frame or args.metric_objective or args.tcn_mapping or args.spatial_cues):raise ValueError('Frequency-Fourier comparison requires a TCN warm-start and no other branch')
    if not 0 < args.learning_rate < float('inf'):raise ValueError('Use a positive finite learning rate')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root, output = Path(args.data_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    objective = QualityObjective()
    mixture, target, hashes = [], [], []
    for i in range(1,9):
        m, sr = sf.read(root/f'mixture_{i}.wav', dtype='float32', always_2d=True)
        t, tsr = sf.read(root/f'target_{i}.wav', dtype='float32', always_2d=True)
        assert sr==tsr==16000 and len(m)==len(t)==96000 and m.shape[1]>=8
        mixture.append(torch.from_numpy(m[:,:8].copy()))
        target.append(torch.from_numpy(t[:,0].copy()))
        hashes.append({name: hashlib.sha256((root/f'{name}_{i}.wav').read_bytes()).hexdigest()
                       for name in ('mixture','target')})
    mixture, target = torch.stack(mixture), torch.stack(target)
    report = dict(status='running', scope='Eight fixed real training samples, CPU fitting only; not generalization',
        seed=1234, updates=args.updates, effective_batch=2, learning_rate=args.learning_rate, segment_samples=96000,
        sample_audio_sha256=hashes, objective='Unchanged QualityObjective', precision='CPU FP32',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=[])
    if args.band or args.global_capacity or args.multi_frame or args.metric_objective or args.tcn_mapping or args.spatial_cues or args.frequency_fourier:
        report.pop('sample_audio_sha256')
        report['private_audio_hashes_exported']=False
    source = None
    if args.warmstart:
        from light_candidate_runtime import read_candidate_checkpoint
        checkpoint = Path(args.warmstart).expanduser().resolve()
        expected='causal_recurrent_unet65_v1' if (args.global_capacity or args.multi_frame or args.metric_objective or args.tcn_mapping or args.spatial_cues) else ('causal_unet_global_v1' if args.band else 'causal_unet_spatial_v1')
        if args.frequency_fourier:expected='causal_recurrent_tcn_v1'
        source, _ = read_candidate_checkpoint(checkpoint, expected)
        report['scope'] = 'Current trained causal model, eight-sample fitting capacity only; not generalization'
        report['warmstart'] = dict(checkpoint=str(checkpoint),epoch=source['epoch'],
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),optimizer='fresh Adam')
    shuffle = torch.Generator().manual_seed(1234)
    indices = torch.cat([torch.randperm(8,generator=shuffle) for _ in range((args.updates+3)//4)])
    assert len(indices)>=args.updates*2
    constructors = [('frequency17',CausalUNetSpatial)]
    if args.frequency_fourier:
        from experiments.causal_tcn_mapping import CausalTCNMapping
        from experiments.causal_frequency_fourier import CausalFrequencyFourierTCN
        constructors=[('tcn_control',CausalTCNMapping),('frequency_fourier',CausalFrequencyFourierTCN)]
        report['scope']='Two fully trainable TCN models; global frequency Fourier mixing per current frame, no time-axis FFT; same eight fixed training crops, not generalization'
        report['fourier_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_frequency_fourier.py').read_bytes()).hexdigest()
        report['tcn_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_tcn_mapping.py').read_bytes()).hexdigest()
    if args.spatial_cues:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        from experiments.causal_spatial_cues import CausalSpatialCueUNet65
        constructors=[('recurrent65_control',CausalRecurrentUNet65),('spatial_cues',CausalSpatialCueUNet65)]
        report['scope']='Two fully trainable recurrent65 models; current-frame IPD and magnitude-ratio features; same eight fixed training crops, not generalization'
        report['spatial_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_spatial_cues.py').read_bytes()).hexdigest()
    if args.tcn_mapping:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        from experiments.causal_tcn_mapping import CausalTCNMapping
        constructors=[('recurrent65_control',CausalRecurrentUNet65),('full_frequency_tcn_mapping',CausalTCNMapping)]
        report['scope']='Two fully trainable models from the same recurrent65 checkpoint; full-frequency causal TCN mapping; eight fixed training crops only, not generalization'
        report['tcn_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_tcn_mapping.py').read_bytes()).hexdigest()
    if args.metric_objective:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        constructors=[('physical_control',CausalRecurrentUNet65),('metric_supervision',CausalRecurrentUNet65)]
        report['scope']='Same fully trainable causal recurrent65 model and checkpoint; physical versus training-only real WB-PESQ supervision; eight-sample fitting only, not generalization'
        report['objective']='Control QualityObjective; comparison MetricObjective weight0.05, fresh D Adam1e-4, no inference changes'
        report['metric_implementation_sha256']=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('light_metric_loss.py').read_bytes()).hexdigest()
    if args.global_capacity:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        from experiments.causal_global_capacity import CausalGlobalCapacity
        constructors=[('recurrent65_control',CausalRecurrentUNet65),('global512_capacity',CausalGlobalCapacity)]
        report['capacity_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_global_capacity.py').read_bytes()).hexdigest()
        report['scope']='Two fully trainable recurrent65 models from the same checkpoint; fixed eight-sample fitting only, not generalization'
    if args.multi_frame:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        from experiments.causal_multiframe import CausalMultiFrame
        constructors=[('recurrent65_control',CausalRecurrentUNet65),('past4_multiframe',CausalMultiFrame)]
        report['multiframe_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_multiframe.py').read_bytes()).hexdigest()
        report['scope']='Two fully trainable recurrent65 models from the same checkpoint; four past-frame taps; fixed eight-sample fitting only, not generalization'
    if args.band:
        from light_causal_global_model import CausalGlobalSpatial
        from experiments.causal_band_beamformer import CausalBandBeamformer
        constructors=[('global_control',CausalGlobalSpatial),('full_frequency_gru',CausalBandBeamformer)]
        report['band_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('causal_band_beamformer.py').read_bytes()).hexdigest()
        report['scope']='Two fully trainable models from the same global checkpoint; eight-sample fitting only, not generalization'
    if args.refiner:
        from experiments.causal_gru_refiner import CausalGRURefiner
        constructors=[('causal_gru_refiner',CausalGRURefiner)]
        report['refiner_implementation_sha256']=hashlib.sha256(
            Path(__file__).with_name('causal_gru_refiner.py').read_bytes()).hexdigest()
    if source is None:constructors.append(('frequency65',ResolutionSpatial))
    for name, constructor in constructors:
        torch.manual_seed(1234)
        model=constructor()
        if args.spatial_cues or args.frequency_fourier:assert all(parameter.requires_grad for parameter in model.parameters())
        cached_baseline=None
        if args.refiner:
            model.backbone.load_state_dict(source['model_state_dict'],strict=True)
            model.backbone.requires_grad_(False);model.backbone.eval()
            with torch.no_grad():
                cached=[]
                for i in range(8):
                    features,_=objective.build_features(mixture[i:i+1],target[i:i+1],512,160,320,.5,'cpu',[96000])
                    cached.append(model.backbone(features))
                cached_baseline=torch.cat(cached)
            report['refiner_probe_backbone']='Frozen and cached for the exact fixed6s crops; no future frames or reference inputs'
        elif source is not None:
            values=model.state_dict()
            extra=set(values)-set(source['model_state_dict'])
            allowed=(args.band and all(key.startswith('band_') for key in extra)) or (args.global_capacity and all(key.startswith('layers.5.') for key in extra)) or (args.multi_frame and all(key.startswith('past_beamformer.') for key in extra)) or (args.tcn_mapping and all(key.startswith('mapping_') for key in extra)) or (args.spatial_cues and all(key.startswith('stem.spatial_projection.') for key in extra)) or (args.frequency_fourier and all(key.startswith('mapping_input.fourier_') for key in extra))
            assert not extra or allowed,extra
            values.update(source['model_state_dict'])
            model.load_state_dict(values,strict=True)
        metric_branch=args.metric_objective and name=='metric_supervision'
        if args.metric_objective:
            from light_metric_loss import MetricObjective
            with torch.random.fork_rng(devices=[]):
                objective=MetricObjective('cpu',accelerated=True) if metric_branch else QualityObjective()
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        start=time.perf_counter()
        rows=[]; gradient_norms=[]
        for update in range(args.updates+1):
            if update in (0,args.updates//2,args.updates):
                model.eval()
                if metric_branch:objective.training=False
                objective.reset_stats()
                with torch.no_grad():
                    scores=[]; losses=[]
                    for i in range(8):
                        features,ref=objective.build_features(mixture[i:i+1],target[i:i+1],512,160,320,0.5,'cpu',[96000])
                        estimate=(model.refine(features,cached_baseline[i:i+1])[0]
                                  if args.refiner else model(features))
                        losses.append(float(objective(estimate,ref,[601])))
                        wave=objective.reconstruct(estimate,96000)[0].numpy()
                        scores.append(float(pesq(16000,target[i].numpy(),wave,'wb')))
                row=dict(updates=update,mean_pesq=sum(scores)/8,per_sample_pesq=scores,
                         mean_loss=sum(losses)/8,loss_components=objective.stats(),
                         gradient_norm_mean=sum(gradient_norms)/max(1,len(gradient_norms)),
                         gradient_norm_max=max(gradient_norms,default=0),
                         clipped_update_fraction=sum(x>5 for x in gradient_norms)/max(1,len(gradient_norms)),
                         elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
                print(name,json.dumps(row),flush=True)
                report['active']=dict(candidate=name,completed_updates=update,results=rows)
                (output/'probe.json').write_text(json.dumps(report,indent=2))
            if update==args.updates:break
            model.train()
            if metric_branch:objective.training=True
            ids=indices[update*2:update*2+2]
            features,ref=objective.build_features(mixture[ids],target[ids],512,160,320,0.5,'cpu',[96000]*2)
            optimizer.zero_grad(set_to_none=True)
            estimate=model.refine(features,cached_baseline[ids])[0] if args.refiner else model(features)
            loss=objective(estimate,ref,[601]*2)
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(norm)
            gradient_norms.append(float(norm))
            optimizer.step()
            if update<3 or (update+1)%25==0:
                print(name,'update',update+1,'elapsed',round(time.perf_counter()-start,2),flush=True)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        if metric_branch:
            assert all(torch.isfinite(v).all() for v in objective.discriminator.state_dict().values())
            stats=objective.metric_stats()
            assert stats['valid_pesq_labels']>0 and stats['discriminator_updates_per_rank']>0
            report['training_metric_stats']=stats
            report['inference_unchanged']=True
            torch.save(objective.state_dict(),output/'training_only_metric_state.pt')
            objective.close()
        torch.save(model.state_dict(),output/f'{name}.pt')
        report['results'].append(dict(candidate=name,parameters=sum(p.numel() for p in model.parameters()),
                                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                      elapsed_seconds=time.perf_counter()-start,measurements=rows))
    report.pop('active',None)
    if args.band or args.global_capacity or args.multi_frame or args.metric_objective or args.tcn_mapping or args.spatial_cues or args.frequency_fourier:
        assert all(hashlib.sha256((root/f'{name}_{i}.wav').read_bytes()).hexdigest()==hashes[i-1][name]
            for i in range(1,9) for name in ('mixture','target'))
        report['audio_file_occurrences_unchanged']=16
    report['status']='complete'
    (output/'probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--output-dir' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output-dir')+1])
            if output.is_dir():
                result=output/'probe.json'
                report=json.loads(result.read_text()) if result.exists() else {}
                report.update(status='failed',error=f'{type(error).__name__}: {error}')
                result.write_text(json.dumps(report,indent=2))
        raise
