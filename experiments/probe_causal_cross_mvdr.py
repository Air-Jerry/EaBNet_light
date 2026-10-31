"""CPU-only causal MVDR screen using the network's complex cross spectrum."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import torch
from pesq import pesq

from evaluate_light import load_model
from experiments.probe_causal_mvdr import past_mvdr
from light_causal_spatial_model import CausalSpatialEnhancer
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


def cross_mvdr(mixture, estimate, state=None, alpha=.98):
    """Estimate RTF from causal E[x*s.conj()], with mixture-only noise covariance."""
    x=np.asarray(mixture,dtype=np.complex128)
    s=np.asarray(estimate,dtype=np.complex128)
    if x.ndim!=3 or min(x.shape)<1 or s.shape!=x.shape[:2] or not 0<=alpha<1:
        raise ValueError('Expected nonempty (T,F,M), matching estimate (T,F), stable alpha')
    if not np.isfinite(x).all() or not np.isfinite(s).all():
        raise ValueError('Non-finite cross-MVDR input')
    previous={} if state is None else state
    power=np.abs(s)**2
    mask=(power/(power+np.abs(x[...,0]-s)**2+1e-10)).clip(.001,.999)
    outer=x[..., :,None]*x[...,None,:].conj()
    terms={'noise':(1-mask)[...,None,None]*outer,'cross':x*s[...,None].conj()}
    values={};updated={}
    for name,value in terms.items():
        shape=(1,*value.shape[1:])
        initial=np.asarray(previous.get(name,np.zeros(shape,dtype=np.complex128)))
        if initial.shape!=shape or not np.isfinite(initial).all():
            raise ValueError('Cross-MVDR state differs from this stream')
        values[name],updated[name]=lfilter([1-alpha],[1,-alpha],value,axis=0,zi=initial)
    channels=x.shape[-1]
    eye=np.eye(channels)
    noise=values['noise']
    loading=1e-3*np.trace(noise,axis1=-2,axis2=-1).real/channels+1e-8
    noise=noise+loading[...,None,None]*eye
    cross=values['cross'];reference=cross[...,0:1]
    valid=np.abs(reference[...,0])>1e-10
    rtf=cross/np.where(valid[...,None],reference,1)
    inverse=np.linalg.solve(noise,rtf[...,None])[...,0]
    denominator=(rtf.conj()*inverse).sum(axis=-1).real.clip(1e-8)
    weights=inverse/denominator[...,None]
    result=np.where(valid,(weights.conj()*x).sum(axis=-1),s)
    if not np.isfinite(result).all():raise ValueError('Non-finite cross-MVDR output')
    return result,updated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    os.nice(15)
    cores=os.sched_getaffinity(0).intersection({16,18})
    if cores:os.sched_setaffinity(0,cores)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve();root=Path(args.data_dir).resolve()
    source_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    protected={p:hashlib.sha256(p.read_bytes()).digest()
               for index in range(1,9) for p in (root/f'mixture_{index}.wav',root/f'target_{index}.wav')}
    model,_=load_model(checkpoint,torch.device('cpu'),'causal_recurrent_unet65_v1')
    objective=QualityObjective();rows=[];start=time.monotonic()
    report=dict(status='running',scope='Eight fixed6s training crops; CPU inference screen, no training or generalization claim',
        source_checkpoint_sha256=source_sha,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formula_source='https://docs.pytorch.org/audio/2.5.0/generated/torchaudio.functional.mvdr_weights_rtf.html',
        covariance_uses_current_and_past_frames_only=True,mask_uses_clean_reference=False,
        target_gain_matching=False,external_normalization=False,private_audio_hashes_exported=False,
        source_candidate='causal_recurrent_unet65_v1',source_optimizer_updates=0,
        num_mics=8,reference_mic=0,alphas=[.9,.98,.995],physical_blend_weights=[.25,.5,1.0],rows=rows)
    with torch.inference_mode():
        for index in range(1,9):
            mix,sr=sf.read(root/f'mixture_{index}.wav',dtype='float32',always_2d=True)
            target,tsr=sf.read(root/f'target_{index}.wav',dtype='float32',always_2d=True)
            assert sr==tsr==16000 and len(mix)==len(target)==96000 and mix.shape[1]>=8
            mixture=torch.from_numpy(mix[:,:8].copy())[None]
            features,_=build_stft_batch(mixture,mixture[:,:,0],512,160,320,.5,torch.device('cpu'))
            estimate=model(features)
            x=torch.view_as_complex(CausalSpatialEnhancer.decompress(features[0]).contiguous()).numpy()
            s=torch.view_as_complex(CausalSpatialEnhancer.decompress(estimate[0].permute(1,2,0).contiguous()).contiguous()).numpy()
            metrics={'baseline':float(pesq(16000,target[:,0],objective.reconstruct(estimate,96000)[0].numpy(),'wb'))}
            for alpha in report['alphas']:
                cross,_=cross_mvdr(x,s,alpha=alpha)
                mask=np.abs(s)**2/(np.abs(s)**2+np.abs(x[...,0]-s)**2+1e-10)
                standard,_=past_mvdr(x,mask,alpha=alpha)
                for method,value in [('cross',cross),('souden',standard['souden'])]:
                    for weight in report['physical_blend_weights']:
                        blended=s+weight*(value-s)
                        raw=torch.view_as_real(torch.from_numpy(blended).to(torch.complex64))
                        compressed=CausalSpatialEnhancer.compress(raw).permute(2,0,1)[None]
                        wave=objective.reconstruct(compressed,96000)[0].numpy()
                        metrics[f'{method}_alpha{alpha}_blend{weight}']=float(pesq(16000,target[:,0],wave,'wb'))
            rows.append(dict(sample_index=index,pesq=metrics))
            (output/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(processed_samples=index,baseline=metrics['baseline'])),flush=True)
    means={name:float(np.mean([row['pesq'][name] for row in rows])) for name in rows[0]['pesq']}
    best=max((name for name in means if name!='baseline'),key=means.get)
    assert all(hashlib.sha256(p.read_bytes()).digest()==value for p,value in protected.items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_sha
    report.update(status='complete',completed_samples=8,mean_pesq=means,best_screened_variant=best,
        best_gain_over_baseline=means[best]-means['baseline'],elapsed_seconds=time.monotonic()-start,
        private_audio_occurrences_verified_unchanged=16,source_checkpoint_unchanged=True)
    (output/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',baseline=means['baseline'],best=best,best_pesq=means[best],gain=report['best_gain_over_baseline'])),flush=True)


if __name__=='__main__':main()
