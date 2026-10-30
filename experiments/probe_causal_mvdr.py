"""CPU screening of past-only mask MVDR; not a deployed real-time model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import torch
from pesq import pesq

from evaluate_light import load_model,compute_si_snr_db
from light_causal_spatial_model import CausalSpatialEnhancer
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


def past_mvdr(mixture,mask,state=None,alpha=0.98):
    """Souden and principal-vector MVDR with bounded, strictly past-only PSD state."""
    mixture=np.asarray(mixture,dtype=np.complex128)
    mask=np.asarray(mask,dtype=np.float64)
    if mixture.ndim!=3 or mask.shape!=mixture.shape[:2] or not 0<=alpha<1:
        raise ValueError('Expected (frames,frequencies,microphones), matching mask and stable alpha')
    if not np.isfinite(mixture).all() or not np.isfinite(mask).all():
        raise ValueError('Non-finite MVDR input')
    mask=mask.clip(1e-3,1-1e-3)
    outer=mixture[..., :,None]*mixture[...,None,:].conj()
    previous={} if state is None else state
    psd={};updated={}
    for name,weights in (('speech',mask),('noise',1-mask)):
        initial=previous.get(name,np.zeros((1,*outer.shape[1:]),dtype=np.complex128))
        psd[name],updated[name]=lfilter([1-alpha],[1,-alpha],weights[...,None,None]*outer,axis=0,zi=initial)
    channels=mixture.shape[-1]
    eye=np.eye(channels)
    diagonal=1e-3*np.trace(psd['noise'],axis1=-2,axis2=-1).real/channels+1e-8
    noise=psd['noise']+diagonal[...,None,None]*eye
    transfer=np.linalg.solve(noise,psd['speech'])
    trace=np.trace(transfer,axis1=-2,axis2=-1).real.clip(1e-8)
    souden=transfer[..., :,0]/trace[...,None]
    vector=psd['speech'][..., :,0].copy()
    for _ in range(3):
        vector=(psd['speech']@vector[...,None])[...,0]
        vector/=np.maximum(np.linalg.norm(vector,axis=-1,keepdims=True),1e-20)
    reference=vector[...,0:1]
    valid=np.abs(reference[...,0])>1e-8
    vector/=np.where(np.abs(reference)>1e-8,reference,1)
    inverse=(np.linalg.solve(noise,vector[...,None]))[...,0]
    denominator=(vector.conj()*inverse).sum(axis=-1).real.clip(1e-8)
    rank1=inverse/denominator[...,None]
    rank1=np.where(valid[...,None],rank1,eye[0])
    outputs={name:(weight.conj()*mixture).sum(axis=-1)
             for name,weight in (('souden',souden),('rank1',rank1))}
    outputs.update({name+'_wiener':value*np.sqrt(mask) for name,value in tuple(outputs.items())})
    assert all(np.isfinite(value).all() for value in outputs.values())
    return outputs,updated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    model,_=load_model(checkpoint,torch.device('cpu'),'causal_unet_spatial_v1')
    objective=QualityObjective();rows=[];start=time.perf_counter()
    report=dict(status='running',scope='Eight fixed training samples; causal CPU screening only, not generalization or real-time proof',
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mask_source='Current trained model output and mixture only; no clean reference',
        alpha=0.98,diagonal_loading=0.001,mask_clip=[0.001,0.999],power_iterations=3,
        reference_mic=0,num_mics=8,frontend=dict(sample_rate=16000,n_fft=512,hop_length=160,win_length=320,power=0.5),
        target_gain_matching=False,external_normalization=False,rows=rows,
        formula_source='https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.mvdr_weights_souden.html')
    with torch.inference_mode():
        for index in range(1,9):
            root=Path(args.data_dir)
            mix,sr=sf.read(root/f'mixture_{index}.wav',dtype='float32',always_2d=True)
            clean,tsr=sf.read(root/f'target_{index}.wav',dtype='float32',always_2d=True)
            assert sr==tsr==16000 and len(mix)==len(clean)==96000 and mix.shape[1]>=8
            mixture=torch.from_numpy(mix[:,:8].copy())[None]
            features,_=build_stft_batch(mixture,mixture[:,:,0],512,160,320,0.5,torch.device('cpu'))
            estimate=model(features)
            raw_x=CausalSpatialEnhancer.decompress(features[0]).contiguous()
            raw_s=CausalSpatialEnhancer.decompress(estimate[0].permute(1,2,0).contiguous())
            x=torch.view_as_complex(raw_x).numpy();s=torch.view_as_complex(raw_s).numpy()
            mask=np.abs(s)**2/(np.abs(s)**2+np.abs(x[:,:,0]-s)**2+1e-10)
            spectra,_=past_mvdr(x,mask)
            waves={'baseline':objective.reconstruct(estimate,96000)[0].numpy()}
            for name,value in spectra.items():
                raw=torch.from_numpy(np.stack((value.real,value.imag),axis=-1).astype(np.float32))
                compressed=CausalSpatialEnhancer.compress(raw).permute(2,0,1)[None]
                waves[name]=objective.reconstruct(compressed,96000)[0].numpy()
            row=dict(sample_id=index,num_samples=96000,metrics={name:dict(
                pesq=float(pesq(16000,clean[:,0],wave,'wb')),
                si_snr_db=compute_si_snr_db(clean[:,0],wave)) for name,wave in waves.items()})
            rows.append(row);print(json.dumps(row),flush=True)
            (output/'probe.json').write_text(json.dumps(report,indent=2))
    report.update(status='complete',completed_samples=8,elapsed_seconds=time.perf_counter()-start,
        mean={name:{metric:float(np.mean([row['metrics'][name][metric] for row in rows]))
                    for metric in ('pesq','si_snr_db')} for name in rows[0]['metrics']})
    (output/'probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean']),flush=True)


if __name__=='__main__':main()
