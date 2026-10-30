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
    args = parser.parse_args()
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
        seed=1234, updates=args.updates, effective_batch=2, learning_rate=0.001, segment_samples=96000,
        sample_audio_sha256=hashes, objective='Unchanged QualityObjective', precision='CPU FP32',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=[])
    shuffle = torch.Generator().manual_seed(1234)
    indices = torch.cat([torch.randperm(8,generator=shuffle) for _ in range((args.updates+3)//4)])
    assert len(indices)>=args.updates*2
    for name, constructor in [('frequency17',CausalUNetSpatial),('frequency65',ResolutionSpatial)]:
        torch.manual_seed(1234)
        model=constructor()
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        start=time.perf_counter()
        rows=[]
        for update in range(args.updates+1):
            if update in (0,args.updates//2,args.updates):
                model.eval()
                with torch.no_grad():
                    scores=[]; losses=[]
                    for i in range(8):
                        features,ref=objective.build_features(mixture[i:i+1],target[i:i+1],512,160,320,0.5,'cpu',[96000])
                        estimate=model(features)
                        losses.append(float(objective(estimate,ref,[601])))
                        wave=objective.reconstruct(estimate,96000)[0].numpy()
                        scores.append(float(pesq(16000,target[i].numpy(),wave,'wb')))
                row=dict(updates=update,mean_pesq=sum(scores)/8,per_sample_pesq=scores,
                         mean_loss=sum(losses)/8,elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
                print(name,json.dumps(row),flush=True)
                report['active']=dict(candidate=name,completed_updates=update,results=rows)
                (output/'probe.json').write_text(json.dumps(report,indent=2))
            if update==args.updates:break
            model.train()
            ids=indices[update*2:update*2+2]
            features,ref=objective.build_features(mixture[ids],target[ids],512,160,320,0.5,'cpu',[96000]*2)
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(features),ref,[601]*2)
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(norm)
            optimizer.step()
            if update<3 or (update+1)%25==0:
                print(name,'update',update+1,'elapsed',round(time.perf_counter()-start,2),flush=True)
        torch.save(model.state_dict(),output/f'{name}.pt')
        report['results'].append(dict(candidate=name,parameters=sum(p.numel() for p in model.parameters()),
                                      elapsed_seconds=time.perf_counter()-start,measurements=rows))
    report.pop('active',None)
    report['status']='complete'
    (output/'probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
