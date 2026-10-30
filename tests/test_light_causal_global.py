"""Global-frequency causality and real warm-start/train/resume plumbing."""

import os
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import evaluate_light
from light_causal_global_model import CausalGlobalSpatial,GLOBAL_CANDIDATE
from light_causal_fast_model import CausalUNetSpatial
from light_candidate_runtime import read_candidate_checkpoint
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli,ROOT
from test_light_quality_ddp import run_two_windows_gloo_ranks
from train_light_causal_global import aligned_sampler_type


@pytest.fixture
def model():
    torch.set_num_threads(2);torch.manual_seed(91)
    model=CausalGlobalSpatial(hidden=32,ffn=64,layers=2,context=5,global_hidden=16)
    model.beamformer.weight.data.normal_(std=.01)
    model.layers[-1].output.weight.data.normal_(std=.01)
    return model


def test_global_constructor_and_sampler_preserve_source_data_randomness():
    torch.manual_seed(1234);CausalUNetSpatial();expected=torch.get_rng_state()
    torch.manual_seed(1234);CausalGlobalSpatial();torch.testing.assert_close(torch.get_rng_state(),expected,rtol=0,atol=0)
    sampler=torch.utils.data.distributed.DistributedSampler
    baseline=sampler(list(range(23)),num_replicas=2,rank=0,shuffle=True)
    changed=aligned_sampler_type(sampler,9)(list(range(23)),num_replicas=2,rank=0,shuffle=True)
    baseline.set_epoch(10);changed.set_epoch(1)
    assert list(baseline)==list(changed)


@pytest.mark.parametrize('training',[False,True])
def test_global_recurrence_never_uses_future_frames(model,training):
    model.train(training)
    x=(torch.randn(1,19,257,8,2)*.01).requires_grad_()
    changed=x.detach().clone();changed[:,11:]*=17
    output=model(x)
    torch.testing.assert_close(output[:,:,:11],model(changed)[:,:,:11],rtol=0,atol=0)
    output[:,:,:11].square().mean().backward()
    assert torch.count_nonzero(x.grad[:,11:])==0
    assert torch.count_nonzero(x.grad[:,:11])>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_global_hidden_state_is_bounded_and_chunk_outputs_match(model):
    model.eval();x=torch.randn(1,31,257,8,2)*.01
    with torch.inference_mode():
        expected=model(x);state=None;parts=[]
        for start,end in ((0,1),(1,8),(8,19),(19,31)):
            part,state=model.forward_stream(x[:,start:end],state);parts.append(part)
            assert state['middle'][-1]['hidden'].shape==(1,1,16)
        torch.testing.assert_close(torch.cat(parts,2),expected,rtol=3e-5,atol=2e-6)


def test_global_waveform_stream_preserves_frontend_and_warmup(model):
    model.eval();mixture=torch.randn(1,1327,8)*.01
    with torch.inference_mode():
        features,_=build_stft_batch(mixture,mixture[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),1327)
        stream=CausalWaveformStream(model);stream.warmup()
        assert stream.state is None and stream.total_samples==0
        actual=torch.cat((stream.push(mixture[:,:311]),stream.push(mixture[:,311:]),stream.flush()),1)
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-6)


def global_case(argv,tmp_path):
    run_cli('train_light_causal_fast.py',[*argv,'--num-epochs','1'])
    source=Path(argv[argv.index('--checkpoint-dir')+1])/'model_epoch_1.pt'
    arguments=list(argv)
    for flag,part in (('--checkpoint-dir','checkpoints'),('--best-dir','best'),('--log-dir','logs')):
        arguments[arguments.index(flag)+1]=str(tmp_path/'global'/part)
    run_cli('train_light_causal_global.py',[*arguments,'--warmstart',str(source),'--num-epochs','0'])
    return arguments,source


def test_global_zero_residual_warmstart_and_actual_resume(argv,tmp_path):
    arguments,source=global_case(argv,tmp_path)
    directory=Path(arguments[arguments.index('--checkpoint-dir')+1])
    state,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',GLOBAL_CANDIDATE)
    assert state['epoch']==0 and not state['optimizer_state_dict']['state']
    original,_=read_candidate_checkpoint(source,'causal_unet_spatial_v1')
    for key,value in original['model_state_dict'].items():torch.testing.assert_close(state['model_state_dict'][key],value,rtol=0,atol=0)
    a,_=evaluate_light.load_model(source,torch.device('cpu'),'causal_unet_spatial_v1')
    b,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),GLOBAL_CANDIDATE)
    with torch.inference_mode():
        x=torch.randn(1,7,257,8,2)*.01
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    for epoch in (1,2):
        run_cli('train_light_causal_global.py',[*arguments,'--num-epochs',str(epoch)])
        trained,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',GLOBAL_CANDIDATE)
        assert {int(value['step']) for value in trained['optimizer_state_dict']['state'].values()}=={2*epoch}
        assert all(torch.isfinite(value).all() for value in trained['model_state_dict'].values())
        assert torch.count_nonzero(trained['model_state_dict']['layers.8.output.weight'])>0


def test_actual_two_rank_global_update(argv,tmp_path):
    arguments,_=global_case(argv,tmp_path)
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    for name in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):environment.pop(name,None)
    arguments=[*arguments,'--num-epochs','1','--parallel-mode','ddp']
    if os.name=='nt':
        result=run_two_windows_gloo_ranks(tmp_path,arguments,environment,entry_module='train_light_causal_global')
    else:
        result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            'train_light_causal_global.py',*arguments],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    state,_=read_candidate_checkpoint(Path(arguments[arguments.index('--checkpoint-dir')+1])/'checkpoint_latest.pt',GLOBAL_CANDIDATE)
    assert {int(value['step']) for value in state['optimizer_state_dict']['state'].values()}=={1}


@pytest.mark.parametrize('fault',[None,'partial','gain','baseline_changed'])
def test_global_launcher_requires_full600_and_exact_warmstart_protocol(tmp_path,monkeypatch,fault):
    source=tmp_path/'source.pt';source_logs=tmp_path/'source_logs'
    torch.save(dict(epoch=9,args={'log_dir':str(source_logs)}),source)
    report=dict(status='complete',completed_samples=600,expected_samples=600,
        frontend={'n_fft':512},frontend_protocol={'normalization':'none'},num_mics=8,
        selected_pairs_sha256='same',metadata_sha256='same',metric_protocol={'target_gain_matching':False},
        structure_candidate='causal_unet_spatial_v1',candidate_evidence={'model_causal':True,'future_frames':0},
        candidate_implementation_sha256={'model':'source'},checkpoint=str(source),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mean={'enhanced':{'pesq':2.24},'noisy':{'pesq':1.08}})
    path=source_logs/'pesq_selection/model_epoch_9/summary.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(report))
    for epoch in (0,1):
        value=copy.deepcopy(report);value['structure_candidate']=GLOBAL_CANDIDATE
        value['candidate_implementation_sha256']={'model':'global'}
        if epoch==0 and fault=='baseline_changed':value['mean']['enhanced']['pesq']+=.01
        if epoch==1:
            value['mean']['enhanced']['pesq']=3.45
            if fault=='partial':value['completed_samples']=599
            if fault=='gain':value['metric_protocol']['target_gain_matching']=True
        path=tmp_path/f'run/pesq_selection/model_epoch_{epoch}/summary.json'
        path.parent.mkdir(parents=True);path.write_text(json.dumps(value))
    shell=(ROOT/'run_light_causal_global.sh').read_text()
    code=shell.split("<<'PY_RESULTS'\n",1)[1].split('\nPY_RESULTS',1)[0]
    monkeypatch.setattr(sys,'argv',['-',str(tmp_path/'run'),str(source)])
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'global full600 gate','exec'),{})
    else:
        exec(compile(code,'global full600 gate','exec'),{})
        ranked=json.loads((tmp_path/'run/pesq_selection/checkpoint_ranking.json').read_text())
        assert ranked['target_reached'] and ranked['model_is_causal']
