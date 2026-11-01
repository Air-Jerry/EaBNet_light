"""Actual source training, exact spatial warm-start, saved inference and resumed Adam."""

from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light_causal_fullband as fullband
import train_light_causal_fullband_spatial as spatial
from light_candidate_runtime import FULLBAND_TCN_CANDIDATE,FULLBAND_SPATIAL_CANDIDATE,read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def test_default_spatial_actual_source_warmstart_resume_and_saved_inference(argv,tmp_path):
    torch.set_num_threads(2)
    run_cli('train_light_causal_fullband.py',[*argv,'--num-epochs','1'])
    directory=Path(argv[argv.index('--checkpoint-dir')+1])
    source=directory/'model_epoch_1.pt'
    original,_=read_candidate_checkpoint(source,FULLBAND_TCN_CANDIDATE)
    for flag,name in (('--checkpoint-dir','checkpoints'),('--best-dir','best'),('--log-dir','logs')):
        argv[argv.index(flag)+1]=str(tmp_path/'spatial'/name)
    entry='train_light_causal_fullband_spatial.py'
    parser,initializer,sampler=fullband.parse_args,fullband.causal.initialize_checkpoint,fullband.causal.trainer.DistributedSampler
    run_cli(entry,[*argv,'--warmstart',str(source),'--num-epochs','0'])
    assert fullband.parse_args is parser and fullband.causal.initialize_checkpoint is initializer
    assert fullband.causal.trainer.DistributedSampler is sampler
    directory=Path(argv[argv.index('--checkpoint-dir')+1])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',FULLBAND_SPATIAL_CANDIDATE)
    assert initial['args']['candidate_schema']==15 and initial['args']['spatial_fullband_sample_epoch_offset']==1
    assert initial['args']['pmsqe_weight']==.1 and not initial['optimizer_state_dict']['state']
    assert entry in initial['args']['causal_training_implementation_sha256']
    assert torch.count_nonzero(initial['model_state_dict']['spatial_readout.weight'])==0
    for name,value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][name],value,rtol=0,atol=0)
    del original,initial
    before,_=evaluate_light.load_model(source,torch.device('cpu'),FULLBAND_TCN_CANDIDATE)
    after,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),FULLBAND_SPATIAL_CANDIDATE)
    assert sum(p.numel() for p in after.parameters())==45501488 and all(p.requires_grad for p in after.parameters())
    with torch.inference_mode():
        value=torch.randn(1,7,257,8,2)*.02
        torch.testing.assert_close(after(value),before(value),rtol=0,atol=0)
    del before,after
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',FULLBAND_SPATIAL_CANDIDATE)
        assert state['epoch']==epoch
        assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={2*epoch}
        assert torch.count_nonzero(state['model_state_dict']['spatial_readout.weight'])>0
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for item in state['optimizer_state_dict']['state'].values()
                   for v in item.values() if torch.is_tensor(v))
        del state
    with pytest.raises(ValueError,match='warm-start differs'):
        spatial.parse_args([*argv,'--warmstart',str(source.parent/'model_epoch_0.pt')])
