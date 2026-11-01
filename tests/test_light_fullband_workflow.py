"""Default45M model training, saved inference and actual resumed Adam updates."""

from pathlib import Path

import torch

import evaluate_light
import train_light_causal as causal
from light_candidate_runtime import FULLBAND_TCN_CANDIDATE, read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def test_fullband_default_model_train_resume_and_saved_inference(argv):
    torch.set_num_threads(2)
    parser, objective = causal.parse_args, causal.QualityObjective
    entry = 'train_light_causal_fullband.py'
    directory = Path(argv[argv.index('--checkpoint-dir')+1])
    run_cli(entry,[*argv,'--num-epochs','1'])
    assert causal.parse_args is parser and causal.QualityObjective is objective
    first,_ = read_candidate_checkpoint(directory/'checkpoint_latest.pt',FULLBAND_TCN_CANDIDATE)
    assert first['epoch'] == 1
    assert first['args']['candidate_schema'] == 14 and first['args']['pmsqe_weight'] == .1
    assert first['args']['candidate_evidence']['author_confirmed'] is False
    assert {int(v['step']) for v in first['optimizer_state_dict']['state'].values()} == {2}
    assert torch.count_nonzero(first['model_state_dict']['encoder.weight']) > 0
    assert torch.count_nonzero(first['model_state_dict']['mapping.weight']) > 0
    del first
    model,_ = evaluate_light.load_model(directory/'model_epoch_1.pt',torch.device('cpu'),FULLBAND_TCN_CANDIDATE)
    assert model.is_causal and sum(p.numel() for p in model.parameters()) == 45352542
    assert all(p.requires_grad for p in model.parameters())
    value = torch.randn(1,7,257,8,2)*.01
    with torch.inference_mode():
        expected = model(value)
        changed = value.clone(); changed[:,4:] *= 17
        torch.testing.assert_close(model(changed)[:,:,:4],expected[:,:,:4],rtol=0,atol=0)
    del model
    run_cli(entry,[*argv,'--num-epochs','2'])
    second,_ = read_candidate_checkpoint(directory/'checkpoint_latest.pt',FULLBAND_TCN_CANDIDATE)
    assert second['epoch'] == 2
    assert {int(v['step']) for v in second['optimizer_state_dict']['state'].values()} == {4}
    assert all(torch.isfinite(v).all() for v in second['model_state_dict'].values())
    assert all(torch.isfinite(v).all() for item in second['optimizer_state_dict']['state'].values()
               for v in item.values() if torch.is_tensor(v))
