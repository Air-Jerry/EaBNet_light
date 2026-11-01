"""Train the dense full-band causal TCN from scratch through the existing data/loop."""

from pathlib import Path
import sys

import train_light_causal as causal
from light_candidate_runtime import FULLBAND_TCN_CANDIDATE, candidate_metadata, read_candidate_checkpoint
from train_light_causal_pmsqe import FrontendPMSQE


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None, *unused, **kwargs):
    args = causal.trainer.parse_args(sys.argv[1:] if argv is None else argv, defaults=dict(
        checkpoint_dir='./checkpoints_causal_fullband_tcn_v1', best_dir='./bestmodels_causal_fullband_tcn_v1',
        log_dir='./logs_causal_fullband_tcn_v1', num_epochs=30, learning_rate=.001,
        batch_size=4, grad_accum_steps=1, segment_seconds=6, save_every=1,
        train_lr_patience=3, grad_clip=5., norm_type='LN', use_amp='no',
        model_amp='no', allow_tf32='no', stop_on_non_finite='yes', skip_non_finite_batches='no'))
    if (args.sample_rate,args.n_fft,args.hop_length,args.win_length,args.power,
            args.num_mics,args.target_ref_mic) != (16000,512,160,320,.5,8,0):
        raise ValueError('Keep the original16kHz eight-microphone reference0 frontend')
    if args.is_causal != 'yes' or args.norm_type != 'LN' or args.parallel_mode == 'dp':
        raise ValueError('Use causal LayerNorm with none or ddp')
    if any(value != 'no' for value in (args.use_amp,args.model_amp,args.allow_tf32)):
        raise ValueError('Keep internal BF16 activations and FP32 parameters/frontend/loss/inference')
    if args.stop_on_non_finite != 'yes' or args.skip_non_finite_batches != 'no':
        raise ValueError('Nonfinite updates must fail instead of silently skipping records')
    args.causal_training_objective = 'spectral_mrstft_waveform_pmsqe_v1'
    args.causal_training_precision = 'CUDA bfloat16 activations; FP32 parameters, frontend, loss and inference; CPU FP32'
    args.pmsqe_weight = .1
    files = ('train_light_causal_fullband.py','train_light_causal.py','train_light.py',
        'train_light_causal_pmsqe.py','light_quality_loss.py','experiments/causal_pmsqe_objective.py',
        'third_party/asteroid_pmsqe/pmsqe.py','third_party/asteroid_pmsqe/bark_matrix_16k.mat',
        'third_party/asteroid_pmsqe/PROVENANCE.json')
    args.causal_training_implementation_sha256 = {name:causal.quality.sha256_file(ROOT/name) for name in files}
    for name,value in candidate_metadata(FULLBAND_TCN_CANDIDATE).items():
        setattr(args,name,value)
    latest = Path(args.checkpoint_dir).expanduser()/'checkpoint_latest.pt'
    if latest.exists():
        if args.resume != 'yes':
            raise FileExistsError('Use --resume yes or new output directories')
        state,_ = read_candidate_checkpoint(latest,FULLBAND_TCN_CANDIDATE,args)
        for name in ('causal_training_objective','causal_training_precision','pmsqe_weight',
                     'causal_training_implementation_sha256'):
            if state['args'].get(name) != getattr(args,name):
                raise ValueError(f'Full-band resume {name} differs from checkpoint')
    outputs = [Path(getattr(args,name)).expanduser().resolve() for name in ('checkpoint_dir','best_dir','log_dir')]
    datasets = [Path(value).expanduser().resolve() for value in (args.train_dir,args.val_dir)]
    for index,output in enumerate(outputs):
        for other in [*datasets,*outputs[index+1:]]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError('Output directories must be separate and cannot overlap datasets')
    return args


def main(argv=None):
    parser,objective = causal.parse_args,causal.QualityObjective
    causal.parse_args,causal.QualityObjective = parse_args,FrontendPMSQE
    try:
        return causal.main(argv,candidate_id=FULLBAND_TCN_CANDIDATE,entry_file=__file__)
    finally:
        causal.parse_args,causal.QualityObjective = parser,objective


if __name__ == '__main__':
    raise SystemExit(main())
