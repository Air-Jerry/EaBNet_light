"""Compare training-only PMSQE with a verified, reusable fixed-budget TCN control."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
from pesq import pesq

from experiments.causal_pmsqe_objective import PMSQEObjective
from experiments.causal_tcn_mapping import CausalTCNMapping
from light_candidate_runtime import read_candidate_checkpoint
from light_quality_loss import QualityObjective


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(model, objective, mixture, target):
    model.eval(); objective.reset_stats()
    with torch.no_grad():
        scores, losses = [], []
        for index in range(8):
            features, ref = objective.build_features(mixture[index:index + 1], target[index:index + 1],
                                                     512, 160, 320, .5, 'cpu', [96000])
            estimate = model(features)
            losses.append(float(objective(estimate, ref, [601])))
            waveform = objective.reconstruct(estimate, 96000)[0].numpy()
            scores.append(float(pesq(16000, target[index].numpy(), waveform, 'wb')))
    return dict(mean_pesq=sum(scores) / 8, per_sample_pesq=scores,
                mean_loss=sum(losses) / 8, loss_components=objective.stats())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--control-run', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    root, control_dir, output = map(Path, (args.data_dir, args.control_run, args.output_dir))
    previous = json.loads((control_dir / 'probe.json').read_text())
    assert previous['status'] == 'complete' and previous['audio_file_occurrences_unchanged'] == 16
    assert previous['private_audio_hashes_exported'] is False
    assert (previous['seed'], previous['updates'], previous['effective_batch'],
            previous['learning_rate'], previous['segment_samples']) == (1234, 200, 2, .001, 96000)
    assert previous['source_sha256'] == '0af72eedca34f09daf4f4ab057eab72d04ca66c741fe0a0dd2214a4f77210636'
    assert previous['tcn_implementation_sha256'] == sha(Path(__file__).with_name('causal_tcn_mapping.py'))
    control = next(row for row in previous['results'] if row['candidate'] == 'tcn_control')
    assert control['parameters'] == control['trainable_parameters'] == 2557650
    measurements = {row['updates']: row for row in control['measurements']}
    assert set(measurements) == {0, 100, 200}
    checkpoint = Path(previous['warmstart']['checkpoint'])
    assert sha(checkpoint) == previous['warmstart']['sha256']
    state, _ = read_candidate_checkpoint(checkpoint, 'causal_recurrent_tcn_v1')
    assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
    mixture, target, private_hashes = [], [], {}
    for index in range(1, 9):
        mix, sr = sf.read(root / f'mixture_{index}.wav', dtype='float32', always_2d=True)
        ref, rsr = sf.read(root / f'target_{index}.wav', dtype='float32', always_2d=True)
        assert sr == rsr == 16000 and len(mix) == len(ref) == 96000 and mix.shape[1] >= 8
        mixture.append(torch.from_numpy(mix[:, :8].copy())); target.append(torch.from_numpy(ref[:, 0].copy()))
        for name in ('mixture', 'target'):
            path = root / f'{name}_{index}.wav'
            private_hashes[path] = sha(path)
    mixture, target = torch.stack(mixture), torch.stack(target)
    torch.manual_seed(1234)
    model = CausalTCNMapping()
    model.load_state_dict(torch.load(control_dir / 'tcn_control.pt', map_location='cpu', weights_only=True), strict=True)
    observed = measure(model, QualityObjective(), mixture, target)
    assert all(abs(a - b) < 1e-6 for a, b in zip(observed['per_sample_pesq'], measurements[200]['per_sample_pesq']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    objective = PMSQEObjective(weight=.1)
    initial = measure(model, objective, mixture, target)
    assert all(abs(a - b) < 1e-6 for a, b in zip(initial['per_sample_pesq'], measurements[0]['per_sample_pesq']))
    assert all(parameter.requires_grad for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    shuffle = torch.Generator().manual_seed(1234)
    indices = torch.cat([torch.randperm(8, generator=shuffle) for _ in range(50)])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', scope='Fixed eight-crop training fit only; not generalization or full600 quality',
        seed=1234, updates=200, effective_batch=2, learning_rate=.001, segment_samples=96000,
        precision='CPU FP32', objective='QualityObjective plus0.1 training-only upstream PMSQE',
        source_checkpoint_sha256=sha(checkpoint), source_epoch=state['epoch'],
        control_run=str(control_dir), control_probe_sha256=sha(control_dir / 'probe.json'),
        control_weights_sha256=sha(control_dir / 'tcn_control.pt'),
        control_endpoint_recomputed_on_all8=True, initial_per_sample_scores_equal_control=True,
        control_measurements=control['measurements'], private_audio_hashes_exported=False,
        implementation_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('probe_causal_pmsqe.py', 'causal_pmsqe_objective.py', 'causal_tcn_mapping.py')},
        vendor_provenance_sha256=sha(Path(__file__).resolve().parents[1] / 'third_party/asteroid_pmsqe/PROVENANCE.json'),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        measurements=[dict(updates=0, **initial)])

    def save():
        (output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')

    start = time.perf_counter(); norms = []
    print(json.dumps(report['measurements'][0]), flush=True); save()
    try:
        for update in range(1, 201):
            model.train(); ids = indices[(update - 1) * 2:update * 2]
            features, ref = objective.build_features(mixture[ids], target[ids], 512, 160, 320, .5, 'cpu', [96000] * 2)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(features), ref, [601] * 2)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            norms.append(float(norm)); optimizer.step()
            if update <= 3 or update % 25 == 0:
                print('PMSQE update', update, 'elapsed', round(time.perf_counter() - start, 2), flush=True)
            if update in (100, 200):
                row = dict(updates=update, **measure(model, objective, mixture, target))
                row.update(elapsed_seconds=time.perf_counter() - start,
                    gradient_norm_mean=sum(norms) / len(norms), gradient_norm_max=max(norms),
                    clipped_update_fraction=sum(value > 5 for value in norms) / len(norms))
                report['measurements'].append(row); print(json.dumps(row), flush=True); save()
        assert {int(item['step']) for item in optimizer.state.values()} == {200}
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        assert all(torch.isfinite(value).all() for item in optimizer.state.values()
                   for value in item.values() if torch.is_tensor(value))
        assert all(sha(path) == digest for path, digest in private_hashes.items())
        torch.save(model.state_dict(), output / 'pmsqe_training_only_fit.pt')
        report.update(status='complete', adam_updates=200, all_weights_and_adam_finite=True,
            audio_file_occurrences_unchanged=16, inference_unchanged=True,
            endpoint_pesq_delta_vs_control=report['measurements'][-1]['mean_pesq'] - measurements[200]['mean_pesq'])
        save()
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__':
    main()
