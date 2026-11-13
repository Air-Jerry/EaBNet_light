"""Run complete phase-filter epochs and audit a checkpoint that reaches WB-PESQ 3.4."""
import argparse
import csv
import json
import os
import re
import subprocess
import statistics
import sys
import time
from pathlib import Path

import torch

from audit_light_final import require_protocol, sha
from experiments.scaled_recurrent_entry import same_tensors
from light_candidate_runtime import PARALLEL_PHASE_FILTER_CANDIDATE, DOUBLE_PHASE_FILTER_CANDIDATE, PHASE_FILTER_CANDIDATE, read_candidate_checkpoint
from train_light_causal_phase_filter import initialize


EVALUATOR_SHA256 = '0c3ada1425b33e16688e8424daba23681ac9ded4d8bee931ed24c69e59e5730b'


def wait_for_idle_cards():
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                  text=True).strip():
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-path', 'output', 'regressor', 'scaling-helper'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--double-filter', action='store_true')
    mode.add_argument('--parallel-filter', action='store_true')
    parser.add_argument('--refit-score', action='store_true')
    args = parser.parse_args()
    if args.refit_score and not args.double_filter:
        parser.error('--refit-score requires --double-filter')
    torch.set_num_threads(2)
    if args.epochs <= 0 or sha('evaluate_light.py') != EVALUATOR_SHA256:
        raise ValueError('Use the original evaluator and request at least one epoch')
    root = args.output.resolve()
    transition = args.double_filter or args.parallel_filter
    candidate = PHASE_FILTER_CANDIDATE
    if transition:
        candidate = PARALLEL_PHASE_FILTER_CANDIDATE if args.parallel_filter else DOUBLE_PHASE_FILTER_CANDIDATE
    if not args.resume:
        root.mkdir()
    status = root/'run_status.json'
    source_digest = sha(args.source_path)
    previous = json.loads(status.read_text()) if args.resume else None
    if previous is not None:
        if previous['status'] != 'complete' or previous['source_checkpoint_sha256'] != source_digest:
            raise ValueError('Resume requires a completed run and its original source checkpoint')
    settings = dict(status='initializing', pid=os.getpid(), source_checkpoint_sha256=source_digest)

    def record():
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(settings, indent=2, allow_nan=False)+'\n')
        temporary.replace(status)

    checkpoints = root/'checkpoints'
    selection = root/'pesq_selection'
    timing_path = root/'epoch_timing.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUBLAS_WORKSPACE_CONFIG'):
        env.pop(key,None)
    if previous is None:
        results = []
        record()
        initialize(args.source_path, checkpoints, root/'best', root/'logs', double_filter=args.double_filter,
                   refit_regressor=args.regressor if args.refit_score else None, parallel_filter=args.parallel_filter)
        selection.mkdir()
    else:
        results = previous['results']
        last = results[-1]
        epoch = last['epoch']
        timing = json.loads(timing_path.read_text().splitlines()[-1])
        scored = json.loads((selection/f'model_epoch_{epoch}/summary.json').read_text())
        require_protocol(scored)
        if (timing['status'] != 'complete' or timing['epoch'] != epoch
                or scored['checkpoint_sha256'] != last['checkpoint_sha256']
                or scored['mean']['enhanced']['pesq'] != last['wb_pesq']
                or sha(checkpoints/f'model_epoch_{epoch}.pt') != last['checkpoint_sha256']):
            raise ValueError('The latest checkpoint must match the last completed and scored epoch')
    latest, _ = read_candidate_checkpoint(checkpoints/'checkpoint_latest.pt', candidate)
    if previous is not None:
        scored_state, _ = read_candidate_checkpoint(checkpoints/f'model_epoch_{epoch}.pt', candidate)
        if not same_tensors(latest, scored_state):
            raise ValueError('Saved model, Adam, or training state differs from the scored checkpoint')
    origin = latest['args']['phase_filter_source_epoch']
    start = latest['epoch'] + 1
    source_key = 'double_filter_source_sha256' if transition else 'phase_filter_source_sha256'
    if latest['args'][source_key] != source_digest or (previous and start != previous['epoch']+1):
        raise ValueError('Phase-filter checkpoint provenance differs from the requested run')
    settings.update(results=results, resumed_from_epoch=start-1)
    record()
    try:
        train_dir, val_dir = (latest['args'][k] for k in ('train_dir','val_dir'))
        source, _ = read_candidate_checkpoint(args.source_path)
        comparison = Path(source['args']['log_dir'])
        if not comparison.is_absolute():
            comparison = args.source_path.resolve().parent.parent/comparison
        if transition:
            comparison = args.source_path.resolve().parent.parent
        baseline = json.loads((comparison/f"pesq_selection/model_epoch_{source['epoch']}/summary.json").read_text())
        for epoch in range(start,start+args.epochs):
            settings.update(status='waiting_for_idle_gpus',epoch=epoch)
            record()
            wait_for_idle_cards()
            started = time.time()
            settings.update(status='training',epoch=epoch,started_unix=started)
            record()
            gpu_path = root/f'gpu_epoch_{epoch}.csv'
            with (root/f'train_epoch_{epoch}.log').open('x') as log, gpu_path.open('x') as gpu_log:
                sampler = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits','-l','10'],stdout=gpu_log,stderr=subprocess.STDOUT)
                try:
                    command = ['timeout','--kill-after=15s','3330s',sys.executable,'-u','-m','torch.distributed.run',
                        '--standalone','--nproc_per_node=2','train_light_causal_phase_filter.py','train',
                        '--checkpoint-dir',str(checkpoints),'--num-epochs',str(epoch),'--regressor',str(args.regressor.resolve()),
                        '--scaling-helper',str(args.scaling_helper.resolve())]
                    with subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL) as job:
                        pid = job.pid
                        cookie = (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
                        settings.update(launcher_pid=pid,launcher_cookie=cookie)
                        record()
                        if job.wait():
                            raise RuntimeError(f'Dual-GPU epoch {epoch} failed, see {log.name}')
                finally:
                    sampler.terminate()
                    sampler.wait(timeout=10)
            training_end = time.time()
            with gpu_path.open() as handle:
                samples = list(csv.reader(handle))
            utilization = {i:[int(row[2]) for row in samples if len(row)==4 and int(row[1])==i]
                           for i in (0,1)}
            text = (root/f'train_epoch_{epoch}.log').read_text()
            for phase,count in [('train',53600),('val',600)]:
                found = re.findall(r'\[PERF\]\['+phase+r'\] samples\(local/global\)=\d+/(\d+).*?non_finite_skips\(local\)=(\d+)',text)
                assert found and found[-1] == (str(count),'0'), (phase, found)
            checkpoint = checkpoints/f'model_epoch_{epoch}.pt'
            state, _ = read_candidate_checkpoint(checkpoint, candidate)
            adam = state['optimizer_state_dict']
            groups = [set(group['params']) for group in adam['param_groups']]
            assert len(adam['state'])==241 and len(groups)==2 and not (groups[0] & groups[1])
            assert groups[1] <= adam['state'].keys() and len(groups[1])==6
            assert {int(v['step']) for k,v in adam['state'].items() if k in groups[0]}=={epoch*6700}
            assert {int(v['step']) for k,v in adam['state'].items() if k in groups[1]}=={(epoch-origin)*6700}
            assert state['epoch']==epoch
            settings['status']='evaluating'
            record()
            folder=selection/f'model_epoch_{epoch}'
            folder.mkdir()
            report_path=folder/'summary.json'
            with (folder/'evaluation.log').open('x') as log:
                subprocess.run([sys.executable,'-u','evaluate_light.py','--checkpoint',str(checkpoint),
                    '--candidate',candidate,'--val-dir',val_dir,'--device','cuda',
                    '--max-samples','0','--save-samples','no','--match-estimate-level','no',
                    '--estimate-dir',str(folder),'--save-json',str(report_path)],
                    env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=max(1,3600-(time.time()-started)))
            report=json.loads(report_path.read_text())
            require_protocol(report)
            for key in ('frontend','frontend_protocol','metric_protocol','selected_pairs_sha256','metadata_sha256'):
                assert report[key]==baseline[key],key
            assert all(abs(v-baseline['mean']['noisy'][k])<1e-9 for k,v in report['mean']['noisy'].items())
            assert report['checkpoint_sha256']==sha(checkpoint)
            finished=time.time()
            elapsed=finished-started
            timing=dict(status='complete' if elapsed<=3600 else 'exceeded_one_hour',epoch=epoch,
                started_unix=started,finished_unix=finished,total_seconds=elapsed,
                training_validation_checkpoint_seconds=training_end-started,evaluation_audit_seconds=finished-training_end,
                training_samples=53600,validation_samples=600,full_metric_samples=600,
                optimizer_updates_this_epoch=6700,global_batch_size=8,
                gpu_utilization={i:dict(samples=len(values),average_pct=statistics.mean(values) if values else None)
                                 for i,values in utilization.items()})
            with timing_path.open('a') as handle:
                handle.write(json.dumps(timing,allow_nan=False)+'\n')
            assert elapsed<=3600
            score=report['mean']['enhanced']['pesq']
            results.append(dict(epoch=epoch,wb_pesq=score,seconds=elapsed,checkpoint_sha256=sha(checkpoint)))
            settings['results']=results
            record()
            if score>=3.4:
                settings['status']='final_audit_running'
                record()
                with (root/'final_audit.log').open('x') as log:
                    subprocess.run([sys.executable,'-u','audit_light_final.py','--checkpoint',str(checkpoint),
                        '--training-summary',str(report_path),'--training-timings',str(timing_path),
                        '--output-dir',str(root/'final_audit'),'--evaluator-sha256',EVALUATOR_SHA256,
                        '--launcher-pid',str(pid),'--launcher-cookie',cookie,'--train-dir',train_dir,'--val-dir',val_dir],
                        env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
                assert json.loads((root/'final_audit/final_audit.json').read_text())['status']=='passed'
                settings['status']='final_audit_passed'
                return
        settings['status']='complete'
    except BaseException as error:
        settings.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record()


if __name__=='__main__':
    main()
