"""Run the full-data 1011 candidate and matched continuation control on two GPUs."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'runs' / 'full_1011_seed42'
RUNNER = ROOT / 'scripts' / 'trajectory_1011.py'
LOCK = threading.Lock()
STATE = {}


def update(name, **values):
    with LOCK:
        STATE.setdefault(name, {}).update(values)
        STATE[name]['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp = SUITE / 'status.tmp'
        temp.write_text(json.dumps(STATE, indent=2))
        temp.replace(SUITE / 'status.json')


def execute(name, stage, args, gpu):
    command = [sys.executable, '-u', str(RUNNER), *args]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    log = SUITE / f'{name}_{stage}.log'
    update(name, stage=stage, state='running', gpu=gpu, command=command, log=str(log))
    with log.open('x') as handle:
        result = subprocess.run(command, cwd=ROOT.parent, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        update(name, state='failed', returncode=result.returncode)
        raise RuntimeError(f'{name}/{stage} failed ({result.returncode}); see {log}')


def worker(name, gpu, improvements):
    directory = SUITE / name
    common = [
        '--output', str(directory), '--init', str(ROOT / 'runs/gaze_focus_seed42/convnext_base_motion'),
        '--backbone', 'convnext_base', '--dim', '128', '--fpn', '--foveal', '--motion-bias',
        '--window', '16', '--burn-in', '8', '--image-width', '256', '--sigma', '64',
        '--prior-weight', '.25', '--nll-weight', '.5', '--batch-size', '2',
        '--epochs', '15', '--warmup-epochs', '2', '--backbone-mode', 'last',
        '--backbone-lr', '1e-5', '--lr', '1e-4', '--weight-decay', '1e-4',
        '--seed', '42', '--threads', '4', '--device', 'cuda:0', '--selection-mode', 'legacy16',
    ]
    if improvements:
        common += ['--mode', 'mixed', '--mixed-ratio', '.5', '--spatial-history',
                   '--history-length', '4', '--behavior-mixture']
    else:
        common += ['--mode', 'legacy16']
    execute(name, 'train', ['train', *common], gpu)
    execute(name, 'evaluate', ['evaluate', '--checkpoint', str(directory), '--output', str(SUITE / f'{name}_eval'),
                              '--device', 'cuda:0'], gpu)
    for seed, strategy in [(42, 'sample'), (43, 'sample'), (44, 'sample'), (42, 'argmax')]:
        stage = f'{strategy}_{seed}'
        execute(name, stage, ['generate', '--checkpoint', str(directory),
                             '--output', str(SUITE / f'{name}_{stage}'), '--device', 'cuda:0',
                             '--steps', '16', '--strategy', strategy, '--seed', str(seed),
                             '--policy', 'none', '--gaze-diagnostics'], gpu)
    update(name, state='complete', stage='complete', returncode=0)


def main():
    SUITE.mkdir(parents=True, exist_ok=False)
    (SUITE / 'experiment.json').write_text(json.dumps({
        'training_images': 412, 'validation_images': 93, 'epochs': 15, 'training_seed': 42,
        'initializer': 'gaze_focus_seed42/convnext_base_motion/best.pt',
        'candidate': 'mixed .5 + recent4 spatial history + global/local/return mixture',
        'control': 'legacy16 continuation, new history and behavior modules disabled',
        'matched': 'initializer, loss, optimizer, effective batch, input resolution, epochs, checkpoint selection',
        'selection': 'minimum legacy16 validation NLL among trained epochs',
        'evaluation_status': 'exploratory validation reused for initializer and checkpoint selection',
        'generation_seeds': [42, 43, 44], 'test_used': False,
        'note': 'Generation seeds are not independent training seeds. No claim of full ablation or 3-seed training.',
    }, indent=2))
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, 'improved', 0, True), pool.submit(worker, 'control', 1, False)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
    (SUITE / 'completion.json').write_text(json.dumps({'complete': not errors, 'errors': errors}, indent=2))
    if errors:
        raise SystemExit('\n'.join(errors))


if __name__ == '__main__':
    main()
