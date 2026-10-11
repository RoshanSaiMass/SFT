"""Plan and execute separate SFT research arrays without modifying any trainer."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

DATASETS = ['pets', 'cifar100', 'flowers102', 'dtd', 'svhn', 'fgvc_aircraft',
            'eurosat', 'sun397', 'pcam', 'clevr', 'dsprites-loc', 'dsprites-ori']
GROUPS = ['snip_only', 'all_block_snip', 'all_block_snip+peft', 'loo', 'looall',
          'sub_correlation', 'sub_removal']
FOLDERS = {g: 'sub_sft' if g.startswith('sub_') else g for g in GROUPS}
BACKENDS = {
    'snip_only': ('SNIP_DIR', 'train_snip_compact.py'),
    'all_block_snip': ('ORIGINAL_DIR', 'train_sfp_lora.py'),
    'all_block_snip+peft': ('ORIGINAL_DIR', 'train_sfp_lora.py'),
    'loo': ('LOO_DIR', 'train_loo_compact.py'),
    'looall': ('SYMBOLIC_DIR', 'train_sfp_lora.py'),
    'sub_correlation': ('SUBTRACTION_DIR', 'train_subtraction.py'),
    'sub_removal': ('SUBTRACTION_DIR', 'train_subtraction.py'),
}


def methods(include_uni=True):
    result = []
    for family in ('lora', 'dora', 'paca', 'rpaca'):
        tuners = ('direct', 'lora', 'dora') if 'paca' in family else (None,)
        for tuner in tuners:
            name = family if tuner in (None, 'direct') else family + '_' + tuner
            flags = ['--adapter-type', family]
            if tuner is None:
                flags += ['--lora-rank', '16']
            elif tuner == 'direct':
                flags += ['--lora-rank', '32', '--paca-rank', '32', '--paca-tuner', 'direct']
            else:
                flags += ['--paca-tuner', tuner, '--paca-rank', '32', '--paca-adapter-rank', '16']
            result.append((name, flags))
            if tuner != 'direct':
                for value in ('0.9', '1.5'):
                    result.append((name + '_ortho_' + value,
                                   flags + ['--lora-ortho-lambda1', value,
                                            '--lora-ortho-lambda2', value]))
    if include_uni:
        for family in ('unilora', 'unidora'):
            result.append((family, ['--adapter-type', family, '--lora-rank', '16',
                                    '--unilora-dim', '72000']))
    return result


def tasks(group):
    configs = methods(include_uni=group != 'loo') if group in (
        'all_block_snip+peft', 'loo', 'sub_correlation', 'sub_removal') else [('none', [])]
    blocks = range(12) if group in ('all_block_snip', 'all_block_snip+peft', 'looall', 'sub_removal') else [None]
    return [dict(task_id=i, group=group, dataset=dataset, method=method,
                 block=block, method_flags=flags)
            for i, (dataset, method, block, flags) in enumerate(
                (d, m, b, f) for d in DATASETS for m, f in configs for b in blocks)]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def options(args):
    base = Path(os.environ.get('CLUSTER_BASE', '/export/home/achyut/Sarvesh'))
    defaults = {
        'SNIP_DIR': base / 'SFT_BEST/SFT_SNIP_COMPACT',
        'ORIGINAL_DIR': base / 'SFT_FINAL',
        'SYMBOLIC_DIR': base / 'SFT_EXP/SFT_EXPERIEMTN',
        'LOO_DIR': base / 'SFT_LOO/SFT_LOO_COMPACT',
        'SUBTRACTION_DIR': base / 'SFT_COEFF_RED/SFT_BLOCK_SUBTRACTION',
    }
    return dict(backend_dirs={key: str(Path(os.environ.get(key, value)).expanduser().resolve())
                              for key, value in defaults.items()},
                results_root=str(Path(args.results_root or os.environ.get('RESULTS_ROOT', base / 'research_results')).resolve()),
                data_dir=str(Path(os.environ.get('DATA_DIR', base / 'SFT_BEST/data')).resolve()),
                cache_dir=str(Path(os.environ.get('CACHE_DIR', base / 'SFT_BEST/data/research_cache')).resolve()),
                campaign=args.campaign or os.environ.get('CAMPAIGN') or 'manual_' + os.environ.get('SLURM_ARRAY_JOB_ID', 'local'),
                seed=args.seed, epochs=args.epochs, max_epochs=args.max_epochs, patience=args.patience,
                num_samples=args.num_samples, batch_size=args.batch_size,
                head_warmup_epochs=args.head_warmup_epochs, distill_epochs=args.distill_epochs)


def command(task, settings, destination):
    key, script = BACKENDS[task['group']]
    code = Path(settings['backend_dirs'][key])
    flags = ['--dataset', task['dataset'], '--seed', str(settings['seed']), '--device', 'cuda',
             '--num-samples', str(settings['num_samples']), '--batch-size', str(settings['batch_size']),
             '--epochs', str(settings['epochs']), '--max-epochs', str(settings['max_epochs']),
             '--patience', str(settings['patience']), '--data-dir', settings['data_dir'],
             '--output-dir', str(destination / 'results')]
    group = task['group']
    if group == 'snip_only':
        flags += ['--filter-type', 'dense', '--adapter-type', 'none', '--snip-momentum', '0.3',
                  '--snip-samples', '64', '--snip-batch-size', '32',
                  '--snip-search-epochs', '4', '--snip-prune-fraction', '0.5']
    elif group in ('all_block_snip', 'all_block_snip+peft', 'looall'):
        flags += ['--pruned-block', str(task['block']), '--filter-block-layers', '1',
                  '--filter-residual-hidden-dim', '0', '--filter-dropout', '0']
        if group != 'all_block_snip+peft':
            flags += ['--mode', 'sft']
        if group == 'looall':
            flags += ['--filter-type', 'symbolic', '--filter-rank', '16',
                      '--symbolic-max-terms', '2', '--symbolic-penalty', '0.001',
                      '--symbolic-search-tokens', '2048']
    elif group == 'loo':
        flags += ['--filter-type', 'symbolic', '--filter-rank', '16',
                  '--symbolic-max-terms', '2', '--symbolic-penalty', '0.001',
                  '--symbolic-search-tokens', '2048', '--loo-samples', '512', '--loo-batch-size', '32',
                  '--head-warmup-epochs', str(settings['head_warmup_epochs'])]
    else:
        flags += ['--method', 'correlation' if group == 'sub_correlation' else 'drop-one',
                  '--train-layernorms', 'all']
        if group == 'sub_correlation':
            flags += ['--pair-index', '-1', '--min-correlation', '-1',
                      '--adapter-scope', 'retained', '--calibration-samples', '64',
                      '--distill-epochs', str(settings['distill_epochs'])]
        else:
            flags += ['--remove-block', str(task['block']), '--adapter-scope', 'all',
                      '--distill-epochs', '0', '--defer-test']
    if task['method_flags']:
        flags += ['--lora-alpha', '32'] + task['method_flags']
    return [sys.executable, '-u', str(code / script)] + flags, code


def destination(task, settings):
    block = 'selected' if task['block'] is None else f"block{task['block']:02d}"
    return Path(settings['results_root']) / FOLDERS[task['group']] / settings['campaign'] / task['group'] / (
        f"{task['task_id']:04d}_{task['dataset']}_{task['method']}_{block}_seed{settings['seed']}")


def prefetch(code, settings):
    # One subprocess per backend avoids collisions between their similarly named modules.
    # Existing download locks serialize first acquisition, not the actual training runs.
    helper = Path(__file__).resolve().parents[1] / 'slurm/sweep.py'
    payload = dict(code_dir=str(code), cache_dir=settings['cache_dir'], data_dir=settings['data_dir'])
    source = """import importlib.util,json,sys
helper,manifest,task=sys.argv[1:]
spec=importlib.util.spec_from_file_location('research_downloads',helper)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=json.loads(manifest);task=json.loads(task)
m.configure_environment(manifest);m.check_runtime(manifest);m.prefetch(manifest,task)
"""
    subprocess.run([sys.executable, '-c', source, str(helper), json.dumps(payload),
                    json.dumps(dict(dataset=settings['dataset'], seed=settings['seed']))], cwd=code, check=True)
    cache = Path(settings['cache_dir'])
    for key, suffix in {'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
                        'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'TORCH_HOME': 'torch',
                        'XDG_CACHE_HOME': 'xdg', 'MPLCONFIGDIR': 'matplotlib'}.items():
        os.environ[key] = str(cache / suffix)


def run(args):
    settings = options(args)
    if args.manifest:
        saved = json.loads(Path(args.manifest).read_text())
        settings = saved['settings']
        task_list = saved['tasks']
        if saved['group'] != args.group:
            raise ValueError('Manifest group does not match the launcher')
    else:
        task_list = tasks(args.group)
        if not args.dry_run:
            prepare_args = argparse.Namespace(**vars(args))
            prepare_args.groups = [args.group]
            prepare(prepare_args, emit=False)
    index = args.offset + (args.task_id if args.task_id is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    if not 0 <= index < len(task_list):
        raise ValueError(f'Task {index} outside 0..{len(task_list)-1}')
    task = task_list[index]
    output = destination(task, settings)
    argv, code = command(task, settings, output)
    print(shlex.join(argv), flush=True)
    if args.dry_run:
        return 0
    if not Path(argv[2]).is_file():
        raise FileNotFoundError(f'Trainer missing: {argv[2]}. Edit config.sh backend paths.')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(output / 'task.json', dict(**task, settings=settings, command=argv))
        summaries = list((output / 'results').rglob('metrics_summary.json'))
        if summaries:
            if len(summaries) != 1:
                raise ValueError('More than one summary exists for this task')
            prior = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
            if prior.get('status') == 'complete' and prior.get('command') == argv:
                print('Completed task already exists; keeping its result.', flush=True)
                return 0
            raise ValueError('Existing result has no matching completion record; choose a new campaign')
        atomic_json(output / 'status.json', dict(status='preparing', command=argv))
        try:
            if not args.skip_prefetch:
                prefetch(code, dict(settings, dataset=task['dataset']))
            atomic_json(output / 'status.json', dict(status='running', command=argv))
            result = subprocess.run(argv, cwd=code)
            summaries = list((output / 'results').rglob('metrics_summary.json'))
            success = result.returncode == 0 and len(summaries) == 1
            atomic_json(output / 'status.json', dict(status='complete' if success else 'failed',
                        returncode=result.returncode, command=argv, summaries=[str(p) for p in summaries]))
            return 0 if success else (result.returncode or 1)
        except Exception as error:
            atomic_json(output / 'status.json', dict(status='failed', command=argv, error=str(error)))
            raise


def prepare(args, emit=True):
    settings = options(args)
    if args.manifest:
        raise ValueError('Use --manifest only when executing a task')
    root = Path(settings['results_root'])
    records = []
    for group in args.groups or GROUPS:
        planned = tasks(group)
        manifest = root / FOLDERS[group] / settings['campaign'] / group / 'manifest.json'
        value = dict(group=group, settings=settings, tasks=planned)
        if manifest.exists() and json.loads(manifest.read_text()) != value:
            raise ValueError(f'Campaign already exists with different settings: {manifest}')
        atomic_json(manifest, value)
        records.append(dict(group=group, count=len(planned), manifest=str(manifest)))
    campaign_path = root / '_campaigns' / (settings['campaign'] + '.json')
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with campaign_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = json.loads(campaign_path.read_text()) if campaign_path.exists() else dict(groups=[])
        if existing.get('settings', settings) != settings:
            raise ValueError('Campaign already has different settings; choose another --campaign')
        by_group = {item['group']: item for item in existing['groups']}
        by_group.update({item['group']: item for item in records})
        campaign = dict(settings=settings, groups=list(by_group.values()))
        atomic_json(campaign_path, campaign)
    if emit:
        print(json.dumps(campaign, indent=2))
    return 0


LAUNCHERS = dict(zip(GROUPS, ['01_snip_only.sh', '02_all_blocks_dense.sh', '03_all_blocks_peft.sh',
                             '04_loo_peft.sh', '05_all_blocks_symbolic.sh', '06_correlation.sh',
                             '07_removal_sweep.sh']))


def submit(args):
    if not args.campaign and not os.environ.get('CAMPAIGN'):
        args.campaign = datetime.datetime.now(datetime.timezone.utc).strftime('study_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:6]
    if min(args.chunk_size, args.concurrency) < 1:
        raise ValueError('Chunk size and concurrency must be positive')
    settings = options(args)
    if not args.dry_run:
        for group in args.groups or GROUPS:
            key, script = BACKENDS[group]
            if not (Path(settings['backend_dirs'][key]) / script).is_file():
                raise FileNotFoundError(f"Missing {settings['backend_dirs'][key]}/{script}; edit config.sh")
    prepare(args)
    suite = Path(__file__).resolve().parents[1] / 'slurm/research_suite'
    os.environ['SUITE_DIR'] = str(suite)
    logs = Path(settings['results_root']) / 'logs' / settings['campaign']
    logs.mkdir(parents=True, exist_ok=True)
    submitted = []
    records_path = Path(settings['results_root']) / '_campaigns' / (settings['campaign'] + '_jobs.json')
    def record(item):
        import fcntl
        with records_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = json.loads(records_path.read_text()) if records_path.exists() else []
            records.append(item)
            atomic_json(records_path, records)
    for group in args.groups or GROUPS:
        manifest = Path(settings['results_root']) / FOLDERS[group] / settings['campaign'] / group / 'manifest.json'
        count = len(tasks(group))
        for offset in range(0, count, args.chunk_size):
            size = min(args.chunk_size, count - offset)
            argv = ['sbatch', '--parsable', f'--array=0-{size-1}%{args.concurrency}',
                    '--output', str(logs / f'{group}_{offset}_%A_%a.out'),
                    '--error', str(logs / f'{group}_{offset}_%A_%a.err'),
                    str(suite / LAUNCHERS[group]), '--manifest', str(manifest), '--offset', str(offset)]
            print(shlex.join(argv), flush=True)
            job_id = 'DRY_RUN' if args.dry_run else subprocess.check_output(argv, text=True).strip().split(';')[0]
            if not args.dry_run and not job_id.isdecimal():
                raise ValueError(f'Unexpected sbatch job ID: {job_id}')
            item = dict(group=group, offset=offset, count=size, job_id=job_id, command=argv)
            submitted.append(item)
            record(item)
    if not args.no_compile:
        ids = ':'.join(r['job_id'] for r in submitted)
        argv = ['sbatch', '--parsable', '--dependency=afterany:' + ids,
                '--output', str(logs / 'compile_%j.out'), '--error', str(logs / 'compile_%j.err'),
                str(suite / 'compile_results.sh'), '--campaign', settings['campaign'],
                '--out-dir', str(Path(settings['results_root']) / 'compiled_results' / settings['campaign'])]
        print(shlex.join(argv), flush=True)
        if not args.dry_run:
            compile_id = subprocess.check_output(argv, text=True).strip().split(';')[0]
            record(dict(group='compile', job_id=compile_id, command=argv))
    print(f"Campaign: {settings['campaign']}; results: {settings['results_root']}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['task', 'prepare', 'counts', 'submit'])
    p.add_argument('--group', choices=GROUPS)
    p.add_argument('--groups', nargs='+', choices=GROUPS)
    p.add_argument('--task-id', type=int)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--manifest')
    p.add_argument('--campaign')
    p.add_argument('--results-root')
    p.add_argument('--seed', type=int, default=18)
    p.add_argument('--epochs', type=int, default=-1)
    p.add_argument('--max-epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--head-warmup-epochs', type=int, default=5)
    p.add_argument('--distill-epochs', type=int, default=5)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-prefetch', action='store_true')
    p.add_argument('--chunk-size', type=int, default=1000)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--no-compile', action='store_true')
    args = p.parse_args()
    if not 0 <= args.seed < 2**32:
        p.error('Seed must be in 0..2^32-1')
    if args.campaign and (args.campaign in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.campaign)):
        p.error('Campaign must contain only letters, numbers, underscores or hyphens')
    if args.epochs != -1 and args.epochs <= 0:
        p.error('--epochs must be -1 or a positive integer')
    if min(args.max_epochs, args.patience, args.batch_size, args.num_samples) <= 0 or min(args.head_warmup_epochs, args.distill_epochs) < 0:
        p.error('Invalid training budget or sample settings')
    if args.action == 'counts':
        print(json.dumps({g: len(tasks(g)) for g in GROUPS}))
        return 0
    if args.action == 'prepare':
        return prepare(args)
    if args.action == 'submit':
        return submit(args)
    if not args.group:
        p.error('--group is required for task')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
