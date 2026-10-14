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
MAIN_GROUPS = ['snip_only', 'all_block_snip', 'all_block_snip+peft', 'loo', 'looall',
          'sub_correlation', 'sub_removal']
GROUPS = MAIN_GROUPS + ['loo_all_peft', 'snip_compact_peft']
FOLDERS = {g: 'sub_sft' if g.startswith('sub_') else g for g in GROUPS}
BACKENDS = {
    'snip_only': ('SNIP_DIR', 'train_snip_compact.py'),
    'all_block_snip': ('ORIGINAL_DIR', 'train_sfp_lora.py'),
    'all_block_snip+peft': ('ORIGINAL_DIR', 'train_sfp_lora.py'),
    'loo': ('LOO_DIR', 'train_loo_compact.py'),
    'looall': ('SYMBOLIC_DIR', 'train_sfp_lora.py'),
    'sub_correlation': ('SUBTRACTION_DIR', 'train_subtraction.py'),
    'sub_removal': ('SUBTRACTION_DIR', 'train_subtraction.py'),
    'loo_all_peft': ('LOO_ALL_DIR', 'train_loo_all_peft.py'),
    'snip_compact_peft': ('SNIP_DIR', 'train_snip_compact.py'),
}


def methods(include_uni=True, settings=None, default_rank=16):
    settings = settings or {}
    lora_rank = settings.get('lora_rank') or default_rank
    dora_rank = settings.get('dora_rank') or lora_rank
    columns = settings.get('paca_rank', 32)
    result = []
    for family in ('lora', 'dora', 'paca', 'rpaca'):
        tuners = ('direct', 'lora', 'dora') if 'paca' in family else (None,)
        for tuner in tuners:
            name = family if tuner in (None, 'direct') else family + '_' + tuner
            flags = ['--adapter-type', family]
            if tuner is None:
                rank = dora_rank if family == 'dora' else lora_rank
                flags += ['--lora-rank', str(rank), '--lora-alpha', str(settings.get('lora_alpha') or 2 * rank)]
            elif tuner == 'direct':
                flags += ['--lora-rank', str(columns), '--paca-rank', str(columns), '--paca-tuner', 'direct']
            else:
                rank = settings.get('paca_adapter_rank') or (dora_rank if tuner == 'dora' else lora_rank)
                flags += ['--paca-tuner', tuner, '--paca-rank', str(columns), '--paca-adapter-rank', str(rank),
                          '--lora-alpha', str(settings.get('lora_alpha') or 2 * rank)]
            result.append((name, flags))
            if tuner != 'direct':
                for value in ('0.9', '1.5'):
                    result.append((name + '_ortho_' + value,
                                   flags + ['--lora-ortho-lambda1', value,
                                            '--lora-ortho-lambda2', value]))
    if include_uni:
        for family in ('unilora', 'unidora'):
            result.append((family, ['--adapter-type', family, '--lora-rank', str(lora_rank),
                                    '--lora-alpha', str(settings.get('lora_alpha') or 2 * lora_rank),
                                    '--unilora-dim', '72000']))
    return result


def tasks(group, settings=None):
    settings = settings or {}
    configs = methods(include_uni=group not in ('loo', 'loo_all_peft', 'snip_compact_peft'),
                      settings=settings, default_rank=4 if group == 'loo_all_peft' else 16) if group in (
        'all_block_snip+peft', 'loo', 'sub_correlation', 'sub_removal', 'loo_all_peft', 'snip_compact_peft') else [('none', [])]
    selected = settings.get('methods') or (['lora'] if group == 'loo_all_peft' else None)
    if selected and configs != [('none', [])]:
        available = {name for name, flags in configs}
        if set(selected) - available:
            raise ValueError(f"Unavailable methods for {group}: {sorted(set(selected)-available)}")
        configs = [(name, flags) for name, flags in configs if name in selected]
    blocks = range(12) if group in ('all_block_snip', 'all_block_snip+peft', 'looall', 'sub_removal') else [None]
    return [dict(task_id=i, group=group, dataset=dataset, method=method,
                 block=block, method_flags=flags)
            for i, (dataset, method, block, flags) in enumerate(
                (d, m, b, f) for d in settings.get('datasets', DATASETS) for m, f in configs for b in blocks)]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def options(args):
    base = Path(os.environ.get('CLUSTER_BASE', '/export/home/achyut/Sarvesh'))
    repo = Path(os.environ.get('REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
    defaults = dict(SNIP_DIR=repo/'SFT_SNIP_COMPACT', ORIGINAL_DIR=repo/'SFT_ORIGINAL',
                    SYMBOLIC_DIR=repo/'SFT_EXPERIEMTN', LOO_DIR=repo/'SFT_LOO_COMPACT',
                    LOO_ALL_DIR=repo/'SFT_LOO_ALL_PEFT', SUBTRACTION_DIR=repo/'SFT_BLOCK_SUBTRACTION')
    return dict(backend_dirs={key: str(Path(os.environ.get(key, value)).expanduser().resolve())
                              for key, value in defaults.items()},
                results_root=str(Path(args.results_root or os.environ.get('RESULTS_ROOT', base / 'research_results')).resolve()),
                data_dir=str(Path(os.environ.get('DATA_DIR', base / 'SFT_BEST/data')).resolve()),
                cache_dir=str(Path(os.environ.get('CACHE_DIR', base / 'SFT_BEST/data/research_cache')).resolve()),
                campaign=args.campaign or os.environ.get('CAMPAIGN') or 'manual_' + os.environ.get('SLURM_ARRAY_JOB_ID', 'local'),
                seed=args.seed, epochs=args.epochs, max_epochs=args.max_epochs, patience=args.patience,
                num_samples=args.num_samples, batch_size=args.batch_size,
                head_warmup_epochs=args.head_warmup_epochs, distill_epochs=args.distill_epochs,
                datasets=args.datasets or DATASETS, methods=args.methods,
                filter_rank=args.filter_rank, lora_rank=args.lora_rank, dora_rank=args.dora_rank,
                paca_rank=args.paca_rank, paca_adapter_rank=args.paca_adapter_rank,
                lora_alpha=args.lora_alpha, compact_type=args.compact_type,
                symbolic_max_terms=args.symbolic_max_terms, symbolic_penalty=args.symbolic_penalty,
                symbolic_search_tokens=args.symbolic_search_tokens,
                loo_samples=args.loo_samples, snip_samples=args.snip_samples,
                snip_momentum=args.snip_momentum, snip_search_epochs=args.snip_search_epochs,
                snip_filter_type=args.snip_filter_type)


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
        flags += ['--filter-type', settings.get('snip_filter_type', 'dense'), '--adapter-type', 'none']
    elif group in ('all_block_snip', 'all_block_snip+peft', 'looall'):
        flags += ['--pruned-block', str(task['block']), '--filter-block-layers', '1',
                  '--filter-residual-hidden-dim', '0', '--filter-dropout', '0']
        if group != 'all_block_snip+peft':
            flags += ['--mode', 'sft']
        if group == 'looall':
            flags += ['--filter-type', settings.get('compact_type', 'symbolic')]
    elif group in ('loo', 'loo_all_peft'):
        flags += ['--filter-type', settings.get('compact_type', 'symbolic'), '--loo-samples', str(settings.get('loo_samples', 512)), '--loo-batch-size', '32',
                  '--head-warmup-epochs', str(settings['head_warmup_epochs'])]
    elif group == 'snip_compact_peft':
        flags += ['--filter-type', settings.get('compact_type', 'symbolic')]
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
        flags += task['method_flags']
    if group in ('snip_only', 'snip_compact_peft'):
        flags += ['--snip-momentum', str(settings.get('snip_momentum', 0.3)),
                  '--snip-samples', str(settings.get('snip_samples', 64)), '--snip-batch-size', '32',
                  '--snip-search-epochs', str(settings.get('snip_search_epochs', 11))]
    if group in ('snip_only', 'snip_compact_peft', 'loo', 'loo_all_peft', 'looall'):
        flags += ['--filter-rank', str(settings.get('filter_rank', 16)),
                  '--symbolic-max-terms', str(settings.get('symbolic_max_terms', 2)),
                  '--symbolic-penalty', str(settings.get('symbolic_penalty', 0.001)),
                  '--symbolic-search-tokens', str(settings.get('symbolic_search_tokens', 2048))]
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
        task_list = tasks(args.group, settings)
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
    for group in args.groups or MAIN_GROUPS:
        planned = tasks(group, settings)
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


LAUNCHERS = dict(zip(GROUPS, ['01_snip_only.slurm', '02_all_blocks_dense.slurm', '03_all_blocks_peft.slurm',
                             '04_loo_peft.slurm', '05_all_blocks_symbolic.slurm', '06_correlation.slurm',
                             '07_removal_sweep.slurm', '08_loo_all_peft.slurm', '09_snip_compact_peft.slurm']))


def submit(args):
    if not args.campaign and not os.environ.get('CAMPAIGN'):
        args.campaign = datetime.datetime.now(datetime.timezone.utc).strftime('study_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:6]
    if min(args.chunk_size, args.concurrency) < 1:
        raise ValueError('Chunk size and concurrency must be positive')
    settings = options(args)
    if not args.dry_run:
        for group in args.groups or MAIN_GROUPS:
            key, script = BACKENDS[group]
            if not (Path(settings['backend_dirs'][key]) / script).is_file():
                raise FileNotFoundError(f"Missing {settings['backend_dirs'][key]}/{script}; edit config.sh")
    prepare(args)
    suite = Path(__file__).resolve().parents[1] / 'scheduler'
    os.environ['SUITE_DIR'] = str(suite)
    os.environ['SCHEDULER_DIR'] = str(suite)
    os.environ['REPO_ROOT'] = str(suite.parent)
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
    for group in args.groups or MAIN_GROUPS:
        manifest = Path(settings['results_root']) / FOLDERS[group] / settings['campaign'] / group / 'manifest.json'
        count = len(tasks(group, settings))
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
                str(suite / 'compile_results.slurm'), '--campaign', settings['campaign'],
                '--root', settings['results_root'], '--subtraction-dir', settings['backend_dirs']['SUBTRACTION_DIR'],
                '--out-dir', str(Path(settings['results_root']) / 'compiled_results' / settings['campaign'])]
        if not args.no_evaluate_selected:
            argv.append('--evaluate-selected')
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
    p.add_argument('--datasets', nargs='+', choices=DATASETS+['caltech101'])
    p.add_argument('--methods', nargs='+')
    p.add_argument('--filter-rank', '--compact-rank', type=int, default=16)
    p.add_argument('--lora-rank', type=int)
    p.add_argument('--dora-rank', type=int)
    p.add_argument('--paca-rank', type=int, default=32)
    p.add_argument('--paca-adapter-rank', type=int)
    p.add_argument('--lora-alpha', type=float)
    p.add_argument('--compact-type', choices=['dense','lowrank','symbolic'], default='symbolic')
    p.add_argument('--symbolic-max-terms', type=int, default=2)
    p.add_argument('--symbolic-penalty', type=float, default=0.001)
    p.add_argument('--symbolic-search-tokens', type=int, default=2048)
    p.add_argument('--loo-samples', type=int, default=512)
    p.add_argument('--snip-samples', type=int, default=64)
    p.add_argument('--snip-momentum', type=float, default=0.3)
    p.add_argument('--snip-search-epochs', type=int, default=11)
    p.add_argument('--snip-filter-type', choices=['dense','lowrank','symbolic'], default='dense')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-prefetch', action='store_true')
    p.add_argument('--chunk-size', type=int, default=1000)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--no-evaluate-selected', action='store_true',
                   help='Compile without evaluating validation-selected removal checkpoints')
    args = p.parse_args()
    if args.action != 'task' and args.group and not args.groups:
        args.groups = [args.group]
    import math
    if not 1 <= args.filter_rank <= 768 or not 1 <= args.paca_rank <= 768:
        p.error('Filter and PaCA ranks must be 1..768')
    if any(value is not None and value < 1 for value in (args.lora_rank,args.dora_rank,args.paca_adapter_rank)):
        p.error('Adapter ranks must be positive')
    if args.lora_alpha is not None and (not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0):
        p.error('Alpha must be finite and positive')
    if args.snip_search_epochs < 11 or not 0 <= args.snip_momentum < 1:
        p.error('Sequential SNIP needs at least 11 search epochs and momentum in [0,1)')
    if not 1 <= args.symbolic_max_terms <= 9 or not math.isfinite(args.symbolic_penalty) or args.symbolic_penalty < 0:
        p.error('Invalid symbolic settings')
    if min(args.symbolic_search_tokens,args.loo_samples,args.snip_samples) < 1:
        p.error('Calibration sample and token counts must be positive')
    settings = options(args)
    if args.datasets and len(set(args.datasets)) != len(args.datasets):
        p.error('Dataset names must be unique')
    if args.methods and len(set(args.methods)) != len(args.methods):
        p.error('Method names must be unique')
    for group in args.groups or ([args.group] if args.group else MAIN_GROUPS):
        if args.symbolic_max_terms > 3 and group != 'loo':
            p.error('Term limits 4..9 are currently supported by targeted LOO only; use --groups loo')
        for task in tasks(group, settings):
            name, flags = task['method'], task['method_flags']
            if '--paca-adapter-rank' in flags and int(flags[flags.index('--paca-adapter-rank')+1]) > args.paca_rank:
                p.error(f'{name}: fused adapter rank cannot exceed --paca-rank')
    if not 0 <= args.seed < 2**32:
        p.error('Seed must be in 0..2^32-1')
    if args.campaign and (args.campaign in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.campaign)):
        p.error('Campaign must contain only letters, numbers, underscores or hyphens')
    if args.epochs != -1 and args.epochs <= 0:
        p.error('--epochs must be -1 or a positive integer')
    if min(args.max_epochs, args.patience, args.batch_size, args.num_samples) <= 0 or min(args.head_warmup_epochs, args.distill_epochs) < 0:
        p.error('Invalid training budget or sample settings')
    if args.action == 'counts':
        print(json.dumps({g: len(tasks(g, settings)) for g in args.groups or GROUPS}))
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
