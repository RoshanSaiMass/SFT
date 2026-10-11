import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import research_suite as suite
import compile_research as compiler


def settings(tmp_path):
    return dict(backend_dirs={
        'ORIGINAL_DIR': str(ROOT / 'SFT_ORIGINAL'),
        'SYMBOLIC_DIR': str(ROOT / 'SFT_EXPERIEMTN'),
        'LOO_DIR': str(ROOT / 'SFT_LOO_COMPACT'),
        'SUBTRACTION_DIR': str(ROOT / 'SFT_BLOCK_SUBTRACTION'),
        'SNIP_DIR': str(tmp_path / 'external_snip'),
    }, results_root=str(tmp_path), data_dir=str(tmp_path / 'data'), cache_dir=str(tmp_path / 'cache'),
       campaign='test', seed=18, epochs=-1, max_epochs=200, patience=10,
       num_samples=1000, batch_size=32, head_warmup_epochs=5, distill_epochs=5)


def test_complete_grid_and_original_peft_settings(tmp_path):
    expected = [12, 144, 3168, 240, 144, 264, 3168]
    assert [len(suite.tasks(g)) for g in suite.GROUPS] == expected
    assert sum(expected) == 7140
    assert len(suite.methods()) == 22
    assert len(suite.methods(False)) == 20
    cfg = settings(tmp_path)
    for group in suite.GROUPS:
        names = set()
        for task in suite.tasks(group):
            assert task['dataset'] != 'caltech101'
            assert task['task_id'] == len(names)
            path = suite.destination(task, cfg)
            assert str(path) not in names
            names.add(str(path))
            argv, _ = suite.command(task, cfg, path)
            if group == 'all_block_snip+peft':
                assert '--mode' not in argv
                assert '--pruned-block' in argv
            if group == 'loo':
                assert task['method'] not in ('unilora', 'unidora')
                assert '--loo-samples' in argv
            if group == 'sub_removal':
                assert '--defer-test' in argv
            if group == 'sub_correlation':
                assert argv[argv.index('--min-correlation')+1] == '-1'
            if 'ortho' in task['method']:
                a = argv[argv.index('--lora-ortho-lambda1')+1]
                assert a in ('0.9', '1.5')
                assert argv[argv.index('--lora-ortho-lambda2')+1] == a


@pytest.mark.parametrize('key', ['ORIGINAL_DIR', 'SYMBOLIC_DIR', 'LOO_DIR', 'SUBTRACTION_DIR'])
def test_every_generated_command_parses_in_its_actual_backend(tmp_path, key):
    cfg = settings(tmp_path)
    commands = []
    filename = None
    for group in suite.GROUPS:
        if suite.BACKENDS[group][0] != key:
            continue
        # All datasets/blocks are grid-invariant flags; cover each distinct PEFT config.
        for task in suite.tasks(group):
            if task['dataset'] != 'pets' or task['block'] not in (None, 0):
                continue
            argv, _ = suite.command(task, cfg, suite.destination(task, cfg))
            filename = argv[2]
            commands.append(argv[3:])
    payload = tmp_path / f'{key}.json'
    payload.write_text(json.dumps(commands))
    source = r'''
import ast,argparse,json,sys,os
from pathlib import Path
path, payload=sys.argv[1:]
commands=json.loads(Path(payload).read_text())
if Path(path).name=='train_sfp_lora.py':
    tree=ast.parse(Path(path).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    prefix=[]
    for node in main.body:
        prefix.append(node)
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='parse_args':
            break
    prefix.append(ast.Return(value=ast.Name(id='args',ctx=ast.Load())))
    main.body=prefix
    module=ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))
    context={'argparse':argparse,'sys':sys,'os':os,'__file__':path,'torch':type('Torch',(),{'cuda':type('Cuda',(),{'is_available':staticmethod(lambda:False)})})}
    exec(compile(module,path,'exec'),context)
    for flags in commands:
        sys.argv=[path]+flags
        context['main']()
else:
    sys.path.insert(0,str(Path(path).parent))
    module=__import__(Path(path).stem)
    for flags in commands:
        args=module.parser().parse_args(flags)
        module.validate(args)
print('Validated',len(commands),'commands')
'''
    result = subprocess.run([sys.executable, '-c', source, filename, str(payload)],
                            capture_output=True, text=True, env=dict(os.environ, MPLCONFIGDIR=str(tmp_path/'mpl')))
    assert result.returncode == 0, result.stdout + result.stderr


def create_run(root, group, task, cfg, val, test):
    directory = suite.destination(task, cfg)
    results = directory / 'results/run'
    results.mkdir(parents=True)
    summary = dict(dataset=task['dataset'], seed=cfg['seed'], pruned_block_idx=[task['block']],
                   best_val_acc=val, final_test_acc=test, adapter_type='dora', epochs_trained=10,
                   param_breakdown=dict(trainable_params=1000, total_params=100000, trainable_pct=1),
                   original_model_params=120000, config=dict(lora_ortho_lambda1=0, lora_ortho_lambda2=0))
    (results / 'metrics_summary.json').write_text(json.dumps(summary))
    (directory / 'status.json').write_text(json.dumps(dict(status='complete')))
    (results / 'training_history.csv').write_text('epoch,train_loss,val_loss,val_acc\n1,2,3,40\n2,1,2,50\n')
    return results


def test_compilation_selects_validation_and_preserves_missing_tests_and_figures(tmp_path):
    cfg = settings(tmp_path)
    group = 'all_block_snip'
    planned = suite.tasks(group)
    manifest = tmp_path / group / 'test' / group / 'manifest.json'
    suite.atomic_json(manifest, dict(group=group, settings=cfg, tasks=planned))
    for task in planned[:12]:
        create_run(tmp_path, group, task, cfg, 95 if task['block'] == 7 else 40,
                   99 if task['block'] == 0 else 75)
    args = argparse.Namespace(root=str(tmp_path), out_dir=str(tmp_path/'compiled'), campaign='test',
                              evaluate_selected=False, evaluate_all_removals=False, subtraction_dir='',
                              device='cpu', no_histories=False, no_plots=False)
    compiler.compile_results(args)
    import csv
    output = Path(args.out_dir)
    with (output/'selected_by_validation.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]['block'] == '7' and rows[0]['final_test_acc'] == '75.0'
    assert list((output/'figures').glob('*.png'))
    assert list((output/'figures').glob('*.svg'))
    assert (output/'index.html').exists()
    coverage = compiler.read_runs(tmp_path, 'test')[2]
    dense_coverage = next(r for r in coverage if r['group'] == group)
    assert dense_coverage['complete'] == 12 and dense_coverage['missing_or_running'] == 132
    # A high-test, incomplete sweep must never produce a winner.
    subset = compiler.read_runs(tmp_path, 'test')[0][:-1]
    selected, incomplete = compiler.select(subset)
    assert not selected and incomplete
    deferred = dict(subset[0], group='sub_removal', final_test_acc=None)
    assert compiler.number(deferred['final_test_acc']) is None


def test_all_loo_candidate_indices_are_preserved(tmp_path):
    summary = tmp_path/'metrics_summary.json'
    summary.write_text(json.dumps(dict(loo_selection=dict(rows=[dict(block=i,loss_increase=i*.1) for i in range(12)]))))
    row = dict(summary_path=str(summary), campaign='test', group='loo', dataset='pets',
               method_config='lora', seed=18, block=0, history_csv=None)
    compiler.auxiliary([row], tmp_path/'compiled')
    import csv
    with (tmp_path/'compiled/loo_scores_long.csv').open() as handle:
        scores = list(csv.DictReader(handle))
    assert [int(r['candidate_block']) for r in scores] == list(range(12))
    assert all(r['block'] == '0' for r in scores)


@pytest.mark.parametrize('evaluate_all', [False, True])
def test_removal_evaluation_tests_validation_winner_by_default(tmp_path, evaluate_all):
    cfg = settings(tmp_path)
    group = 'sub_removal'
    planned = suite.tasks(group)
    manifest = tmp_path/'sub_sft/test'/group/'manifest.json'
    suite.atomic_json(manifest, dict(group=group, settings=cfg, tasks=planned))
    for task in planned[:12]:
        create_run(tmp_path, group, task, cfg, 90 if task['block'] == 7 else 30, None)
    evaluator = tmp_path/'evaluator'
    evaluator.mkdir()
    (evaluator/'train_subtraction.py').write_text(
        'import json\nfrom pathlib import Path\n'
        'def evaluate_summary(path,device):\n'
        ' p=Path(path);s=json.loads(p.read_text());s["final_test_acc"]=77;s["test_evaluated"]=True;p.write_text(json.dumps(s))\n')
    args = argparse.Namespace(root=str(tmp_path), out_dir=str(tmp_path/'compiled'), campaign='test',
                              evaluate_selected=True, evaluate_all_removals=evaluate_all,
                              subtraction_dir=str(evaluator), device='cpu', no_histories=True, no_plots=True)
    compiler.compile_results(args)
    rows = compiler.read_runs(tmp_path, 'test')[0]
    assert sum(r['final_test_acc'] is not None for r in rows) == (12 if evaluate_all else 1)
    selected, _ = compiler.select(rows)
    assert len(selected) == 1 and selected[0]['block'] == 7 and selected[0]['final_test_acc'] == 77


def test_submission_chunks_cover_all_tasks_and_schedule_compiler(tmp_path):
    command = [sys.executable, str(ROOT/'tools/research_suite.py'), 'submit', '--dry-run',
               '--results-root', str(tmp_path), '--campaign', 'chunks',
               '--groups', 'all_block_snip+peft', '--chunk-size', '1000', '--epochs', '100']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    jobs = json.loads((tmp_path/'_campaigns/chunks_jobs.json').read_text())
    assert [j['offset'] for j in jobs] == [0, 1000, 2000, 3000]
    assert sum(j['count'] for j in jobs) == 3168
    assert '--dependency=afterany:' in result.stdout
    manifest = json.loads(Path(jobs[0]['command'][-3]).read_text())
    assert manifest['settings']['epochs'] == 100
    command[command.index('--groups')+1] = 'looall'
    subprocess.run(command + ['--no-compile'], capture_output=True, text=True, check=True)
    records = json.loads((tmp_path/'_campaigns/chunks_jobs.json').read_text())
    assert len(records) == 5 and records[-1]['group'] == 'looall'


@pytest.mark.parametrize('script', sorted((ROOT/'slurm/research_suite').glob('*.sh')))
def test_shell_syntax(script):
    subprocess.run(['bash', '-n', str(script)], check=True)


def test_array_wrapper_supports_spooled_location_and_offset(tmp_path):
    fake_python = tmp_path/'python3'
    capture = tmp_path/'args.json'
    fake_python.write_text(f'#!{sys.executable}\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    script = tmp_path/'spooled.sh'
    script.write_bytes((ROOT/'slurm/research_suite/03_all_blocks_peft.sh').read_bytes())
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'], DRY_RUN='1',
               SUITE_DIR=str(ROOT/'slurm/research_suite'), SLURM_SUBMIT_DIR=str(tmp_path),
               CLUSTER_BASE=str(tmp_path), RESULTS_ROOT=str(tmp_path/'results'), DATA_DIR=str(tmp_path/'data'),
               CACHE_DIR=str(tmp_path/'cache'), CAPTURE=str(capture))
    subprocess.run(['bash', str(script), '--offset', '3000'], env=env, check=True)
    args = json.loads(capture.read_text())
    assert args[1] == str(ROOT/'slurm/research_suite/../../tools/research_suite.py')
    assert args[args.index('--group')+1] == 'all_block_snip+peft'
    assert args[args.index('--offset')+1] == '3000'
