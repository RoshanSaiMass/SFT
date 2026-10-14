import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import research_scheduler as suite


def settings(tmp_path, custom=False):
    return dict(backend_dirs=dict(SNIP_DIR=str(ROOT/'SFT_SNIP_COMPACT'),
        ORIGINAL_DIR=str(ROOT/'SFT_ORIGINAL'), SYMBOLIC_DIR=str(ROOT/'SFT_EXPERIEMTN'),
        LOO_DIR=str(ROOT/'SFT_LOO_COMPACT'), LOO_ALL_DIR=str(ROOT/'SFT_LOO_ALL_PEFT'),
        SUBTRACTION_DIR=str(ROOT/'SFT_BLOCK_SUBTRACTION')),
        results_root=str(tmp_path), data_dir=str(tmp_path/'data'), cache_dir=str(tmp_path/'cache'),
        campaign='test', seed=18, epochs=-1, max_epochs=200, patience=10,
        num_samples=1000, batch_size=32, head_warmup_epochs=5, distill_epochs=5,
        datasets=suite.DATASETS, methods=None, filter_rank=96 if custom else 16,
        lora_rank=32 if custom else None, dora_rank=48 if custom else None,
        paca_rank=128 if custom else 32, paca_adapter_rank=64 if custom else None,
        lora_alpha=None)


def test_defaults_and_rank_selection(tmp_path):
    cfg=settings(tmp_path)
    assert [len(suite.tasks(g,cfg)) for g in suite.GROUPS]==[12,144,3168,240,144,264,3168,12,240]
    cfg=settings(tmp_path,True)
    configs=dict(suite.methods(False,cfg))
    def value(flags,key): return flags[flags.index(key)+1]
    assert value(configs['lora'],'--lora-rank')=='32'
    assert value(configs['dora'],'--lora-rank')=='48'
    assert value(configs['dora'],'--lora-alpha')=='96'
    assert value(configs['paca'],'--paca-rank')=='128'
    assert value(configs['rpaca_dora'],'--paca-adapter-rank')=='64'
    assert value(configs['rpaca_dora'],'--lora-alpha')=='128'
    cfg.update(datasets=['cifar100'],methods=['lora','dora'])
    assert len(suite.tasks('loo',cfg))==2
    assert len(suite.tasks('all_block_snip+peft',cfg))==24


@pytest.mark.parametrize('key',['ORIGINAL_DIR','SYMBOLIC_DIR','LOO_DIR','LOO_ALL_DIR','SNIP_DIR','SUBTRACTION_DIR'])
@pytest.mark.parametrize('custom',[False,True])
def test_actual_backend_parsers(tmp_path,key,custom):
    cfg=settings(tmp_path,custom)
    commands=[];script=None
    for group in suite.GROUPS:
        if suite.BACKENDS[group][0]!=key: continue
        for task in suite.tasks(group,cfg):
            if task['dataset']!='pets' or task['block'] not in (None,0): continue
            argv,_=suite.command(task,cfg,suite.destination(task,cfg))
            script=argv[2];commands.append(argv[3:])
    payload=tmp_path/'commands.json';payload.write_text(json.dumps(commands))
    source=r'''
import argparse,ast,json,os,sys
from pathlib import Path
path,payload=sys.argv[1:];commands=json.loads(Path(payload).read_text())
if Path(path).name=='train_sfp_lora.py':
    tree=ast.parse(Path(path).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    prefix=[]
    for node in fn.body:
        prefix.append(node)
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='parse_args':break
    fn.body=prefix+[ast.Return(value=ast.Name(id='args',ctx=ast.Load()))]
    tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ctx=dict(argparse=argparse,os=os,sys=sys,__file__=path,torch=type('Torch',(),{'cuda':type('Cuda',(),{'is_available':staticmethod(lambda:False)})}))
    exec(compile(tree,path,'exec'),ctx)
    for flags in commands:sys.argv=[path]+flags;ctx['main']()
else:
    sys.path.insert(0,str(Path(path).parent));m=__import__(Path(path).stem)
    for flags in commands:m.validate(m.parser().parse_args(flags))
'''
    result=subprocess.run([sys.executable,'-c',source,script,str(payload)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('location',['root','scheduler','spooled'])
@pytest.mark.parametrize('group',suite.GROUPS)
def test_launchers_locate_helper_with_stale_suite_dir(tmp_path,location,group):
    original=ROOT/'scheduler'/suite.LAUNCHERS[group]
    script=original
    if location=='spooled':
        script=tmp_path/'slurm_script';script.write_bytes(original.read_bytes())
    cwd=ROOT/'scheduler' if location=='scheduler' else ROOT
    env=dict(os.environ,DRY_RUN='1',SUITE_DIR='/bad/old/helper',SLURM_SUBMIT_DIR=str(cwd),
             SLURM_ARRAY_TASK_ID='0',RESULTS_ROOT=str(tmp_path/'results'))
    env.pop('SCHEDULER_DIR',None);env.pop('REPO_ROOT',None)
    result=subprocess.run(['bash',str(script),'--dry-run'],cwd=cwd,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    argv=shlex.split(result.stdout)
    assert str(ROOT) in argv[2]
    if group in ('snip_only','snip_compact_peft'):
        assert '--snip-prune-fraction' not in argv
        assert argv[argv.index('--snip-search-epochs')+1]=='11'
    if group=='loo_all_peft':assert argv[argv.index('--lora-rank')+1]=='4'


def test_submission_chunks_custom_grid_and_persists_ranks(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'tools/research_scheduler.py'),'submit',
        '--dry-run','--groups','all_block_snip+peft','loo_all_peft',
        '--methods','lora','dora','--datasets','pets','cifar100',
        '--filter-rank','64','--lora-rank','4','--dora-rank','3',
        '--results-root',str(tmp_path),'--campaign','rank_test','--chunk-size','25'],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    campaign=json.loads((tmp_path/'_campaigns/rank_test.json').read_text())
    assert campaign['settings']['filter_rank']==64 and campaign['settings']['dora_rank']==3
    assert {r['group']:r['count'] for r in campaign['groups']}=={'all_block_snip+peft':48,'loo_all_peft':4}
    assert '--array=0-24%4' in result.stdout and '--offset 25' in result.stdout
    assert str(ROOT/'scheduler/compile_results.slurm') in result.stdout


@pytest.mark.parametrize('flags',[['--lora-rank','0'],['--filter-rank','769'],
    ['--snip-search-epochs','4'],['--snip-momentum','nan'],
    ['--paca-rank','32','--paca-adapter-rank','64']])
def test_invalid_rank_settings_rejected(flags):
    result=subprocess.run([sys.executable,str(ROOT/'tools/research_scheduler.py'),'counts']+flags,
                          capture_output=True,text=True)
    assert result.returncode!=0


def test_targeted_loo_scheduler_routes_nine_terms_and_rejects_unsupported_backends(tmp_path):
    script = str(ROOT / 'tools/research_scheduler.py')
    result = subprocess.run([sys.executable, script, 'prepare', '--group', 'loo',
        '--datasets', 'pets', '--methods', 'lora', '--filter-rank', '128', '--lora-rank', '8',
        '--symbolic-max-terms', '9', '--results-root', str(tmp_path), '--campaign', 'nine_terms'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list(tmp_path.rglob('manifest.json'))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest['settings']['symbolic_max_terms'] == 9
    task = manifest['tasks'][0]
    argv, _ = suite.command(task, manifest['settings'], suite.destination(task, manifest['settings']))
    assert argv[argv.index('--symbolic-max-terms') + 1] == '9'
    result = subprocess.run([sys.executable, script, 'counts', '--group', 'snip_only',
                             '--symbolic-max-terms', '9'], capture_output=True, text=True)
    assert result.returncode != 0 and 'targeted LOO only' in result.stderr


def test_new_compiler_handles_extra_groups_and_validation_selection(tmp_path):
    import compile_final_research as compiler
    import csv
    cfg=settings(tmp_path);cfg.update(datasets=['pets'],methods=['lora'])
    for group in ['all_block_snip','loo_all_peft','snip_compact_peft']:
        planned=suite.tasks(group,cfg)
        suite.atomic_json(tmp_path/suite.FOLDERS[group]/'test'/group/'manifest.json',
                          dict(group=group,settings=cfg,tasks=planned))
        for task in planned:
            destination=suite.destination(task,cfg)
            folder=destination/'results/run';folder.mkdir(parents=True)
            block=task['block'] if task['block'] is not None else 1
            summary=dict(dataset='pets',seed=18,pruned_block_idx=[block],replaced_block_idx=block,
                peft_block_indices=[i for i in range(12) if i!=block] if group=='loo_all_peft' else [],
                best_val_acc=95 if block==7 else 50,final_test_acc=99 if block==0 else 75,
                config=dict(lora_rank=4,filter_rank=16),
                param_breakdown=dict(trainable_params=629829,total_params=80_000_000,trainable_pct=0.78),
                original_model_params=85_827_109)
            (folder/'metrics_summary.json').write_text(json.dumps(summary))
            (destination/'status.json').write_text(json.dumps(dict(status='complete')))
    args=argparse.Namespace(root=str(tmp_path),out_dir=str(tmp_path/'compiled'),campaign='test',
        evaluate_selected=False,evaluate_all_removals=False,subtraction_dir='',device='cpu',
        no_histories=False,no_plots=False)
    compiler.compile_results(args)
    with (tmp_path/'compiled/selected_by_validation.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert next(r for r in rows if r['group']=='all_block_snip')['block']=='7'
    assert {r['group'] for r in rows}=={'all_block_snip','loo_all_peft','snip_compact_peft'}
    assert (tmp_path/'compiled/index.html').is_file()
    assert list((tmp_path/'compiled').rglob('*.png'))
    assert list((tmp_path/'compiled').rglob('*.svg'))
    assert list((tmp_path/'compiled').rglob('*.pdf'))
