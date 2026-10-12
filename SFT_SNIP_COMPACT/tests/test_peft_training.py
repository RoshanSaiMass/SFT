import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_sfp_lora as train
from test_peft_integration import CASES

@pytest.fixture
def tiny_workflow(monkeypatch):
    torch.set_num_threads(2)
    models=[]
    def factory(*a,**k):
        m=VisionTransformer(img_size=32,patch_size=16,embed_dim=8,depth=3,num_heads=2,num_classes=3)
        models.append(m);return m
    def loaders(args):
        g=torch.Generator().manual_seed(42)
        def loader(n):
            return DataLoader(TensorDataset(torch.randn(n,3,32,32,generator=g),torch.randint(0,3,(n,),generator=g)),batch_size=23)
        return loader(64),loader(8),loader(8),3,['a','b','c']
    monkeypatch.setattr(train.timm,'create_model',factory)
    monkeypatch.setattr(train,'get_dataloaders',loaders)
    monkeypatch.setattr(train,'plot_training_curves',lambda *a,**k:{})
    for fn in ('plot_lr_schedule','plot_param_breakdown','plot_snip_saliency'):
        monkeypatch.setattr(train,fn,lambda *a,**k:'synthetic-smoke-plot-omitted')
    return models

def arguments(case,tmp_path,multi=False):
    kind,tuner,init=case
    args=['train_sfp_lora.py','--dataset','pets','--device','cpu','--epochs','2',
          '--output-dir',str(tmp_path),'--save-misclassified-images','false',
          '--adapter-type',kind,'--lora-rank','2','--lora-alpha','4','--init-method',init]
    args+=['--num-filter-blocks','2'] if multi else ['--pruned-block','1']
    if kind in ('paca','rpaca'):
        args+=['--paca-rank','3','--paca-tuner',tuner]
        if tuner!='direct':args+=['--paca-adapter-rank','2']
    if kind in ('unilora','unidora'):args+=['--unilora-dim','11']
    return args

@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('multi',[False,True])
def test_real_trainer_runs_all_adapter_configurations(case,multi,tmp_path,monkeypatch,tiny_workflow):
    monkeypatch.setattr(sys,'argv',arguments(case,tmp_path,multi));train.main()
    paths=list(tmp_path.rglob('metrics_summary.json'));assert len(paths)==1
    s=json.loads(paths[0].read_text())
    assert s['adapter_type']==case[0]
    assert s['epochs_trained']==2
    assert s['param_breakdown']['total_params']==s['efficiency']['total_params']
    assert s['param_breakdown']['trainable_backbone']==0
    assert len(s['pruned_block_idx'])==(2 if multi else 1)
    if case[0]=='dora':assert s['mode']=='sft_dora'

@pytest.mark.parametrize('kind',['lora','dora'])
def test_compensated_loftq_injects_once(kind,tmp_path,monkeypatch,tiny_workflow):
    import single_filter_lora as sf
    calls=[];original=sf.LoRALinear._loftq_init
    def counted(self,*a,**k):calls.append(self);return original(self,*a,**k)
    monkeypatch.setattr(sf.LoRALinear,'_loftq_init',counted)
    args=arguments((kind,'direct','loftq'),tmp_path)+['--compensate-params']
    monkeypatch.setattr(sys,'argv',args);train.main()
    assert len(calls)==8
    s=json.loads(next(tmp_path.rglob('metrics_summary.json')).read_text())
    assert s['lora_rank']>2

@pytest.mark.parametrize('extra',[
    ['--mode','sft_lora_ortho','--adapter-type','paca'],
    ['--mode','sft_lora_ortho','--adapter-type','unilora'],
    ['--compensate-params','--num-filter-blocks','2'],
    ['--mode','sft','--adapter-type','paca','--paca-rank','3'],
])
def test_invalid_resolved_combinations_rejected_before_data(extra,monkeypatch):
    monkeypatch.setattr(sys,'argv',['train_sfp_lora.py']+extra)
    monkeypatch.setattr(train,'get_dataloaders',lambda args:pytest.fail('Invalid config reached dataset acquisition'))
    with pytest.raises(SystemExit) as e:train.main()
    assert e.value.code==2

@pytest.mark.parametrize('mode',['sft','full_finetune'])
def test_baseline_presets_still_work(mode,tmp_path,monkeypatch,tiny_workflow):
    args=['train_sfp_lora.py','--mode',mode,'--device','cpu','--epochs','2',
          '--output-dir',str(tmp_path),'--save-misclassified-images','false']
    if mode=='sft':args+=['--pruned-block','1']
    monkeypatch.setattr(sys,'argv',args);train.main()
    s=json.loads(next(tmp_path.rglob('metrics_summary.json')).read_text())
    assert s['mode']==mode
    assert s['param_breakdown']['lora']==0
    if mode=='full_finetune':assert s['efficiency']['trainable_params']==s['efficiency']['total_params']

@pytest.mark.parametrize('case',[
    ('lora','direct','default'),('dora','direct','default'),
    ('paca','lora','default'),('paca','dora','default'),
    ('rpaca','lora','default'),('rpaca','dora','default')])
def test_selective_orthogonal_preset_training(case,tmp_path,monkeypatch,tiny_workflow):
    args=arguments(case,tmp_path)
    pos=args.index('--pruned-block');del args[pos:pos+2]
    args+=['--mode','sft_lora_ortho','--num-ortho-blocks','1']
    monkeypatch.setattr(sys,'argv',args);train.main()
    s=json.loads(next(tmp_path.rglob('metrics_summary.json')).read_text())
    assert s['lora_ortho_lambda1']==s['lora_ortho_lambda2']==1e-4
    assert s['num_ortho_blocks']==1


def test_direct_paca_with_zero_lora_rank_uses_explicit_columns(tmp_path,monkeypatch,tiny_workflow):
    args=arguments(('paca','direct','default'),tmp_path)
    args[args.index('--lora-rank')+1]='0'
    monkeypatch.setattr(sys,'argv',args);train.main()
    s=json.loads(next(tmp_path.rglob('metrics_summary.json')).read_text())
    assert s['adapter_type']=='paca'
    assert s['paca_columns']==3
    assert s['param_breakdown']['lora']>0
