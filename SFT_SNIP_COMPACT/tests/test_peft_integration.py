import copy
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import single_filter_lora as sf

CASES = [('lora','direct','default'), ('dora','direct','default'),
         ('lora','direct','loftq'), ('dora','direct','loftq'),
         ('paca','direct','default'), ('rpaca','direct','default'),
         ('paca','lora','default'), ('paca','dora','default'),
         ('rpaca','lora','default'), ('rpaca','dora','default'),
         ('unilora','direct','default'), ('unidora','direct','default')]
ADAPTERS = (sf.LoRALinear, sf.PaCALinear, sf.PaCAAdapterLinear, sf.UniLoRAAdapterLinear)

def model_for(case, replaced=(1,)):
    kind,tuner,init=case
    torch.manual_seed(7)
    model=VisionTransformer(img_size=32,patch_size=16,embed_dim=8,depth=3,num_heads=2,num_classes=3)
    for i in replaced:
        sf.substitute_filter_block(model,i)
    sf.inject_lora(model,replaced,lora_rank=2,lora_alpha=4,adapter_type=kind,init_method=init,
                   paca_rank=3,paca_tuner=tuner,paca_adapter_rank=2,unilora_dim=11,unilora_seed=7)
    sf.freeze_non_trainable(model,replaced)
    return model

def expected_adapter_ids(model):
    return {id(p) for module in model.modules() if isinstance(module,ADAPTERS+(sf.UniLoRABank,))
            for p in module.parameters(recurse=False)}

@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('replaced',[(1,),(0,2)])
def test_placement_trainability_learning_frozen_weights_and_roundtrip(case,replaced):
    torch.set_num_threads(2)
    model=model_for(case,replaced)
    wrappers=[m for m in model.modules() if isinstance(m,ADAPTERS)]
    assert len(wrappers)==4*(3-len(replaced))
    for i in replaced:
        assert isinstance(model.blocks[i],sf.SingleFilterBlock)
        assert not any(isinstance(m,ADAPTERS) for m in model.blocks[i].modules())
    expected=expected_adapter_ids(model)
    expected.update(id(p) for i in replaced for p in model.blocks[i].parameters())
    expected.update(id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters())
    expected.update(id(p) for p in model.head.parameters())
    actual={id(p) for p in model.parameters() if p.requires_grad}
    assert actual==expected, 'Trainable set does not match adapter/filter/LN/head manifest'
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    frozen={n for n,p in model.named_parameters() if not p.requires_grad}
    adapter_before={id(p):p.detach().clone() for p in model.parameters() if id(p) in expected_adapter_ids(model)}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01,weight_decay=0)
    x=torch.randn(4,3,32,32);y=torch.tensor([0,1,2,0])
    for _ in range(2):
        opt.zero_grad();loss=F.cross_entropy(model(x),y);assert torch.isfinite(loss)
        loss.backward();opt.step()
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if n in frozen)
    assert any(not torch.equal(adapter_before[id(p)],p) for p in model.parameters() if id(p) in adapter_before)
    model.eval();prediction=model(x).detach()
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    reloaded=model_for(case,replaced);reloaded.load_state_dict(torch.load(buf,weights_only=True));reloaded.eval()
    assert torch.allclose(prediction,reloaded(x),atol=1e-6)
    stats=sf.count_parameter_breakdown(model,replaced)
    buckets=('filter_block','lora','layernorm','head','trainable_backbone','frozen_backbone')
    assert sum(stats[k] for k in buckets)==stats['total_params']
    assert stats['trainable_backbone']==0
    if case[0] in ('paca','rpaca'):
        logical=sum(p.numel() for p in model.parameters())
        for m in wrappers:
            logical+=m.frozen_weight.numel()+(m.paca_bias.numel() if m.paca_bias is not None else 0)
            if isinstance(m,sf.PaCALinear):logical-=m.paca_weight.numel()
        assert stats['total_params']==logical

@pytest.mark.parametrize('kind',['lora','dora','paca','paca_lora','paca_dora'])
def test_default_layers_preserve_base_output(kind):
    torch.manual_seed(1);base=nn.Linear(8,6);x=torch.randn(4,5,8);expected=base(x).detach()
    if kind in ('lora','dora'):layer=sf.LoRALinear(copy.deepcopy(base),rank=2,adapter_type=kind)
    elif kind=='paca':layer=sf.PaCALinear(copy.deepcopy(base),rank=3)
    else:layer=sf.PaCAAdapterLinear(copy.deepcopy(base),paca_rank=3,adapter_rank=2,adapter_type=kind.split('_')[1])
    assert torch.allclose(layer(x),expected,atol=1e-6)

@pytest.mark.parametrize('tuner',['direct','lora','dora'])
def test_resampling_preserves_committed_output(tuner):
    torch.manual_seed(2);base=nn.Linear(8,6);x=torch.randn(4,8)
    if tuner=='direct':layer=sf.PaCALinear(base,rank=3)
    else:layer=sf.PaCAAdapterLinear(base,paca_rank=3,adapter_rank=2,adapter_type=tuner)
    opt=torch.optim.AdamW(layer.parameters(),lr=.01)
    loss=layer(x).square().mean();loss.backward();opt.step()
    before=layer(x).detach();reset=layer.resample_columns();after=layer(x).detach()
    assert torch.allclose(before,after,atol=1e-6)
    for p in reset:opt.state.pop(p,None)
    assert all(p not in opt.state for p in reset)

@pytest.mark.parametrize('kind',['unilora','unidora'])
def test_unified_vector_is_registered_once_and_all_layers_reference_it(kind):
    model=model_for((kind,'direct','default'))
    bank=model.unilora_bank.theta_d
    assert sum(p is bank for p in model.parameters())==1
    layers=[m for m in model.modules() if isinstance(m,sf.UniLoRAAdapterLinear)]
    assert all(m._theta_ref[0] is bank for m in layers)
    copy_model=copy.deepcopy(model)
    assert all(m._theta_ref[0] is copy_model.unilora_bank.theta_d for m in copy_model.modules() if isinstance(m,sf.UniLoRAAdapterLinear))

@pytest.mark.parametrize('case',CASES)
def test_adapter_optimizer_group_contains_every_adapter_parameter(case):
    import train_sfp_lora as trainer
    model=model_for(case)
    groups=trainer.build_optimizer_param_groups(model,1e-3,3e-4)
    actual={id(p) for p in groups[1]['params']}
    assert actual==expected_adapter_ids(model)
    assert {id(p) for g in groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}

@pytest.mark.parametrize('case',CASES)
def test_diagnostic_reconstructs_saved_family_and_predictions(case,tmp_path,monkeypatch):
    import json
    import lora_delta_diagnostic as diagnostic
    kind,tuner,init=case
    model=model_for(case);model.eval();x=torch.randn(2,3,32,32)
    for p in model.parameters():
        if p.requires_grad:
            with torch.no_grad():p.add_(.01)
    expected=model(x).detach()
    checkpoint=tmp_path/'best.pt';torch.save(model.state_dict(),checkpoint)
    summary=dict(dataset='synthetic',num_classes=3,adapter_type=kind,lora_rank=2,lora_alpha=4,
                 pruned_block_idx=[1],paca_columns=3,paca_tuner=tuner,paca_adapter_rank=2,
                 paca_selection='random',unilora_dim=11,seed=7,checkpoint_path=str(checkpoint),
                 init_method=init,backbone='vit_base_patch16_224.augreg_in21k')
    (tmp_path/'metrics_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(diagnostic.timm,'create_model',lambda *a,**k: VisionTransformer(
        img_size=32,patch_size=16,embed_dim=8,depth=3,num_heads=2,num_classes=3))
    rebuilt,_=diagnostic.load_run_model(str(tmp_path))
    assert torch.allclose(expected,rebuilt(x),atol=1e-6)
    report=diagnostic.compute_lora_delta_report(rebuilt)
    assert len(report)==8
    assert all(torch.isfinite(torch.tensor(r['ratio_pct'])) for r in report)

@pytest.mark.parametrize('kind',['lora','dora'])
def test_forward_matches_explicit_effective_weight(kind):
    torch.manual_seed(17);layer=sf.LoRALinear(nn.Linear(8,6),rank=2,alpha=4,adapter_type=kind)
    with torch.no_grad():layer.lora_B.normal_()
    x=torch.randn(3,4,8);v=layer.base_layer.weight+layer.scaling*(layer.lora_B@layer.lora_A)
    if kind=='dora':v=layer.magnitude[:,None]*v/v.norm(dim=1,keepdim=True).clamp_min(1e-8)
    assert torch.allclose(layer(x),F.linear(x,v,layer.base_layer.bias),atol=1e-5)

@pytest.mark.parametrize('tuner',['lora','dora'])
def test_fused_delta_is_confined_to_selected_columns(tuner):
    layer=sf.PaCAAdapterLinear(nn.Linear(8,6),paca_rank=3,adapter_rank=2,adapter_type=tuner)
    with torch.no_grad():layer.lora_B.normal_()
    delta,_=layer._delta_subweight();weight=layer.frozen_weight.clone()
    weight[:,layer.selected_idx]+=delta
    x=torch.randn(3,8)
    assert torch.allclose(layer(x),F.linear(x,weight,layer.paca_bias),atol=1e-5)
    untouched=torch.ones(8,dtype=torch.bool);untouched[layer.selected_idx]=False
    assert torch.equal(weight[:,untouched],layer.frozen_weight[:,untouched])

def test_selective_orthogonality_reaches_only_selected_adapters():
    model=VisionTransformer(img_size=32,patch_size=16,embed_dim=8,depth=3,num_heads=2,num_classes=3)
    sf.substitute_filter_block(model,1)
    sf.inject_lora(model,[1],lora_rank=2,ortho_block_indices=[0])
    penalty=sf.compute_lora_orthogonality_loss(model,.2,.3);penalty.backward()
    for i in (0,2):
        layers=[m for m in model.blocks[i].modules() if isinstance(m,sf.LoRALinear)]
        assert all((m.lora_A.grad is not None)==(i==0) for m in layers)

@pytest.mark.parametrize('kwargs',[{'rank':9},{'alpha':0},{'loftq_bits':0},{'loftq_iters':0}])
def test_invalid_loftq_configuration_fails_clearly(kwargs):
    args=dict(rank=2,alpha=4,init_method='loftq');args.update(kwargs)
    with pytest.raises(ValueError,match='LoftQ'):sf.LoRALinear(nn.Linear(8,6),**args)

def test_dora_dropout_preserves_initial_pretrained_path():
    base=nn.Linear(8,6);x=torch.randn(4,8)
    layer=sf.LoRALinear(copy.deepcopy(base),rank=2,adapter_type='dora',dropout=.8)
    layer.train()
    assert torch.allclose(layer(x),base(x),atol=1e-6)

@pytest.mark.parametrize('kind',['dora','unidora'])
def test_dora_dropout_only_affects_low_rank_input(kind):
    base=nn.Linear(8,6);x=torch.randn(4,8)
    if kind=='dora':
        layer=sf.LoRALinear(base,rank=2,alpha=4,adapter_type=kind,dropout=1)
        with torch.no_grad():layer.lora_B.normal_()
        delta=layer.scaling*(layer.lora_B@layer.lora_A);m=layer.magnitude
    else:
        bank=sf.UniLoRABank(11)
        layer=sf.UniLoRAAdapterLinear(base,bank.theta_d,11,2,adapter_type=kind,dropout=1)
        a,b=layer._reconstruct_AB();delta=layer.scaling*(a@b).T;m=layer.lora_magnitude
    v=base.weight+delta;scale=m/v.norm(dim=1).clamp_min(1e-8)
    expected=F.linear(x,base.weight)*scale+base.bias
    layer.train()
    assert torch.allclose(layer(x),expected,atol=1e-6)


def test_analysis_keeps_fused_lora_and_dora_distinct():
    import analyse_results as analysis
    base=dict(adapter_type='paca',lora_rank=2)
    assert analysis.method_label_from_json(dict(base,paca_tuner='lora'))!=analysis.method_label_from_json(dict(base,paca_tuner='dora'))
