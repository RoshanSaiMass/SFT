import json
import sys

import pytest

from test_peft_training import tiny_workflow, arguments
from test_peft_integration import CASES
import train_sfp_lora as train


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("filter_type", ["lowrank", "symbolic"])
@pytest.mark.parametrize("multi", [False, True])
def test_actual_trainer_with_all_peft_variants(case, filter_type, multi, tmp_path, monkeypatch, tiny_workflow):
    argv = arguments(case, tmp_path, multi) + ["--filter-type", filter_type, "--filter-rank", "2"]
    monkeypatch.setattr(sys, "argv", argv); train.main()
    summary = json.loads(next(tmp_path.rglob("metrics_summary.json")).read_text())
    assert summary["filter_type"] == filter_type
    assert summary["compression"]["filter_reduction_pct"] > 0
    assert summary["compression"]["total_model_params"] == summary["param_breakdown"]["total_params"]
    equations = json.loads(next(tmp_path.rglob("filter_equations.json")).read_text())
    assert len(equations) == (2 if multi else 1)
    assert all(e["type"] == filter_type for e in equations)


@pytest.mark.parametrize("filter_type", ["dense", "lowrank", "symbolic"])
def test_manual_multiple_blocks_and_plain_sft(filter_type, tmp_path, monkeypatch, tiny_workflow):
    argv = ["train_sfp_lora.py", "--mode", "sft", "--replace-blocks", "2,0",
            "--filter-type", filter_type, "--filter-rank", "2", "--device", "cpu", "--epochs", "2",
            "--output-dir", str(tmp_path), "--save-misclassified-images", "false"]
    monkeypatch.setattr(sys, "argv", argv); train.main()
    summary = json.loads(next(tmp_path.rglob("metrics_summary.json")).read_text())
    assert summary["pruned_block_idx"] == [0, 2]
    assert summary["block_selection_method"] == "manual"
    assert summary["param_breakdown"]["lora"] == 0


@pytest.mark.parametrize("extra", [
    ["--num-filter-blocks", "12"],
    ["--replace-blocks", ",".join(str(i) for i in range(12))],
    ["--filter-rank", "0"], ["--symbolic-penalty", "nan"],
    ["--replace-blocks", "1,1"], ["--replace-blocks", "12"],
    ["--filter-type", "symbolic", "--filter-block-layers", "2"],
    ["--filter-type", "symbolic", "--mode", "full_finetune"],
])
def test_invalid_configs_fail_before_dataset_access(extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sfp_lora.py"] + extra)
    monkeypatch.setattr(train, "get_dataloaders", lambda _: pytest.fail("Invalid config accessed data"))
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
