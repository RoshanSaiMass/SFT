import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from tools.collect_all import collect


def test_routes_and_collector(tmp_path):
    command, cwd = run.build_command("loo-all-peft", ["--lora-rank", "4"], caller=tmp_path)
    assert Path(command[2]) == ROOT / "SFT_LOO_ALL_PEFT/train_loo_all_peft.py"
    assert command[command.index("--output-dir")+1] == str(ROOT / "outputs/loo-all-peft")
    command, _ = run.build_command("evaluate-loo-all-peft", ["--summary", "result.json"], caller=tmp_path)
    assert Path(command[2]) == ROOT / "tools/evaluate_loo_all_peft.py"
    assert command[command.index("--summary")+1] == str(tmp_path / "result.json")
    folder = tmp_path / "runs"; folder.mkdir()
    (folder / "metrics_summary.json").write_text(json.dumps(dict(
        method="loo-all-peft", filter_type="symbolic", dataset="pets", seed=18,
        adapter_type="lora", mode="loo_compact_all_peft", pruned_block_idx=[0],
        peft_block_indices=list(range(1,12)), best_val_acc=50, final_test_acc=45)))
    rows = collect(folder, tmp_path / "all.csv")
    assert rows[0]["workflow"] == "loo-all-peft"
    assert json.loads(rows[0]["peft_block_indices"]) == list(range(1,12))


@pytest.mark.parametrize("task", range(12))
@pytest.mark.parametrize("epochs", ["-1", "100"])
def test_slurm_dataset_mapping_and_budget(tmp_path, task, epochs):
    script = ROOT / "slurm/run_loo_all_peft_a100.slurm"
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task),
               CODE_DIR=str(ROOT / "SFT_LOO_ALL_PEFT"), DATA_DIR=str(tmp_path / "data"),
               RESULTS_ROOT=str(tmp_path / "outputs"))
    result = subprocess.run(["bash", str(script), "--dry-run", "--epochs", epochs,
                             "--campaign", "rank4test"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    flags = shlex.split(result.stdout)
    datasets = ["pets", "cifar100", "flowers102", "dtd", "svhn", "fgvc_aircraft",
                "eurosat", "sun397", "pcam", "clevr", "dsprites-loc", "dsprites-ori"]
    def value(flag): return flags[flags.index(flag)+1]
    assert value("--dataset") == datasets[task]
    assert value("--filter-type") == "symbolic" and value("--filter-rank") == "16"
    assert value("--adapter-type") == "lora" and value("--lora-rank") == "4"
    assert value("--lora-alpha") == "8" and value("--epochs") == epochs
    assert str(tmp_path / "outputs/loo_all_rank4/rank4test") in value("--output-dir")
