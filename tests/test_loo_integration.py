import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from tools.collect_all import collect


def test_loo_route_defaults_and_evaluation_paths(tmp_path):
    command, cwd = run.build_command("loo-compact", ["--loo-samples", "512"], caller=tmp_path)
    assert cwd == ROOT / "SFT_LOO_COMPACT"
    assert command[command.index("--data-dir")+1] == str(ROOT / "data")
    assert command[command.index("--output-dir")+1] == str(ROOT / "outputs/loo-compact")
    command, cwd = run.build_command("evaluate-loo-compact", ["--summary", "./result.json"], caller=tmp_path)
    assert command[-1] == str(tmp_path / "result.json")


def test_loo_collection_preserves_distinct_workflow_and_both_blocks(tmp_path):
    folder = tmp_path / "one_run"; folder.mkdir()
    (folder / "metrics_summary.json").write_text(json.dumps(dict(
        method="loo-compact", filter_type="symbolic", dataset="pets", seed=18,
        pruned_block_idx=[3], peft_block_idx=8, best_val_acc=90.5, final_test_acc=89.1,
        loo_selection=dict(baseline_loss=1.0), trainable_pct_original=0.2)))
    output = tmp_path / "all.csv"; collect(tmp_path, output)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["workflow"] == "loo-compact"
    assert rows[0]["block_indices"] == "[3]" and rows[0]["peft_block_idx"] == "8"
    assert rows[0]["loo_selection.baseline_loss"] == "1.0"
    assert rows[0]["final_test_acc"] == "89.1" and rows[0]["trainable_pct_original"] == "0.2"


@pytest.mark.parametrize("task,kind,tuner", [
    (0, "lora", "direct"), (1, "dora", "direct"),
    (2, "paca", "direct"), (3, "rpaca", "direct"),
    (4, "paca", "lora"), (5, "paca", "dora"),
    (6, "rpaca", "lora"), (7, "rpaca", "dora")])
def test_a100_launcher_dispatches_each_method_and_forwards_overrides(task, kind, tuner, tmp_path):
    # Stub only environment activation and the final Python call. Execute the
    # actual shell launcher, without Slurm, CUDA, downloads or training.
    executable = tmp_path / "python3"
    executable.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                          "open(os.environ['CAPTURE_ARGS'],'w').write(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o755)
    capture = tmp_path / "arguments.json"
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ["PATH"],
               CODE_DIR=str(ROOT), CAPTURE_ARGS=str(capture), SLURM_ARRAY_TASK_ID=str(task),
               SLURM_ARRAY_JOB_ID="321", SLURM_CPUS_PER_TASK="4")
    script = ROOT / "slurm/run_loo_compact_a100.sh"
    result = subprocess.run(["bash", "-c",
        'source() { :; }; conda() { :; }; export -f source conda; exec bash "$@"',
        "mock-launch", str(script), "--dataset", "cifar100", "--epochs", "100"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[:3] == ["-u", "run.py", "loo-compact"]
    flags = args[3:]
    # argparse uses the final occurrence for explicit CLI overrides.
    value = lambda flag: flags[max(i for i, v in enumerate(flags) if v == flag)+1]
    assert value("--adapter-type") == kind
    assert (value("--paca-tuner") if "--paca-tuner" in flags else "direct") == tuner
    assert value("--dataset") == "cifar100" and value("--epochs") == "100"
    assert value("--output-dir") == str(ROOT / "outputs/loo_compact_321")
    assert value("--head-warmup-epochs") == "5"
    assert value("--lora-rank") == ("32" if task in (2, 3) else "16")


def test_loo_train_and_evaluate_help_are_real_subprocesses():
    for workflow in ("loo-compact", "evaluate-loo-compact"):
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), workflow, "--help"],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "LOO" in result.stdout
