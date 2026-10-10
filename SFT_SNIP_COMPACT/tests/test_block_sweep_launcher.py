import json
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parents[1]


def invoke(tmp_path, *arguments):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                    "open(os.environ['SBATCH_RECORD'],'w').write(json.dumps(sys.argv[1:]))\n"
                    "print('Fake submission: no cluster contacted')\n")
    fake.chmod(0o755)
    record = tmp_path / "submission.json"
    env = dict(os.environ, CODE_DIR=str(HERE), MAX_CONCURRENT="4", SBATCH_RECORD=str(record),
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    root = tmp_path / "sweep"
    result = subprocess.run(["bash", str(HERE / "submit_block_sweep.sh"), "--output-root", str(root),
                             *arguments], env=env, text=True, capture_output=True)
    return result, root, record


@pytest.mark.parametrize("budget,total", [("fixed", 60), ("early", 60), ("both", 120)])
def test_all_positions_and_training_budgets(tmp_path, budget, total):
    result, root, record = invoke(tmp_path, "--training-budget", budget)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == total
    assert {t["block_index"] for t in tasks} == set(range(12))
    assert {t["dataset"] for t in tasks} == {"pets"}
    assert {t["seed"] for t in tasks} == {18}
    assert len({t["task_dir"] for t in tasks}) == total
    assert { (t["filter_type"], t["rank"]) for t in tasks } == {
        ("dense", None), ("lowrank", 8), ("lowrank", 16), ("symbolic", 8), ("symbolic", 16)}
    for task in tasks:
        argv = task["args"]
        assert task["blocks"] == 1
        assert argv[argv.index("--pruned-block") + 1] == str(task["block_index"])
        assert "--num-filter-blocks" not in argv
        assert argv[argv.index("--epochs") + 1] == ("-1" if task["budget"] == "early" else "100")
    command = json.loads(record.read_text())
    assert f"--array=0-{total-1}%4" in command
    assert command[-3:] == [str(HERE / "experiment.sbatch"), str(HERE), str(root / "manifest.json")]
    assert (root / "logs").is_dir()


def test_all_eligible_multiblock_counts(tmp_path):
    result, root, record = invoke(tmp_path, "--block-counts", *(str(b) for b in range(1, 12)))
    assert result.returncode == 0, result.stderr
    tasks = json.loads((root / "manifest.json").read_text())["tasks"]
    assert len(tasks) == 55
    assert {t["blocks"] for t in tasks} == set(range(1, 12))
    for task in tasks:
        assert task["block_index"] is None
        argv = task["args"]
        assert "--pruned-block" not in argv
        assert argv[argv.index("--num-filter-blocks") + 1] == str(task["blocks"])
    assert "--array=0-54%4" in json.loads(record.read_text())


def test_subset_positions_and_dataset_seed_multipliers(tmp_path):
    result, root, _ = invoke(tmp_path, "--block-indices", "0", "5", "11", "--ranks", "16",
                             "--datasets", "pets", "svhn", "--seeds", "18", "42")
    assert result.returncode == 0, result.stderr
    tasks = json.loads((root / "manifest.json").read_text())["tasks"]
    assert len(tasks) == 36
    assert {t["block_index"] for t in tasks} == {0, 5, 11}
    assert {t["dataset"] for t in tasks} == {"pets", "svhn"}
    assert {t["seed"] for t in tasks} == {18, 42}


@pytest.mark.parametrize("invalid", [
    ["--block-counts", "12"], ["--block-indices", "12"], ["--block-indices", "1", "1"],
    ["--epochs", "0"], ["--seeds", "-1"],
])
def test_invalid_input_is_not_submitted(tmp_path, invalid):
    result, root, record = invoke(tmp_path, *invalid)
    assert result.returncode == 2
    assert not record.exists()
    assert not (root / "manifest.json").exists()
