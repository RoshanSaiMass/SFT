import json
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "run_block_sweep_b200.sh"


def launch(tmp_path, *args, concurrent="2"):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    record = tmp_path / "submission.json"
    stub = bin_dir / "sbatch"
    stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                    "open(os.environ['SBATCH_RECORD'],'w').write(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    env = dict(os.environ, CODE_DIR=str(HERE), MAX_CONCURRENT=concurrent,
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], SBATCH_RECORD=str(record))
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        env.pop(name, None)
    root = tmp_path / "results"
    result = subprocess.run(["bash", str(SCRIPT), "--output-root", str(root), *args],
                            text=True, capture_output=True, env=env)
    return result, root, record


def test_supplied_b200_resources_path_and_no_a100_node_constraint():
    text = SCRIPT.read_text()
    for line in ("#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4",
                 "#SBATCH --time=11:59:59", "#SBATCH --mem=64G", "#SBATCH --partition=mig90g",
                 "#SBATCH --gres=gpu:3g.90gb:1"):
        assert line in text
    assert "/home/achyutm01/SFT/SFT_EXPERIEMTN" in text
    assert "#SBATCH --nodelist" not in text
    assert "gpu-a100" not in text


@pytest.mark.parametrize("budget,count", [("fixed", 60), ("early", 60), ("both", 120)])
def test_b200_submits_self_with_two_concurrent_jobs(tmp_path, budget, count):
    result, root, record = launch(tmp_path, "--training-budget", budget)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["tasks"]) == count
    assert {t["block_index"] for t in manifest["tasks"]} == set(range(12))
    command = json.loads(record.read_text())
    assert f"--array=0-{count-1}%2" in command
    assert command[-3:] == [str(SCRIPT), str(root / "manifest.json"), str(HERE)]
    assert (root / "logs").is_dir()
    for task in manifest["tasks"]:
        argv = task["args"]
        assert argv[argv.index("--data-dir") + 1] == str(HERE.parent / "data")
        assert argv[argv.index("--epochs") + 1] == ("100" if task["budget"] == "fixed" else "-1")


def test_b200_multiblock_sweep(tmp_path):
    result, root, record = launch(tmp_path, "--block-counts", *(str(i) for i in range(1, 12)))
    assert result.returncode == 0, result.stderr
    assert len(json.loads((root / "manifest.json").read_text())["tasks"]) == 55
    assert "--array=0-54%2" in json.loads(record.read_text())


def test_b200_rejects_more_than_two_simultaneous_jobs(tmp_path):
    result, root, record = launch(tmp_path, concurrent="4")
    assert result.returncode == 2
    assert not record.exists()
    assert not (root / "manifest.json").exists()
