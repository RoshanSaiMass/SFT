import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_dedicated_snip_experiment_is_not_packaged_or_routed():
    assert not (ROOT / "SFT_SNIP_COMPACT").exists()
    assert not (ROOT / "tools/evaluate_snip_compact.py").exists()
    assert not (ROOT / "slurm/run_snip_dense_baseline_a100.sh").exists()
    assert not (ROOT / "downloads/SFT-snip-compact.zip").exists()
    for name in ("snip-compact", "evaluate-snip-compact"):
        with pytest.raises(ValueError, match="Unknown workflow"):
            run.build_command(name, [])
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "snip-compact" not in result.stdout
    for name in ("original", "symbolic", "lowrank", "correlation", "drop-one", "loo-compact"):
        assert name in result.stdout


def capture_launcher(script, tmp_path, task=None, flags=()):
    executable = tmp_path / "python3"
    executable.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                          "open(os.environ['CAPTURE_ARGS'],'w').write(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o755)
    capture = tmp_path / "args.json"
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ["PATH"],
               CAPTURE_ARGS=str(capture), SLURM_ARRAY_JOB_ID="456", SLURM_JOB_ID="456",
               SLURM_CPUS_PER_TASK="4", SLURM_SUBMIT_DIR=str(ROOT))
    # Default location must follow the submission directory, even after extraction.
    env.pop("CODE_DIR", None)
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(task)
    result = subprocess.run(["bash", "-c",
        'source() { :; }; conda() { :; }; mkdir() { :; }; export -f source conda mkdir; exec bash "$@"',
        "mock", str(script), *flags], env=env, capture_output=True, text=True)
    return result, json.loads(capture.read_text()) if capture.exists() else None


def final_value(flags, name, default=None):
    indexes = [i for i, value in enumerate(flags) if value == name]
    return flags[indexes[-1]+1] if indexes else default


def test_all_104_loo_dataset_peft_combinations_and_output_isolation(tmp_path):
    datasets = ["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100",
                "fgvc_aircraft", "eurosat", "sun397", "pcam", "clevr",
                "dsprites-loc", "dsprites-ori"]
    methods = [("lora", "direct"), ("dora", "direct"), ("paca", "direct"), ("rpaca", "direct"),
               ("paca", "lora"), ("paca", "dora"), ("rpaca", "lora"), ("rpaca", "dora")]
    observed, outputs = set(), set()
    for task in range(104):
        result, argv = capture_launcher(ROOT / "slurm/run_loo_all_datasets_a100.sh", tmp_path,
                                         task=task, flags=("--epochs", "100"))
        assert result.returncode == 0, result.stderr
        assert argv[:3] == ["-u", "run.py", "loo-compact"]
        flags = argv[3:]
        dataset = final_value(flags, "--dataset")
        method = (final_value(flags, "--adapter-type"), final_value(flags, "--paca-tuner", "direct"))
        assert dataset == datasets[task//8] and method == methods[task % 8]
        assert final_value(flags, "--epochs") == "100"
        assert final_value(flags, "--loo-samples") == "512"
        assert final_value(flags, "--head-warmup-epochs") == "5"
        assert final_value(flags, "--data-dir") == str(ROOT / "data")
        observed.add((dataset, *method)); outputs.add(final_value(flags, "--output-dir"))
    assert len(observed) == len(outputs) == 104
    for task in (-1, 104):
        result, _ = capture_launcher(ROOT / "slurm/run_loo_all_datasets_a100.sh", tmp_path, task=task)
        assert result.returncode != 0 and "Invalid array task" in result.stderr


def test_correlation_launcher_uses_highest_pair_in_final_repo(tmp_path):
    result, argv = capture_launcher(ROOT / "slurm/run_correlation_a100.sh", tmp_path,
                                    flags=("--dataset", "cifar100", "--epochs", "100"))
    assert result.returncode == 0, result.stderr
    assert argv[:3] == ["-u", "run.py", "correlation"]
    flags = argv[3:]
    assert final_value(flags, "--min-correlation") == "-1"
    assert final_value(flags, "--pair-index") == "-1"
    assert final_value(flags, "--adapter-scope") == "retained"
    assert final_value(flags, "--distill-epochs") == "5"
    assert final_value(flags, "--dataset") == "cifar100"
    assert final_value(flags, "--epochs") == "100"
    assert final_value(flags, "--output-dir") == str(ROOT / "outputs/correlation_456")


def test_original_submitter_defaults_to_materialized_backend_and_correct_runner(tmp_path):
    env = dict(os.environ, OUTPUT_ROOT=str(tmp_path / "grid outputs"),
               PREPARE_PYTHON=sys.executable, DATA_DIR=str(tmp_path / "shared data"),
               CACHE_DIR=str(tmp_path / "cache"))
    env.pop("CODE_DIR", None)
    env.pop("SEEDS", None)
    result = subprocess.run(["bash", str(ROOT / "slurm/submit_all.sh"),
                             "--training-budget", "both", "--dry-run"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifests = list((tmp_path / "grid outputs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["code_dir"] == str(ROOT / "SFT_ORIGINAL")
    assert manifest["runner_path"] == str(ROOT / "slurm/sweep.py")
    assert len(manifest["tasks"]) == 520
    assert {row["budget"] for row in manifest["tasks"]} == {"fixed", "early"}
    assert str(ROOT / "slurm/all_methods.sbatch") in result.stdout
    assert "--array=0-519%" in result.stdout


def test_original_batch_uses_recorded_runner_not_trainer_child_folder(tmp_path):
    # Emulate a Slurm-spooled script, where BASH_SOURCE is not the source folder.
    runner = tmp_path / "runner with spaces.py"
    capture = tmp_path / "batch.json"
    runner.write_text("import json,os,sys\nopen(os.environ['CAPTURE_BATCH'],'w').write(json.dumps(sys.argv[1:]))\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(code_dir=str(ROOT / "SFT_ORIGINAL"), runner_path=str(runner))))
    batch = tmp_path / "spooled_batch.sh"; batch.write_text((ROOT / "slurm/all_methods.sbatch").read_text())
    env = dict(os.environ, SLURM_ARRAY_TASK_ID="3", CAPTURE_BATCH=str(capture))
    result = subprocess.run(["bash", "-c", 'conda() { :; }; source() { :; }; export -f conda source; exec bash "$@"',
                             "mock", str(batch), str(manifest)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text()) == ["task", "--manifest", str(manifest), "--task-id", "3"]


def test_original_batch_accepts_previous_manifest_without_runner_path(tmp_path):
    # Preserve old standalone manifests through the original fallback path.
    code = tmp_path / "old code"; (code / "slurm").mkdir(parents=True)
    capture = tmp_path / "old_batch.json"
    (code / "slurm/sweep.py").write_text("import json,os,sys\nopen(os.environ['CAPTURE_BATCH'],'w').write(json.dumps(sys.argv[1:]))\n")
    manifest = tmp_path / "old_manifest.json"; manifest.write_text(json.dumps(dict(code_dir=str(code))))
    env = dict(os.environ, SLURM_ARRAY_TASK_ID="1", CAPTURE_BATCH=str(capture))
    result = subprocess.run(["bash", "-c", 'conda() { :; }; source() { :; }; export -f conda source; exec bash "$@"',
                             "mock", str(ROOT / "slurm/all_methods.sbatch"), str(manifest)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text())[-2:] == ["--task-id", "1"]
