import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from tools.collect_all import collect


@pytest.mark.parametrize("workflow", list(run.ROUTES))
def test_every_route_exists_and_paths_are_resolved(workflow, tmp_path):
    command, cwd = run.build_command(workflow, ["--help"], caller=tmp_path)
    assert Path(command[2]).is_file()
    assert cwd == Path(command[2]).parent
    assert "--help" in command


@pytest.mark.parametrize("workflow,flag,value", [
    ("original", "--data-dir", "./shared data"),
    ("symbolic", "--output-dir", "./my results"),
    ("collect-removals", "--root", "./sweep"),
    ("evaluate-removal", "--summary", "./run/metrics_summary.json"),
])
def test_relative_paths_resolve_from_caller(workflow, flag, value, tmp_path):
    command, _ = run.build_command(workflow, [flag, value], caller=tmp_path)
    index = command.index(flag)
    assert command[index + 1] == str((tmp_path / value).resolve())


def test_equal_path_flags_and_unknown_backend_flags_are_preserved(tmp_path):
    command, _ = run.build_command("symbolic", ["--output-dir=./results", "--symbolic-max-terms", "3"], tmp_path)
    assert "--output-dir=" + str(tmp_path / "results") in command
    assert command[-2:] == ["--symbolic-max-terms", "3"]


@pytest.mark.parametrize("workflow,key,value", [
    ("symbolic", "--filter-type", "dense"),
    ("lowrank", "--filter-type", "symbolic"),
    ("correlation", "--method", "drop-one"),
    ("drop-one", "--method", "correlation"),
])
def test_conflicting_workflow_flags_rejected(workflow, key, value):
    with pytest.raises(ValueError, match="requires"):
        run.build_command(workflow, [key, value])


def test_peft_flags_are_forwarded_without_modification():
    flags = ["--adapter-type", "paca", "--paca-tuner", "dora", "--paca-rank", "32",
             "--paca-adapter-rank", "16", "--lora-ortho-lambda1", "0.9", "--epochs", "-1"]
    command, _ = run.build_command("original", flags)
    assert command[-len(flags):] == flags


def test_compact_defaults_do_not_disable_requested_peft():
    command, _ = run.build_command("symbolic", [])
    assert command[command.index("--mode") + 1] == "sft"
    command, _ = run.build_command("symbolic", ["--adapter-type", "dora", "--lora-rank", "16"])
    assert "--mode" not in command


def test_actual_compact_peft_validation_is_reachable_without_downloads():
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "symbolic",
                             "--adapter-type", "dora", "--lora-rank", "16", "--filter-rank", "0"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "filter-rank must be" in result.stderr
    assert "mode sft trains" not in result.stderr


def test_actual_grid_preparation_isolated_and_preserves_budget_options(tmp_path):
    folder = tmp_path / "grid"
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "symbolic-grid", "prepare",
                             "--datasets", "pets", "--block-counts", "1", "--ranks", "16",
                             "--training-budget", "both", "--output-root", str(folder)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(manifest["tasks"]) == 6
    assert {task["budget"] for task in manifest["tasks"]} == {"fixed", "early"}
    assert {task["filter_type"] for task in manifest["tasks"]} == {"dense", "lowrank", "symbolic"}


def test_collect_supplied_log_format_through_common_cli(tmp_path):
    log = """[Seed] Global seed set to 18 (deterministic=True).
[SFP] Dataset: PETS
[SFP] Mode: SFT-only | Replaced Block(s): [9] (Single Filter Block)
[SFP] Best Val Acc: 90.50% (epoch 19)
[SFP] Final Test Acc: 84.49% | Final Test Loss: 1.2344
"""
    (tmp_path / "out_27703_0.log").write_text(log)
    output = tmp_path / "combined.csv"
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "collect-logs",
                             "--root", str(tmp_path), "--out", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    row = rows[0]
    assert row["block_replaced"] == "9" and row["task_id"] == "0"
    assert row["best_val_acc_pct"] == "90.50" and row["test_acc_pct"] == "84.49"
    assert sum(r["status"] == "missing_log" for r in rows) == 35


def test_highest_correlation_default_and_explicit_threshold():
    command, _ = run.build_command("correlation", [])
    assert command[command.index("--min-correlation") + 1] == "-1"
    command, _ = run.build_command("correlation", ["--min-correlation", "0.9"])
    assert command.count("--min-correlation") == 1
    assert command[command.index("--min-correlation") + 1] == "0.9"


@pytest.mark.parametrize("workflow", ["original-grid", "symbolic-grid"])
def test_grid_subcommand_stays_before_options(workflow):
    command, _ = run.build_command(workflow, ["prepare", "--epochs", "100"])
    assert command[3] == "prepare"


def test_real_subprocess_keeps_backend_isolated_and_propagates_failure(monkeypatch, tmp_path, capfd):
    script = tmp_path / "SFT_ORIGINAL/train_sfp_lora.py"
    script.parent.mkdir()
    script.write_text("import json,os,sys\nprint(json.dumps({'cwd':os.getcwd(),'args':sys.argv[1:]}))\nsys.exit(7)\n")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    status = run.main(["original", "--adapter-type", "dora"])
    assert status == 7
    result = json.loads(capfd.readouterr().out)
    assert result["cwd"] == str(script.parent)
    assert result["args"][-2:] == ["--adapter-type", "dora"]


def test_collect_all_preserves_methods_blocks_and_deferred_test(tmp_path):
    for name, summary in [
        ("original", dict(dataset="pets", seed=18, pruned_block_idx=[2], best_val_acc=90, final_test_acc=89)),
        ("symbolic", dict(dataset="pets", seed=18, filter_type="symbolic", pruned_block_idx=[3],
                          best_val_acc=89, final_test_acc=88, compression={"filter_reduction_pct":95.7})),
        ("removed", dict(dataset="pets", seed=18, method="drop-one", removed_block_idx=4,
                         best_val_acc=91, final_test_acc=None, config={"lora_rank":16})),
    ]:
        folder=tmp_path/name; folder.mkdir()
        (folder/"metrics_summary.json").write_text(json.dumps(summary))
    output = tmp_path/"compiled/all.csv"
    rows = collect(tmp_path, output)
    assert len(rows) == 3
    with output.open() as stream:
        saved = list(csv.DictReader(stream))
    assert {row["workflow"] for row in saved} == {"original", "symbolic", "drop-one"}
    removed=next(row for row in saved if row["workflow"]=="drop-one")
    assert removed["final_test_acc"] == "" and removed["test_available"] == "False"
    assert removed["block_indices"] == "[4]" and removed["config.lora_rank"] == "16"


def test_empty_collection_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No saved"):
        collect(tmp_path,tmp_path/"all.csv")


def test_long_original_cli_names_fit_real_filesystem_and_remain_distinct(tmp_path):
    spec = importlib.util.spec_from_file_location("integrated_original_names", ROOT / "SFT_ORIGINAL/run_naming.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flags = ["--dataset", "pets", "--adapter-type", "paca", "--paca-tuner", "dora",
             "--paca-rank", "32", "--paca-adapter-rank", "16", "--seed", "18",
             "--epochs", "100", "--max-epochs", "200", "--patience", "10",
             "--save-misclassified-images", "false", "--lora-ortho-lambda1", "0.9",
             "--lora-ortho-lambda2", "0.9", "--data-dir", str(tmp_path / "cache")]
    first = module.build_run_folder_name(flags)
    second = module.build_run_folder_name(flags + ["--lora-lr", "0.0002"])
    assert len(first.encode()) <= 240 and len(second.encode()) <= 240
    assert first != second
    (tmp_path / first).mkdir()
    (tmp_path / second).mkdir()


@pytest.mark.parametrize("workflow,expected", [
    ("snip-compact", "--snip-momentum"),
    ("evaluate-snip-compact", "--summary"),
    ("original", "--data-dir"),
    ("symbolic", "--symbolic-max-terms"),
    ("correlation", "--min-correlation"),
    ("removal-sweep", "--adapter-types"),
    ("evaluate-removal", "--summary"),
])
def test_backend_help_is_reachable(workflow, expected):
    result = subprocess.run([sys.executable, str(ROOT/"run.py"), workflow, "--help"],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_new_workflow_keeps_top_peft_flags_and_csv_indices(tmp_path):
    flags = ["--adapter-type", "paca", "--paca-tuner", "dora", "--paca-rank", "32",
             "--paca-adapter-rank", "16", "--filter-type", "symbolic"]
    command, _ = run.build_command("snip-compact", flags, tmp_path)
    assert command[-len(flags):] == flags
    assert "--mode" not in command
    assert command[command.index("--output-dir")+1] == str(ROOT/"outputs/snip-compact")
    folder = tmp_path/"run"; folder.mkdir()
    (folder/"metrics_summary.json").write_text(json.dumps(dict(method="snip-compact",
        dataset="pets", seed=18, filter_type="symbolic", pruned_block_idx=[4],
        replaced_block_idx=4, peft_block_idx=8, best_val_acc=90, final_test_acc=89)))
    rows = collect(tmp_path, tmp_path/"all.csv")
    assert rows[0]["workflow"] == "snip-compact"
    assert rows[0]["block_indices"] == "[4]"
    assert rows[0]["peft_block_idx"] == 8 and rows[0]["final_test_acc"] == 89
