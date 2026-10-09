import argparse
import json

import pytest
import run_grid
from collect_experiments import collect


def options(tmp_path, **changes):
    values = dict(epochs=100, max_epochs=200, patience=10, block_counts=[1, 2, 4, 6, 8],
                  ranks=[8, 16], datasets=["pets"], seeds=[18], training_budget="fixed",
                  output_root=str(tmp_path), data_dir=str(tmp_path / "data"))
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("budget,count", [("fixed", 25), ("early", 25), ("both", 50)])
def test_grid_budget_counts_unique_outputs_and_arguments(tmp_path, budget, count):
    run_grid.prepare(options(tmp_path, training_budget=budget))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["tasks"]) == count
    assert len({t["task_dir"] for t in manifest["tasks"]}) == count
    for task in manifest["tasks"]:
        argv = task["args"]
        assert argv[argv.index("--epochs") + 1] == ("100" if task["budget"] == "fixed" else "-1")
        assert argv[argv.index("--mode") + 1] == "sft"


def test_grid_multidataset_and_multiple_seeds(tmp_path):
    run_grid.prepare(options(tmp_path, datasets=["pets", "cifar100"], seeds=[18, 42]))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["tasks"]) == 100
    assert {t["dataset"] for t in manifest["tasks"]} == {"pets", "cifar100"}
    with pytest.raises(FileExistsError):
        run_grid.prepare(options(tmp_path))


@pytest.mark.parametrize("change", [dict(block_counts=[12]), dict(ranks=[0]), dict(seeds=[18, 18])])
def test_grid_rejects_invalid_or_duplicate_runs(tmp_path, change):
    with pytest.raises(ValueError):
        run_grid.prepare(options(tmp_path, **change))


def test_collector_preserves_config_blocks_coefficients_and_metrics(tmp_path):
    for index, kind in enumerate(["lowrank", "symbolic"]):
        folder = tmp_path / str(index); folder.mkdir()
        summary = dict(filter_type=kind, filter_rank=16, pruned_block_idx=[1, 3], final_test_acc=90 + index,
                       compression=dict(filter_params=25344, total_reduction_vs_original_pct=16.4),
                       cli_args=["--filter-type", kind])
        (folder / "metrics_summary.json").write_text(json.dumps(summary))
    rows = collect(tmp_path)
    assert len(rows) == 2
    assert [r["filter_type"] for r in rows] == ["lowrank", "symbolic"]
    assert json.loads(rows[0]["pruned_block_idx"]) == [1, 3]
    assert rows[0]["compression.filter_params"] == 25344
