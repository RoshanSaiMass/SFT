"""One CLI for the original SFT/PEFT, compact filters and block subtraction."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ROUTES = {
    "snip-compact": ("SFT_SNIP_COMPACT/train_snip_compact.py", []),
    "evaluate-snip-compact": ("tools/evaluate_snip_compact.py", []),
    "original": ("SFT_ORIGINAL/train_sfp_lora.py", []),
    "dense": ("SFT_ORIGINAL/train_sfp_lora.py", []),
    "lowrank": ("SFT_EXPERIEMTN/train_sfp_lora.py", ["--filter-type", "lowrank"]),
    "symbolic": ("SFT_EXPERIEMTN/train_sfp_lora.py", ["--filter-type", "symbolic"]),
    "correlation": ("SFT_BLOCK_SUBTRACTION/train_subtraction.py", ["--method", "correlation"]),
    "drop-one": ("SFT_BLOCK_SUBTRACTION/train_subtraction.py", ["--method", "drop-one"]),
    "removal-sweep": ("SFT_BLOCK_SUBTRACTION/run_removal_sweep.py", []),
    "original-grid": ("slurm/sweep.py", []),
    "symbolic-grid": ("SFT_EXPERIEMTN/run_grid.py", []),
    "collect-original": ("SFT_ORIGINAL/analyse_results.py", []),
    "collect-symbolic": ("SFT_EXPERIEMTN/collect_experiments.py", []),
    "collect-removals": ("SFT_BLOCK_SUBTRACTION/select_removals.py", []),
    "collect-all": ("tools/collect_all.py", []),
    "collect-logs": ("tools/collect_logs.py", []),
    "evaluate-removal": ("tools/evaluate_removal.py", []),
    "download-data": ("SFT_EXPERIEMTN/download_data.py", []),
}
TRAINING = {"snip-compact", "original", "dense", "lowrank", "symbolic", "correlation", "drop-one", "removal-sweep"}
PATH_FLAGS = {"--data-dir", "--output-dir", "--output-root", "--root", "--out", "--out-dir",
              "--outdir", "--manifest", "--code-dir", "--cache-dir", "--data-root", "--summary"}


def option_values(flags, key):
    values = []
    for i, flag in enumerate(flags):
        if flag == key and i + 1 < len(flags):
            values.append(flags[i + 1])
        elif flag.startswith(key + "="):
            values.append(flag.split("=", 1)[1])
    return values


def normalize_paths(flags, caller):
    """Relative paths refer to the calling directory, not the backend directory."""
    flags = list(flags)
    for i, flag in enumerate(flags):
        key, sep, value = flag.partition("=")
        if key not in PATH_FLAGS:
            continue
        if not sep:
            if i + 1 >= len(flags) or flags[i + 1].startswith("--"):
                raise ValueError(f"{key} needs a path.")
            value = flags[i + 1]
        path = Path(value).expanduser()
        absolute = str(path.resolve() if path.is_absolute() else (caller / path).resolve())
        if sep:
            flags[i] = key + "=" + absolute
        else:
            flags[i + 1] = absolute
    return flags


def build_command(workflow, flags, caller=None):
    if workflow not in ROUTES:
        raise ValueError(f"Unknown workflow: {workflow}")
    caller = Path(caller or Path.cwd()).resolve()
    flags = list(flags)
    if flags[:1] == ["--"]:
        flags.pop(0)
    script, forced = ROUTES[workflow]
    for key, value in zip(forced[::2], forced[1::2]):
        if any(actual != value for actual in option_values(flags, key)):
            raise ValueError(f"Workflow {workflow} requires {key} {value}; choose the matching workflow.")
    defaults = list(forced)
    def default(key, value):
        if not option_values(flags, key):
            defaults.extend([key, str(value)])
    if workflow in TRAINING:
        default("--data-dir", ROOT / "data")
        family = "original" if workflow == "dense" else workflow
        default("--output-dir", ROOT / "outputs" / family)
        if workflow in ("lowrank", "symbolic"):
            peft_flags = ("--adapter-type", "--lora-rank", "--paca-tuner", "--paca-rank",
                          "--paca-adapter-rank", "--unilora-dim", "--init-method",
                          "--lora-ortho-lambda1", "--lora-ortho-lambda2")
            # Do not force no-adapter SFT when the caller explicitly enables PEFT.
            if not any(option_values(flags, key) for key in peft_flags) and "--full-finetune" not in flags:
                default("--mode", "sft")
        if workflow == "correlation":
            # User requested highest valid correlation rather than a 0.9 cutoff.
            default("--min-correlation", -1)
    elif workflow == "collect-original":
        default("--root", ROOT / "outputs" / "original")
        default("--out-dir", ROOT / "compiled_results" / "original")
    elif workflow == "collect-symbolic":
        default("--root", ROOT / "outputs")
        default("--out", ROOT / "compiled_results" / "experiments_all.csv")
    elif workflow == "collect-all":
        default("--root", ROOT / "outputs")
        default("--out", ROOT / "compiled_results" / "all_results.csv")
    elif workflow == "collect-logs":
        default("--root", ROOT / "SFT_EXPERIEMTN")
        default("--out", ROOT / "compiled_results" / "symbolic_block_results.csv")
    elif workflow == "download-data":
        default("--data-root", ROOT / "data")
    elif workflow == "symbolic-grid" and flags[:1] == ["prepare"]:
        default("--data-dir", ROOT / "data")
        default("--output-root", ROOT / "outputs" / "compact_grid" / f"sweep_{time.time_ns()}")
    elif workflow == "original-grid" and flags[:1] == ["prepare"]:
        default("--code-dir", ROOT / "SFT_ORIGINAL")
        default("--data-dir", ROOT / "data")
        default("--cache-dir", ROOT / "data" / "backbone_cache")
        default("--output-root", ROOT / "outputs" / "original_grid")
    # Backend subcommands must precede injected options.
    if workflow in ("original-grid", "symbolic-grid") and flags and not flags[0].startswith("-"):
        forwarded = [flags[0]] + defaults + flags[1:]
    else:
        forwarded = defaults + flags
    target = ROOT / script
    if not target.is_file():
        raise FileNotFoundError(target)
    return [sys.executable, "-u", str(target)] + normalize_paths(forwarded, caller), target.parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use: python run.py WORKFLOW --help for all backend flags. "
        "Use: python run.py --dry-run WORKFLOW FLAGS to preview a command."))
    parser.add_argument("workflow", choices=ROUTES, nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    if not argv or argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    dry_run = argv[:1] == ["--dry-run"]
    if dry_run:
        argv.pop(0)
    if not argv:
        parser.error("Choose a workflow after --dry-run.")
    workflow, flags = argv[0], argv[1:]
    try:
        command, cwd = build_command(workflow, flags)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if dry_run:
        print(f"Working directory: {cwd}")
        print(shlex.join(command))
        return 0
    return subprocess.call(command, cwd=cwd)


if __name__ == "__main__":
    raise SystemExit(main())
