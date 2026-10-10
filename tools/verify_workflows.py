"""Run each workflow's tests in isolation to avoid identical module names."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for folder in (root, root / "SFT_ORIGINAL", root / "SFT_EXPERIEMTN", root / "SFT_BLOCK_SUBTRACTION", root / "SFT_SNIP_COMPACT"):
    print(f"Testing {folder.name}", flush=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=folder)
    if result.returncode:
        raise SystemExit(result.returncode)
