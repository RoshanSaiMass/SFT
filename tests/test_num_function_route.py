import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('num_function_routes', ROOT / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_exact_count_route_and_default_output_isolation(tmp_path):
    for name in ['num_function', 'num-function']:
        command, cwd = run.build_command(name, ['--num-functions', '9', '--filter-rank', '128', '--lora-rank', '8'], tmp_path)
        assert Path(command[2]) == ROOT / 'SFT_NUM_FUNCTION/train_num_function.py'
        assert cwd == ROOT / 'SFT_NUM_FUNCTION'
        assert command[command.index('--output-dir') + 1] == str(ROOT / 'outputs/num_function')
        assert command[command.index('--num-functions') + 1] == '9'
    for name in ['loo-compact', 'snip-compact', 'original']:
        command, _ = run.build_command(name, [], tmp_path)
        assert '/SFT_NUM_FUNCTION/' not in command[2]


def test_exact_count_help_and_saved_evaluator_route():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'num_function', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--num-functions' in result.stdout and 'exactly' in result.stdout
    command, _ = run.build_command('evaluate-num-function', ['--summary', '/tmp/result.json'])
    assert Path(command[2]) == ROOT / 'tools/evaluate_num_function.py'
