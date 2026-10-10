"""Collect the supplied epoch-log format, marking missing/unfinished runs."""
import argparse
import csv
import re
from pathlib import Path

parser = argparse.ArgumentParser(description="Compile symbolic SLURM logs; no training or test evaluation.")
parser.add_argument("--root", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--jobs", nargs="+", default=["out:27703", "out1:27716", "out2:27718"],
                    help="Filename prefix and job ID, e.g. out:27703")
args = parser.parse_args()
root = Path(args.root)
destination = Path(args.out).parent
destination.mkdir(parents=True, exist_ok=True)
jobs = []
for specification in args.jobs:
    pieces = specification.split(":")
    if len(pieces) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", pieces[0]) or not pieces[1].isdigit():
        parser.error("--jobs entries must be filename_prefix:job_id")
    jobs.append(tuple(pieces))
if len(set(jobs)) != len(jobs):
    parser.error("Duplicate --jobs entries")

def last(pattern, text):
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    return matches[-1] if matches else None

def parse(path, prefix, job, task):
    text = path.read_text(errors='replace')
    # If a log was appended on restart, use its last training invocation.
    starts = list(re.finditer(r'\[Seed\] Global seed set to', text))
    if starts:
        text = text[starts[-1].start():]
    row = dict(series=prefix, job_id=job, task_id=task, block_replaced='',
               dataset='', seed='', best_val_acc_pct='', test_acc_pct='',
               best_epoch='', status='incomplete', log_file=str(path))
    patterns = {
        'block_replaced': r'Replaced Block\(s\):\s*\[(\d+)\]',
        'dataset': r'\[SFP\] Dataset:\s*(\S+)',
        'seed': r'Global seed set to\s+(\d+)',
        'test_acc_pct': r'Final Test Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%',
    }
    for key, pattern in patterns.items():
        match = last(pattern, text)
        if match:
            row[key] = match.group(1)
    if not row['block_replaced']:
        match = last(r'pruned-block_(\d+)(?:_|\s|$)', text)
        if match:
            row['block_replaced'] = match.group(1)
    if not row['dataset']:
        match = last(r"\[Data\] Loaded '([^']+)'", text)
        if match:
            row['dataset'] = match.group(1)
    match = last(r'Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\(epoch\s+(\d+)\)', text)
    if match:
        row['best_val_acc_pct'], row['best_epoch'] = match.groups()
    else:
        # In an unfinished run this is only the best observed validation score.
        epochs = re.findall(r'\[Epoch\s+(\d+)[^\]]*\][^\n]*?Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%', text)
        if epochs:
            epoch, accuracy = max(epochs, key=lambda item: float(item[1]))
            row['best_epoch'], row['best_val_acc_pct'] = epoch, accuracy
    if all(row[key] != '' for key in ('block_replaced', 'best_val_acc_pct', 'test_acc_pct', 'seed', 'dataset')):
        row['status'] = 'complete'
    return row

rows = []
for prefix, job in jobs:
    paths = {}
    for path in root.glob(f'{prefix}_{job}_*.log'):
        match = re.fullmatch(rf'{re.escape(prefix)}_{job}_(\d+)\.log', path.name)
        if match:
            task = int(match.group(1))
            if task in paths:
                raise RuntimeError(f'Duplicate task logs: {paths[task]} and {path}')
            paths[task] = path
    for task in sorted(set(range(12)) | set(paths)):
        if task in paths:
            rows.append(parse(paths[task], prefix, job, task))
        else:
            rows.append(dict(series=prefix, job_id=job, task_id=task,
                             status='missing_log', log_file=str(root / f'{prefix}_{job}_{task}.log')))

fields = ['series', 'job_id', 'task_id', 'block_replaced', 'dataset', 'seed',
          'best_val_acc_pct', 'test_acc_pct', 'best_epoch', 'status', 'log_file']
output = Path(args.out)
temporary = output.with_suffix('.csv.tmp')
with temporary.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(output)

print(f'CSV saved: {output}')
for prefix, job in jobs:
    group = [row for row in rows if row['job_id'] == job]
    completed = [row for row in group if row['status'] == 'complete']
    blocks = sorted({int(row['block_replaced']) for row in completed})
    print(f'{prefix}_{job}: {len(completed)} complete logs; completed blocks: {blocks}')
    print(f'  Blocks without completed results: {sorted(set(range(12)) - set(blocks))}')
print('Accuracies are percentages. Missing/unfinished results have blank values and a status flag.')
