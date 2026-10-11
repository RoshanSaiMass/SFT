"""Compile saved research runs, select sweep blocks by validation, and draw figures."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from research_suite import FOLDERS, GROUPS, destination


def flatten(value, prefix=''):
    row = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            row.update(flatten(item, name))
        else:
            row[name] = json.dumps(item) if isinstance(item, list) else item
    return row


def write_csv(path, rows, first=()):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(first) + sorted({k for row in rows for k in row} - set(first))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def read_runs(root, campaign=None):
    rows, problems, coverage = [], [], []
    for group in GROUPS:
        folder = root / FOLDERS[group]
        manifests = sorted(folder.glob(f'{campaign or "*"}/{group}/manifest.json'))
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as error:
                problems.append(dict(path=str(manifest_path), error=str(error)))
                continue
            completed = failed = missing = 0
            # Compute relative paths from the manifest, so copied result trees work.
            for task in manifest['tasks']:
                old_folder = destination(task, manifest['settings'])
                task_folder = manifest_path.parent / old_folder.name
                paths = sorted((task_folder / 'results').rglob('metrics_summary.json'))
                status_path = task_folder / 'status.json'
                try:
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                except (OSError, ValueError) as error:
                    status = dict(status='failed')
                    problems.append(dict(path=str(status_path), error=str(error)))
                if len(paths) != 1 or status.get('status') != 'complete':
                    failed += status.get('status') == 'failed'
                    missing += status.get('status') != 'failed'
                    problems.append(dict(group=group, campaign=manifest['settings']['campaign'],
                                         task_id=task['task_id'], path=str(task_folder),
                                         status=status.get('status', 'missing'), summary_count=len(paths)))
                    continue
                try:
                    summary = json.loads(paths[0].read_text())
                    if summary['dataset'] != task['dataset'] or summary['seed'] != manifest['settings']['seed']:
                        raise ValueError('Summary dataset/seed does not match its planned task')
                    block = task['block']
                    if block is not None and summary.get('pruned_block_idx') != [block]:
                        raise ValueError('Summary block does not match its planned task')
                    config = summary.get('config', summary)
                    params = summary.get('param_breakdown', {})
                    compression = summary.get('compression', {})
                    original = summary.get('original_model_params', compression.get('original_model_params'))
                    efficiency = summary.get('efficiency', {})
                    row = flatten(summary)
                    row.update(group=group, campaign=manifest['settings']['campaign'],
                               dataset=task['dataset'], seed=summary['seed'], method_config=task['method'],
                               block=block if block is not None else summary.get('replaced_block_idx', summary.get('removed_block_idx')),
                               peft_block=summary.get('peft_block_idx'), retained_block=summary.get('retained_block_idx'),
                               best_val_acc=number(summary.get('best_val_acc')),
                               final_test_acc=number(summary.get('final_test_acc')),
                               trainable_params=params.get('trainable_params'), total_params=params.get('total_params'),
                               trainable_pct_final_model=params.get('trainable_pct'),
                               trainable_pct_original=(100 * params['trainable_params'] / original if original else None),
                               original_model_params=original,
                               train_wall_seconds=efficiency.get('train_wall_seconds', efficiency.get('train_wall_s')),
                               train_compute_seconds=efficiency.get('train_compute_seconds', efficiency.get('train_compute_s')),
                               selection_seconds=efficiency.get('snip_search_s', efficiency.get('loo_selection_s', efficiency.get('calibration_s'))),
                               peak_gpu_mem_allocated_mb=efficiency.get('peak_gpu_mem_allocated_mb'),
                               throughput_samples_per_sec=efficiency.get('throughput_samples_per_sec'),
                               total_reduction_pct=summary.get('total_reduction_pct', compression.get('total_reduction_vs_original_pct')),
                               lambda_a=config.get('lora_ortho_lambda1', 0) or 0,
                               lambda_b=config.get('lora_ortho_lambda2', 0) or 0,
                               epochs_requested=manifest['settings']['epochs'],
                               test_available=summary.get('final_test_acc') is not None,
                               summary_path=str(paths[0].resolve()), task_id=task['task_id'],
                               checkpoint_path=summary.get('checkpoint_path'),
                               history_csv=summary.get('history_csv'))
                    # Explicit manual replacement does not execute SNIP selection.
                    if group in ('all_block_snip', 'all_block_snip+peft', 'looall'):
                        row['position_selection'] = 'manual_block_sweep'
                    elif group == 'snip_only':
                        row['position_selection'] = 'snip_candidate_search'
                    else:
                        row['position_selection'] = group
                    rows.append(row)
                    completed += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    failed += 1
                    problems.append(dict(path=str(paths[0]), error=str(error)))
            coverage.append(dict(group=group, campaign=manifest['settings']['campaign'],
                                 planned=len(manifest['tasks']), complete=completed, failed=failed,
                                 missing_or_running=missing))
        if not manifests:
            coverage.append(dict(group=group, campaign=campaign, planned=None, complete=0,
                                 failed=0, missing_or_running=None, note='No manifest found'))
    return rows, problems, coverage


def select(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['campaign'], row['group'], row['dataset'], row['method_config'], row['epochs_requested']].append(row)
    selected, incomplete = [], []
    for key, candidates in sorted(grouped.items()):
        sweep = key[1] in ('all_block_snip', 'all_block_snip+peft', 'looall', 'sub_removal')
        seeds = sorted({r['seed'] for r in candidates})
        if sweep:
            lookup = {(r['block'], r['seed']): r for r in candidates}
            if len(lookup) != len(candidates):
                raise ValueError(f'Duplicate block/seed in {key}; no selection permitted')
            missing = [(b, s) for b in range(12) for s in seeds if (b, s) not in lookup]
            if missing:
                incomplete.append(dict(campaign=key[0], group=key[1], dataset=key[2], method_config=key[3],
                                       missing=json.dumps(missing)))
                continue
            if any(r['best_val_acc'] is None for r in candidates):
                raise ValueError(f'Missing finite validation accuracy in {key}')
            means = {b: statistics.mean(lookup[b, s]['best_val_acc'] for s in seeds) for b in range(12)}
            winner = max(range(12), key=lambda b: (means[b], -b))
            winners = [lookup[winner, s] for s in seeds]
        else:
            if len(candidates) != len(seeds):
                raise ValueError(f'Duplicate selected run/seed in {key}')
            winners = candidates
        for row in winners:
            selected.append(dict(row, selection_rule='maximum_mean_validation_accuracy' if sweep else 'training_only_position_selection',
                                 block_coverage_complete=True))
    return selected, incomplete


def evaluate_removal(row, code_dir, device):
    if row['group'] != 'sub_removal' or row['final_test_acc'] is not None:
        return
    script = """import sys
from train_subtraction import evaluate_summary
evaluate_summary(sys.argv[1],sys.argv[2])
"""
    subprocess.run([sys.executable, '-c', script, row['summary_path'], device], cwd=code_dir, check=True)


def auxiliary(rows, output):
    histories, loo_scores, snip_scores = [], [], []
    for row in rows:
        run_dir = Path(row['summary_path']).parent
        tags = {k: row[k] for k in ('campaign', 'group', 'dataset', 'method_config', 'seed', 'block')}
        history = Path(row.get('history_csv') or run_dir / 'training_history.csv')
        if not history.is_file():
            choices = list(run_dir.glob('*history*.csv'))
            history = choices[0] if len(choices) == 1 else history
        if history.is_file():
            with history.open(newline='') as handle:
                for epoch in csv.DictReader(handle):
                    histories.append(dict(epoch, **tags))
        summary = json.loads(Path(row['summary_path']).read_text())
        selection = summary.get('loo_selection', {})
        for score in selection.get('scores', selection.get('rows', [])):
            if isinstance(score, dict):
                loo_scores.append(dict(flatten(score), **tags, candidate_block=score.get('block')))
        for epoch in summary.get('snip_selection', {}).get('search_history', []):
            for candidate in epoch['candidates']:
                snip_scores.append(dict(candidate, **tags, search_epoch=epoch['epoch'],
                                        survived=candidate['candidate_block'] in epoch['survived']))
    write_csv(output / 'training_history_long.csv', histories)
    write_csv(output / 'loo_scores_long.csv', loo_scores)
    write_csv(output / 'snip_scores_long.csv', snip_scores)


def figures(rows, selected, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures_dir = output / 'figures'; figures_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, stem):
        for suffix in ('png', 'svg', 'pdf'):
            path = figures_dir / f'{stem}.{suffix}'
            fig.savefig(path, bbox_inches='tight', dpi=180)
            files.append(str(path.relative_to(output)))
        plt.close(fig)

    grouped = defaultdict(list)
    for row in rows:
        if row['group'] in ('all_block_snip', 'all_block_snip+peft', 'looall', 'sub_removal'):
            grouped[row['campaign'], row['group'], row['dataset']].append(row)
    for key, values in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for method in sorted({r['method_config'] for r in values}):
            data = [r for r in values if r['method_config'] == method]
            for ax, metric in zip(axes, ('best_val_acc', 'final_test_acc')):
                points = sorted({r['block'] for r in data if r[metric] is not None})
                means = [statistics.mean(r[metric] for r in data if r['block'] == b and r[metric] is not None) for b in points]
                if points:
                    ax.plot(points, means, marker='o', markersize=3, label=method)
        for ax, label in zip(axes, ('Best validation accuracy (%)', 'Test accuracy (%)')):
            ax.set(xlabel='Original block index', ylabel=label, xticks=range(12), ylim=(0, 100))
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .5), fontsize=8)
        fig.suptitle(' / '.join(key))
        save(fig, 'blocks_' + hashlib.sha256('|'.join(key).encode()).hexdigest()[:12])
    for campaign in sorted({r['campaign'] for r in selected}):
        values = [r for r in selected if r['campaign'] == campaign]
        labels = sorted({r['group'] + ' / ' + r['method_config'] for r in values})
        datasets = sorted({r['dataset'] for r in values})
        matrix = np.full((len(labels), len(datasets)), np.nan)
        pivots = []
        for i, label in enumerate(labels):
            pivot = dict(campaign=campaign, method=label)
            for j, dataset in enumerate(datasets):
                scores = [r['final_test_acc'] for r in values if r['dataset'] == dataset
                          and r['group'] + ' / ' + r['method_config'] == label and r['final_test_acc'] is not None]
                if scores:
                    matrix[i, j] = statistics.mean(scores)
                    pivot[dataset] = matrix[i, j]
            pivots.append(pivot)
        write_csv(output / f'test_accuracy_wide_{campaign}.csv', pivots, ('campaign', 'method'))
        if np.isfinite(matrix).any():
            fig, ax = plt.subplots(figsize=(max(8, len(datasets)*.8), max(4, len(labels)*.28)))
            im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', vmin=0, vmax=100, cmap='viridis')
            ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha='right')
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_title(f'{campaign}: test accuracy of validation-selected/algorithm-selected runs')
            fig.colorbar(im, ax=ax, label='Test accuracy (%)')
            save(fig, f'selected_test_{campaign}')
        for dataset in datasets:
            points = [r for r in values if r['dataset'] == dataset and r['final_test_acc'] is not None
                      and number(r['trainable_params']) is not None and r['trainable_params'] > 0]
            if not points:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for group in sorted({r['group'] for r in points}):
                subset = [r for r in points if r['group'] == group]
                ax.scatter([r['trainable_params'] for r in subset], [r['final_test_acc'] for r in subset], label=group)
            ax.set(xscale='log', xlabel='Trainable parameters (log scale)', ylabel='Test accuracy (%)',
                   title=f'{campaign} / {dataset}', ylim=(0, 100))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            save(fig, f'efficiency_{campaign}_{dataset}')
            for metric, xlabel, stem in (
                ('total_params', 'Total logical model parameters', 'total_parameters'),
                ('train_wall_seconds', 'Recovery training wall time (seconds)', 'training_time'),
                ('peak_gpu_mem_allocated_mb', 'Peak allocated GPU memory (MB)', 'gpu_memory')):
                measured = [r for r in points if number(r.get(metric)) is not None and number(r[metric]) > 0]
                if not measured:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5))
                for group in sorted({r['group'] for r in measured}):
                    subset = [r for r in measured if r['group'] == group]
                    ax.scatter([float(r[metric]) for r in subset], [r['final_test_acc'] for r in subset], label=group)
                ax.set(xlabel=xlabel, ylabel='Test accuracy (%)', title=f'{campaign} / {dataset}', ylim=(0, 100))
                ax.grid(alpha=.2); ax.legend(fontsize=8)
                save(fig, f'{stem}_{campaign}_{dataset}')
    return files


def learning_figures(selected, output):
    import matplotlib.pyplot as plt
    grouped = defaultdict(list)
    for row in selected:
        directory = Path(row['summary_path']).parent
        path = Path(row.get('history_csv') or directory / 'training_history.csv')
        if not path.is_file():
            choices = list(directory.glob('*history*.csv'))
            if len(choices) != 1:
                continue
            path = choices[0]
        with path.open(newline='') as handle:
            for record in csv.DictReader(handle):
                if number(record.get('epoch')) is not None:
                    grouped[row['campaign'], row['group'], row['dataset']].append(dict(record, method_config=row['method_config']))
    files = []
    for key, values in grouped.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for method in sorted({r['method_config'] for r in values}):
            records = [r for r in values if r['method_config'] == method]
            for ax, metric in zip(axes, ('train_loss', 'val_loss', 'val_acc')):
                epochs = sorted({float(r['epoch']) for r in records if number(r.get(metric)) is not None})
                means = [statistics.mean(float(r[metric]) for r in records if float(r['epoch']) == epoch and number(r.get(metric)) is not None) for epoch in epochs]
                if epochs:
                    ax.plot(epochs, means, label=method)
        for ax, label in zip(axes, ('Training loss', 'Validation loss', 'Validation accuracy (%)')):
            ax.set(xlabel='Recovery training epoch', ylabel=label); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1, .5), loc='center left', fontsize=7)
        fig.suptitle(' / '.join(key) + ' / selected runs')
        stem = 'learning_' + hashlib.sha256('|'.join(key).encode()).hexdigest()[:12]
        for suffix in ('png', 'svg', 'pdf'):
            path = output / 'figures' / f'{stem}.{suffix}'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches='tight', dpi=180)
            files.append(str(path.relative_to(output)))
        plt.close(fig)
    return files


def compile_results(args):
    root = Path(args.root).resolve()
    output = Path(args.out_dir or root / 'compiled_results').resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, problems, coverage = read_runs(root, args.campaign)
    if not rows:
        write_csv(output / 'coverage.csv', coverage)
        (output / 'problems.json').write_text(json.dumps(problems, indent=2))
        raise ValueError('No completed suite results found; coverage.csv records planned/missing tasks')
    selected, incomplete = select(rows)
    if args.evaluate_selected or args.evaluate_all_removals:
        candidates = rows if args.evaluate_all_removals else selected
        for row in candidates:
            evaluate_removal(row, args.subtraction_dir, args.device)
        rows, problems, coverage = read_runs(root, args.campaign)
        selected, incomplete = select(rows)
    first = ('campaign', 'group', 'dataset', 'method_config', 'seed', 'block', 'peft_block',
             'best_val_acc', 'final_test_acc', 'trainable_params', 'total_params', 'summary_path')
    write_csv(output / 'all_results.csv', rows, first)
    write_csv(output / 'selected_by_validation.csv', selected, first)
    write_csv(output / 'coverage.csv', coverage)
    write_csv(output / 'incomplete_selections.csv', incomplete)
    (output / 'problems.json').write_text(json.dumps(problems, indent=2))
    for group in GROUPS:
        write_csv(output / 'by_group' / f'{group}.csv', [r for r in rows if r['group'] == group], first)
    aggregate = []
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['campaign'], row['group'], row['dataset'], row['method_config']].append(row)
    for (campaign, group, dataset, method), values in grouped.items():
        tests = [r['final_test_acc'] for r in values if r['final_test_acc'] is not None]
        aggregate.append(dict(campaign=campaign, group=group, dataset=dataset, method_config=method,
                              n_seeds=len({r['seed'] for r in values}), n_test_seeds=len(tests),
                              mean_val_acc=statistics.mean(r['best_val_acc'] for r in values),
                              mean_test_acc=statistics.mean(tests) if tests else None,
                              std_test_acc=statistics.stdev(tests) if len(tests) > 1 else None))
    write_csv(output / 'selected_summary.csv', aggregate)
    if not args.no_histories:
        auxiliary(rows, output)
    files = [] if args.no_plots else figures(rows, selected, output)
    if not args.no_plots:
        files += learning_figures(selected, output)
    import html
    links = ['<li><a href="' + html.escape(str(p.relative_to(output)), quote=True) + '">' + html.escape(str(p.relative_to(output))) + '</a></li>'
             for p in sorted(output.rglob('*.csv'))]
    links += ['<li><a href="' + html.escape(f, quote=True) + '">' + html.escape(f) + '</a></li>' for f in files]
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SFT research results</title>'
        '<h1>SFT research results</h1><p>Block sweeps are selected by validation accuracy. Missing test values stay blank. '
        'Accuracy values are percentages. Coverage and problems list missing/failed tasks.</p><ul>' + ''.join(links) + '</ul>')
    print(f'Compiled {len(rows)} completed runs; {len(selected)} selected runs; {len(incomplete)} incomplete selections. Output: {output}')


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out-dir')
    p.add_argument('--campaign')
    p.add_argument('--evaluate-selected', action='store_true')
    p.add_argument('--evaluate-all-removals', action='store_true')
    p.add_argument('--subtraction-dir', default=os.environ.get('SUBTRACTION_DIR', '/export/home/achyut/Sarvesh/SFT_COEFF_RED/SFT_BLOCK_SUBTRACTION'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--no-histories', action='store_true')
    args = p.parse_args()
    compile_results(args)


if __name__ == '__main__':
    main()
