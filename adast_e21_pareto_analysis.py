#!/usr/bin/env python3
"""AdaST E21: offline accuracy–physical-token Pareto analysis.

Reads completed E14 fixed-budget sweep and E20 frozen component interventions.
No GPU, inference, training, or additional dependencies beyond matplotlib.
"""
import argparse
import csv
import json
from pathlib import Path


def load_points(e14_path, e20_path):
    e14 = json.loads(Path(e14_path).read_text(encoding='utf-8'))
    e20 = json.loads(Path(e20_path).read_text(encoding='utf-8'))
    fixed = e14.get('results', [])
    if len(fixed) != 15:
        raise ValueError(f'Expected 15 E14 operating points, found {len(fixed)}')
    points = []
    for item in fixed:
        if int(item['samples']) != 3783:
            raise ValueError('E14 contains a non-full-test-set operating point')
        points.append(dict(name=f"fixed_r{float(item['rho_requested']):.4f}_S{int(item['S_requested'])}",
                           kind='fixed', spatial_r=float(item['rho_requested']),
                           temporal_s=int(item['S_requested']), n=int(item['samples']),
                           top1=float(item['top1']), top5=float(item['top5']),
                           tokens=float(item['physical_compact_patch_tokens'])))
    for variant in ('full', 'spatial_only', 'temporal_only'):
        item = e20[variant]
        if int(item['n']) != 3783:
            raise ValueError(f'E20 {variant} is not a full-test-set result')
        points.append(dict(name=f'adaptive_{variant}', kind='adaptive', spatial_r='', temporal_s='',
                           n=int(item['n']), top1=float(item['top1_percent']),
                           top5=float(item['top5_percent']),
                           tokens=float(item['physical_patch_tokens_mean'])))
    return points


def dominates(a, b, eps=1e-9):
    return (a['tokens'] <= b['tokens'] + eps and a['top1'] >= b['top1'] - eps
            and (a['tokens'] < b['tokens'] - eps or a['top1'] > b['top1'] + eps))


def save_csv(path, records, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e14', default=r'.\runs\e14_budget_sweep\budget_sweep.json')
    ap.add_argument('--e20', default=r'.\runs\e20_component_ablation_amp\summary.json')
    ap.add_argument('--output', default=r'.\runs\e21_pareto_analysis')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    points = load_points(args.e14, args.e20)
    for p in points:
        p['reduction_pct'] = (1 - p['tokens'] / 1568) * 100
        dominators = [q['name'] for q in points if q is not p and dominates(q, p)]
        p['pareto'] = not dominators
        p['dominated_by'] = ';'.join(dominators)
    frontier = sorted((p for p in points if p['pareto']), key=lambda x: x['tokens'])
    fields = ['name','kind','spatial_r','temporal_s','n','top1','top5','tokens','reduction_pct','pareto','dominated_by']
    save_csv(out / 'e21_all_operating_points.csv', points, fields)
    save_csv(out / 'e21_pareto_frontier.csv', frontier, fields)
    full = next(p for p in points if p['name'] == 'adaptive_full')
    full_dominators = sorted([q for q in points if dominates(q, full)], key=lambda q: (q['tokens'], -q['top1']))
    summary = {'experiment':'E21 offline accuracy-token Pareto analysis',
               'objective':'maximize top1; minimize mean physical patch tokens',
               'test_set':'UCF101 Split-1 (3783 videos)',
               'num_points':len(points), 'num_pareto_points':len(frontier),
               'pareto_points':[p['name'] for p in frontier],
               'adaptive_full_is_pareto':full['pareto'],
               'adaptive_full_dominated_by':[{'name':q['name'],'top1':q['top1'],'tokens':q['tokens']} for q in full_dominators],
               'caution':'Frozen budget interventions; accuracy-token dominance does not imply latency, FLOPs, energy, or statistical significance. Do not tune policies on test data.'}
    (out / 'e21_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.2, 5.3))
        fixed = [p for p in points if p['kind']=='fixed']
        adaptive = [p for p in points if p['kind']=='adaptive']
        ax.scatter([p['tokens'] for p in fixed], [p['top1'] for p in fixed], label='Fixed budgets', marker='o', alpha=.8)
        ax.scatter([p['tokens'] for p in adaptive], [p['top1'] for p in adaptive], label='Adaptive interventions', marker='s', s=65)
        ax.plot([p['tokens'] for p in frontier], [p['top1'] for p in frontier], linestyle='--', linewidth=1.6, label='Pareto frontier')
        ax.annotate('Full AdaST', (full['tokens'],full['top1']), xytext=(9,-19), textcoords='offset points')
        ax.set(xlabel='Mean physical patch tokens per video (lower is better)', ylabel='Top-1 accuracy (%) (higher is better)', title='AdaST E21 | UCF101 Split-1')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'e21_pareto_frontier.png', dpi=300)
        fig.savefig(out / 'e21_pareto_frontier.pdf')
        plt.close(fig)
    except ImportError:
        print('matplotlib not installed; CSV/JSON generated, plot skipped')
    print(json.dumps(summary,indent=2))
    print('Saved outputs to:', out.resolve())


if __name__ == '__main__':
    main()
