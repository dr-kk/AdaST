"""Clean H19: training-only, retrospective multi-view budget eligibility audit.
No model loading, GPU, validation/test, or checkpoint modification.
"""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

BUDGETS = (6, 7, 8)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views-csv', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--h16-summary', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--loss-tolerance', type=float, default=None,
                   help='Must match H16 protocol tolerance; no post-hoc tuning.')
    a = p.parse_args()
    paths = {name: Path(getattr(a, name)) for name in ('views_csv','protocol','h16_summary','profile')}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}')
    protocol = json.loads(paths['protocol'].read_text(encoding='utf-8'))
    h16 = json.loads(paths['h16_summary'].read_text(encoding='utf-8'))
    if protocol.get('profile_sha256') != sha256(paths['profile']):
        raise RuntimeError('H9-A profile SHA256 does not match H16 protocol')
    tolerance = float(protocol.get('loss_tolerance', -1))
    if tolerance < 0 or (a.loss_tolerance is not None and abs(a.loss_tolerance-tolerance) > 1e-12):
        raise RuntimeError('Loss tolerance must match frozen H16 protocol')
    expected_n = int(protocol['args']['per_target']) * 3
    expected_views = int(protocol['args']['views'])
    if expected_n != int(h16['selected_n']) or expected_views != int(h16['views_per_video']):
        raise RuntimeError('H16 protocol/summary mismatch')
    selected = set(map(int, protocol['selected_indices']))
    if len(selected) != expected_n:
        raise RuntimeError('Invalid selected indices in protocol')
    original = {}
    with paths['profile'].open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            idx = int(row['index'])
            if idx in original:
                raise RuntimeError('Duplicate H9-A profile index')
            original[idx] = row
    if len(original) != int(protocol['train_n']):
        raise RuntimeError('H9-A train count mismatch')
    grouped = defaultdict(dict)
    view_rows = []
    with paths['views_csv'].open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'index','path','label','original_target','view','target_s','loss_s6','loss_s7','loss_s8','correct_s6','correct_s7','correct_s8'}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError('H16 CSV missing required fields: '+str(sorted(required-set(reader.fieldnames or []))))
        for row in reader:
            idx, view = int(row['index']), int(row['view'])
            if idx not in selected or not 1 <= view <= expected_views or view in grouped[idx]:
                raise RuntimeError('Unexpected or duplicate index/view in H16 CSV')
            src = original[idx]
            if row['path'] != src['path'] or int(row['label']) != int(src['label']) or int(row['original_target']) != int(src['target_s']):
                raise RuntimeError(f'H16/H9-A identity mismatch at index {idx}')
            loss = {s: float(row[f'loss_s{s}']) for s in BUDGETS}
            correct = {s: int(row[f'correct_s{s}']) for s in BUDGETS}
            if any(c not in (0,1) for c in correct.values()):
                raise RuntimeError('Invalid correctness value')
            eligible = {s: bool(correct[s] and loss[s] <= loss[8] + tolerance + 1e-9) for s in BUDGETS}
            if bool(correct[8]) != eligible[8]:
                raise RuntimeError('S8 eligibility invariant failed')
            derived = min((s for s in BUDGETS if eligible[s]), default=8)
            if int(row['target_s']) != derived:
                raise RuntimeError(f'H16 target rule mismatch at index={idx}, view={view}')
            grouped[idx][view] = (row, eligible, correct)
            view_rows.append({'index':idx, 'view':view, 'original_target':int(row['original_target']),
                              's8_correct':correct[8], **{f'eligible_s{s}':int(eligible[s]) for s in BUDGETS}})
    if len(grouped) != expected_n or len(view_rows) != expected_n*expected_views:
        raise RuntimeError('Incomplete H16 audit')
    if any(set(v) != set(range(1,expected_views+1)) for v in grouped.values()):
        raise RuntimeError('Missing view for one or more videos')
    result = []
    status = Counter()
    by_original = defaultdict(Counter)
    for idx in sorted(grouped):
        views = [grouped[idx][v] for v in range(1,expected_views+1)]
        eligible_all = {s:all(item[1][s] for item in views) for s in BUDGETS}
        eligible_any = {s:any(item[1][s] for item in views) for s in BUDGETS}
        s8_all_correct = all(item[2][8] for item in views)
        # 'robust' is an observed three-view eligibility label, NOT a safety guarantee.
        if not s8_all_correct:
            label, category = '', 'unresolved_s8_error'
        elif eligible_all[6]:
            label, category = '6', 'eligible_s6_all_views'
        elif eligible_all[7]:
            label, category = '7', 'eligible_s7_all_views'
        elif eligible_all[8]:
            label, category = '8', 'eligible_s8_all_views_only'
        else:
            raise RuntimeError('Unexpected budget eligibility state')
        orig = int(views[0][0]['original_target'])
        status[category] += 1
        by_original[str(orig)][category] += 1
        result.append({'index':idx, 'path':original[idx]['path'], 'label':original[idx]['label'],
                       'original_target':orig, 'observed_robust_budget':label, 'category':category,
                       's8_correct_all_views':int(s8_all_correct),
                       **{f's{s}_eligible_all_views':int(eligible_all[s]) for s in BUDGETS},
                       **{f's{s}_eligible_any_view':int(eligible_any[s]) for s in BUDGETS},
                       **{f's{s}_eligible_view_count':sum(int(v[1][s]) for v in views) for s in BUDGETS},
                       'view_targets':'|'.join(str(v[0]['target_s']) for v in views)})
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Output directory is not empty: {out}. Choose a fresh output path.')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'per_video_budget_eligibility.csv', result, list(result[0]))
    write_csv(out/'per_view_budget_eligibility.csv', view_rows, list(view_rows[0]))
    summary = {'experiment':'Clean H19 retrospective three-view budget eligibility audit',
               'sample_n':len(result), 'views_per_video':expected_views,
               'loss_tolerance':tolerance, 'category_counts':dict(sorted(status.items())),
               'by_original_target':{k:dict(sorted(v.items())) for k,v in sorted(by_original.items())},
               's6_eligible_all_views_n':sum(r['s6_eligible_all_views'] for r in result),
               's7_eligible_all_views_n':sum(r['s7_eligible_all_views'] for r in result),
               's8_correct_all_views_n':sum(r['s8_correct_all_views'] for r in result),
               'source_sha256':{k:sha256(v) for k,v in paths.items()},
               'limitations':['H16 sample stratified by original pseudo-target; not population representative.',
                              'Eligibility means correct AND loss <= S8 loss + frozen tolerance on each sampled view.',
                              'S8 incorrect on any sampled view is unresolved, not safe.',
                              'Three-view agreement does not guarantee unseen-view correctness or real-world safety.',
                              'Retrospective oracle uses true labels and forced-budget losses; cannot be deployed as an inference-time controller.',
                              'No validation/test data, no GPU, no training; shape-preserving gating is not physical token deletion.']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n',encoding='utf-8')
    print('H19 RESULT:', json.dumps(summary))
    print('CLEAN H19 COMPLETE:', (out/'summary.json').resolve())

if __name__ == '__main__':
    main()
