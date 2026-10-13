"""AdaST E24: frozen fixed-budget official-test paired audit.

Run from the AdaST repository directory. Does not train or select a policy.
Requires adast_e14_budget_sweep.py and adast_e12_rep_preserve.py.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import adast_e14_budget_sweep as e14
import adast_e12_rep_preserve as e12
from transformers import AutoImageProcessor, TimesformerForVideoClassification


def key_of(path):
    # Relative class/video ID is stable when C: and D: contain the same dataset.
    parts = str(path).replace('\\', '/').strip().split('/')
    if len(parts) < 2:
        raise ValueError(f'Invalid video path: {path!r}')
    return '/'.join(parts[-2:]).lower()


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_existing(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 3783:
        raise RuntimeError(f'E12 CSV must contain 3783 entries, found {len(rows)}')
    required = {'video', 'label', 'prediction', 'correct', 'compact_patch_tokens'}
    if not required.issubset(rows[0]):
        raise RuntimeError(f'E12 CSV missing columns: {required - set(rows[0])}')
    lookup = {}
    for r in rows:
        k = key_of(r['video'])
        if k in lookup:
            raise RuntimeError(f'Duplicate E12 key: {k}')
        lookup[k] = r
    return lookup


def compute_paired(fixed, adaptive, n_boot, seed):
    a_ok, f_ok, a_tokens, f_tokens = [], [], [], []
    paired = []
    if len(fixed) != len(adaptive) or set(r['video_id'] for r in fixed) != set(adaptive):
        raise RuntimeError('E12 and fixed predictions are not a one-to-one match')
    for r in fixed:
        k = r['video_id']
        ref = adaptive[k]
        if int(r['label']) != int(ref['label']):
            raise RuntimeError(f'Ground-truth label mismatch: {k}')
        ac = int(ref['prediction']) == int(ref['label'])
        fc = int(r['prediction']) == int(r['label'])
        if int(ref['correct']) != int(ac):
            raise RuntimeError(f'Incorrect E12 correct flag: {k}')
        a_ok.append(int(ac)); f_ok.append(int(fc))
        a_tokens.append(float(ref['compact_patch_tokens']))
        f_tokens.append(float(r['compact_patch_tokens']))
        paired.append({
            'video_id': k, 'label': int(r['label']),
            'adaptive_prediction': int(ref['prediction']),
            'fixed_prediction': int(r['prediction']),
            'adaptive_correct': int(ac), 'fixed_correct': int(fc),
            'adaptive_patch_tokens': float(ref['compact_patch_tokens']),
            'fixed_patch_tokens': float(r['compact_patch_tokens'])
        })
    a = np.asarray(a_ok, dtype=np.int8)
    f = np.asarray(f_ok, dtype=np.int8)
    at = np.asarray(a_tokens, dtype=float)
    ft = np.asarray(f_tokens, dtype=float)
    both_correct = int(np.sum((a == 1) & (f == 1)))
    both_wrong = int(np.sum((a == 0) & (f == 0)))
    adaptive_only = int(np.sum((a == 1) & (f == 0)))
    fixed_only = int(np.sum((a == 0) & (f == 1)))
    discordant = adaptive_only + fixed_only
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(adaptive_only, fixed_only), discordant, 0.5).pvalue) if discordant else 1.0
    except ImportError:
        if discordant:
            tail = sum(math.comb(discordant, i) for i in range(min(adaptive_only, fixed_only) + 1)) / (2 ** discordant)
            p = float(min(1.0, 2.0 * tail))
        else:
            p = 1.0
    rng = np.random.default_rng(seed)
    delta = (f.astype(float) - a.astype(float)) * 100.0
    bootstrap = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ix = rng.integers(0, len(delta), len(delta))
        bootstrap[i] = np.mean(delta[ix])
    ci = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    summary = {
        'experiment': 'E24_frozen_fixed_vs_E12_adaptive',
        'dataset': 'UCF101 official Split-1 test',
        'samples': len(paired),
        'fixed_policy': {'rho': 0.4375, 'K': 86, 'S': 7},
        'adaptive_top1_pct': float(np.mean(a) * 100),
        'fixed_top1_pct': float(np.mean(f) * 100),
        'fixed_minus_adaptive_top1_pp': float(np.mean(delta)),
        'paired_bootstrap_95pct_ci_pp': [float(x) for x in ci],
        'bootstrap_resamples': n_boot,
        'mcnemar_exact_two_sided_p': p,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'adaptive_correct_fixed_wrong': adaptive_only,
        'fixed_correct_adaptive_wrong': fixed_only,
        'adaptive_mean_patch_tokens': float(np.mean(at)),
        'fixed_mean_patch_tokens': float(np.mean(ft)),
        'fixed_token_saving_pct_vs_adaptive': float(100 * (1 - np.mean(ft) / np.mean(at))),
        'token_metric': 'physical compact patch tokens; not measured wall-clock speedup',
        'selection_provenance': 'fixed K=86 S=7 selected on reused E12 checkpoint-selection validation; no test retuning',
    }
    return paired, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--e12-checkpoint', required=True)
    p.add_argument('--e12-csv', default='runs/e12_rep_preserve/per_video_test.csv')
    p.add_argument('--model-name', default='facebook/timesformer-base-finetuned-k400')
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--insert-block', type=int, default=6)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--print-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output', default='runs/e24_paired_test')
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('--bootstrap must be at least 100')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    adaptive = load_existing(Path(args.e12_csv))
    print(f'[AUDIT] E12 reference entries: {len(adaptive)}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[DEVICE]', device)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    loader = e14.build_test_loader(args, processor)
    if len(loader.dataset) != 3783:
        raise RuntimeError(f'Expected 3783 official test videos, found {len(loader.dataset)}')
    backbone = TimesformerForVideoClassification.from_pretrained(
        args.model_name, num_labels=101, ignore_mismatched_sizes=True)
    model = e14.AdaSTE14(backbone, block=args.insert_block,
                          T=args.num_frames, temperature=args.temperature)
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, args.e12_checkpoint)
    model.to(device).eval()
    model.set_forced_budget(0.4375, 7)
    rows = []
    print('[FROZEN] rho=0.4375, K=86, S=7; test evaluation only')
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            x, y, path = e12.unpack_batch(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=args.amp and device.type == 'cuda'):
                logits = model(x).logits
            if y.numel() != 1:
                raise RuntimeError('Expected batch size 1')
            if path is None:
                raise RuntimeError('Dataset did not return a video path')
            video = path[0] if isinstance(path, (list, tuple)) else path
            if isinstance(video, Path):
                video = str(video)
            k = key_of(video)
            prediction = int(logits.argmax(dim=1).item())
            label = int(y.item())
            k_actual = int(round(float(model.last_k.reshape(-1)[0].item())))
            s_actual = int(round(float(model.last_s.reshape(-1)[0].item())))
            tokens = float(model.last_compact_tokens)
            if k_actual != 86 or s_actual != 7 or abs(tokens - 602) > 1e-4:
                raise RuntimeError(f'Unexpected forced budget at {k}: K={k_actual}, S={s_actual}, tokens={tokens}')
            rows.append({'video_id': k, 'video': str(video), 'label': label,
                         'prediction': prediction, 'correct': int(prediction == label),
                         'k_per_frame': k_actual, 'temporal_S': s_actual,
                         'compact_patch_tokens': tokens})
            if step % args.print_every == 0 or step == len(loader):
                print(f'{step}/{len(loader)} | fixed Top1={100*sum(r["correct"] for r in rows)/len(rows):.4f}%')
    if len(rows) != 3783 or len(set(r['video_id'] for r in rows)) != 3783:
        raise RuntimeError('Incomplete or duplicated fixed predictions')
    write_csv(out / 'per_video_fixed.csv', rows, list(rows[0]))
    paired, summary = compute_paired(rows, adaptive, args.bootstrap, args.seed)
    write_csv(out / 'paired_per_video.csv', paired, list(paired[0]))
    with open(out / 'paired_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\n=== E24 PAIRED TEST RESULTS ===')
    print(json.dumps(summary, indent=2))
    print(f'[SAVE] {out / "per_video_fixed.csv"}')
    print(f'[SAVE] {out / "paired_per_video.csv"}')
    print(f'[SAVE] {out / "paired_statistics.json"}')


if __name__ == '__main__':
    main()
