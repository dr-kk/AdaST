#!/usr/bin/env python3
"""E22-B: reconstructed E12 validation-selected fixed-budget vs original adaptive AdaST.

This is a baseline/selection study, NOT learned-controller calibration.
Official UCF101 test set is used only after validation selection is frozen.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import adast_e14_budget_sweep as e14
import adast_e6_v2_pretrained_compact as e6


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fingerprint(items):
    s = '\n'.join(str(x) for x in items)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def loader(items, processor, args):
    return DataLoader(e6.UCF(items, processor, args.num_frames),
                      batch_size=1, shuffle=False, drop_last=False,
                      num_workers=args.workers,
                      pin_memory=torch.cuda.is_available())


def run_point(model, dl, device, args, rho, s, name):
    if rho is None and s is None:
        model.force_spatial_r = None
        model.force_temporal_s = None
    else:
        model.set_forced_budget(rho, s)
    result = e14.evaluate_point(model, dl, device, args, rho, s) if rho is not None else evaluate_adaptive(model, dl, device, args)
    result['name'] = name
    result['mode'] = 'adaptive_original' if rho is None else 'fixed'
    return result


@torch.no_grad()
def evaluate_adaptive(model, dl, device, args):
    # Mirrors E14's inference, but retains original argmax budget choices.
    import numpy as np
    import adast_e12_rep_preserve as e12
    model.eval()
    model.force_spatial_r = None
    model.force_temporal_s = None
    n = good1 = good5 = 0
    toks = []
    kvals = []
    svals = []
    for i, batch in enumerate(dl, 1):
        x, y, _ = e12.unpack_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=args.amp and device.type == 'cuda'):
            logits = model(x).logits
        good1 += int((logits.argmax(1) == y).sum().item())
        good5 += int(logits.topk(min(5, logits.shape[-1]), dim=1).indices.eq(y[:, None]).any(1).sum().item())
        n += int(y.numel())
        toks.append(float(model.last_compact_tokens))
        kvals.extend(model.last_k.float().cpu().tolist())
        svals.extend(model.last_s.float().cpu().tolist())
        if i % args.print_every == 0 or i == len(dl):
            print(f'  adaptive {i}/{len(dl)} top1={100*good1/n:.3f}%')
    mean_tok = float(np.mean(toks))
    return dict(samples=n, top1=100*good1/n, top5=100*good5/n,
                physical_compact_patch_tokens=mean_tok,
                physical_token_reduction_pct=100*(1-mean_tok/(196*args.num_frames)),
                K_per_frame=float(np.mean(kvals)), S_mean=float(np.mean(svals)),
                rho_requested=None, S_requested=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--e12-checkpoint', required=True)
    p.add_argument('--output', default=r'.\runs\e22b_e12_validation')
    p.add_argument('--model-name', default='facebook/timesformer-base-finetuned-k400')
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--insert-block', type=int, default=6)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-count', type=int, default=None,
                   help='Deprecated; exact E12 validation partition is always used')
    p.add_argument('--token-cap', type=float, default=750.0,
                   help='Precommitted maximum mean physical patch tokens')
    p.add_argument('--print-every', type=int, default=250)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--validation-only', action='store_true',
                   help='Stop after validation selection; do not open official test')
    a = p.parse_args()
    if a.val_count is not None or a.num_frames != 8:
        p.error('Do not specify --val-count; exact E12 validation partition is used. Requires 8 frames')
    e14.seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    train_items, test_items = e6.read_split(Path(a.data_root), Path(a.split_root), a.split)
    if len(train_items) != 9537 or len(test_items) != 3783:
        raise RuntimeError(f'Unexpected UCF101 split counts: train={len(train_items)}, test={len(test_items)}')
    ckpt = torch.load(a.e12_checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get('args', {})
    for key, value in [('split', a.split), ('num_frames', a.num_frames),
                       ('insert_block', a.insert_block), ('temperature', a.temperature)]:
        if key in cfg and cfg[key] != value:
            raise RuntimeError(f'E12 checkpoint {key}={cfg[key]} differs from E22 {value}')
    if 'seed' not in cfg or 'val_fraction' not in cfg:
        raise RuntimeError('Checkpoint must record E12 seed and val_fraction')
    e12_train_items, val_items = e6.stratified(train_items, cfg['val_fraction'], cfg['seed'])
    def pathset(items):
        return {str(Path(z[0] if isinstance(z, (tuple, list)) else z).resolve()).casefold() for z in items}
    tr, va, te = map(pathset, (e12_train_items, val_items, test_items))
    if tr & va or tr & te or va & te:
        raise RuntimeError('E12 reconstructed partitions overlap')
    if (len(e12_train_items), len(val_items)) != (8578, 959):
        raise RuntimeError('Unexpected E12 partition sizes')
    print(f'[AUDIT] E12 train={len(tr)}, validation={len(va)}, test={len(te)}; zero path overlap')
    meta = dict(seed=a.seed, e12_split_seed=cfg['seed'], e12_val_fraction=cfg['val_fraction'],
                split=a.split, val_count=len(val_items),
                validation_sha256=fingerprint(val_items),
                train_sha256=fingerprint(train_items), test_sha256=fingerprint(test_items),
                token_cap=a.token_cap, checkpoint=str(Path(a.e12_checkpoint).resolve()),
                model_name=a.model_name, num_frames=a.num_frames, insert_block=a.insert_block,
                temperature=a.temperature, amp=a.amp,
                selection_rule='max validation top1 subject to mean physical tokens <= cap; ties: fewer tokens')
    manifest = out / 'selection_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous != meta:
            raise RuntimeError('Existing manifest differs; choose a new output directory for changed settings')
    else:
        dump(manifest, meta)
    (out / 'validation_paths.txt').write_text('\n'.join(str(x[0] if isinstance(x, (tuple, list)) else x) for x in val_items)+'\n', encoding='utf-8')
    print('Validation selection uses reconstructed E12 validation entries:', len(val_items))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(a.model_name)
    val_dl = loader(val_items, processor, a)
    backbone = TimesformerForVideoClassification.from_pretrained(
        a.model_name, num_labels=101, ignore_mismatched_sizes=True)
    model = e14.AdaSTE14(backbone, block=a.insert_block,
                          T=a.num_frames, temperature=a.temperature)
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, a.e12_checkpoint)
    model.to(device).eval()
    path = out / 'validation_results.json'
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    seen = {r['name'] for r in rows}
    points = [(f'fixed_r{rho:.4f}_S{s}', rho, s)
              for s in e14.TEMPORAL_LEVELS for rho in e14.SPATIAL_LEVELS]
    points.append(('adaptive_original', None, None))
    for name, rho, s in points:
        if name in seen:
            print('[RESUME]', name)
            continue
        print('\n[VALIDATION]', name)
        result = run_point(model, val_dl, device, a, rho, s, name)
        rows.append(result)
        dump(path, rows)
        write_csv(out / 'validation_results.csv', rows)
    fixed_eligible = [r for r in rows if r['mode'] == 'fixed'
                      and r['physical_compact_patch_tokens'] <= a.token_cap]
    if not fixed_eligible:
        raise RuntimeError('No fixed operating point meets the precommitted token cap')
    selected = sorted(fixed_eligible,
                      key=lambda r: (-r['top1'], r['physical_compact_patch_tokens'], r['name']))[0]
    decision = dict(selected_fixed=selected, original_adaptive=next(r for r in rows if r['name']=='adaptive_original'),
                    selection_source='reconstructed E12 checkpoint-selection validation partition (reused)',
                    test_data_used_for_selection=False)
    dump(out / 'frozen_selection.json', decision)
    print('\nFROZEN VALIDATION CHOICE:', selected['name'],
          'val top1=', selected['top1'], 'tokens=', selected['physical_compact_patch_tokens'])
    if a.validation_only:
        print('Validation-only mode: official test not evaluated.')
        return
    test_path = out / 'test_results.json'
    if test_path.exists():
        print('Test results already exist; refusing automatic re-evaluation:', test_path)
        return
    print('\n[TEST] Evaluating only frozen fixed choice and original adaptive baseline')
    test_dl = loader(test_items, processor, a)
    test_rows = []
    for name, rho, s in [(selected['name'], selected['rho_requested'], selected['S_requested']),
                         ('adaptive_original', None, None)]:
        print('[TEST]', name)
        result = run_point(model, test_dl, device, a, rho, s, name)
        test_rows.append(result)
        dump(out / 'test_results_partial.json', test_rows)
    dump(test_path, test_rows)
    write_csv(out / 'test_results.csv', test_rows)
    print('\nE22-A complete. Outputs:', out.resolve())


if __name__ == '__main__':
    main()
