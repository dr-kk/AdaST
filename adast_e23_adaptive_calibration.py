#!/usr/bin/env python3
"""E23: frozen E12 controller-logit calibration on its reconstructed validation split.

Exploratory calibration: E12 already used this validation split for checkpoint selection.
No official test access in this script. Run one final locked-policy test separately.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import adast_e14_budget_sweep as e14
import adast_e6_v2_pretrained_compact as e6
import adast_e12_rep_preserve as e12
from adast_e22b_e12_validation import loader, fingerprint, dump, write_csv


def pathset(items):
    return {str(Path(x[0] if isinstance(x, (tuple, list)) else x).resolve()).casefold() for x in items}


def bias_hook(beta):
    def apply(_module, _inputs, logits):
        # Penalize larger budget levels; retain per-video controller rankings.
        idx = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        return logits - float(beta) * idx.unsqueeze(0)
    return apply


@torch.no_grad()
def evaluate(model, dl, device, args, sb, tb):
    model.eval()
    model.force_spatial_r = None
    model.force_temporal_s = None
    hs = [model.budget.register_forward_hook(bias_hook(sb)),
          model.temporal_budget.register_forward_hook(bias_hook(tb))]
    total = good1 = good5 = 0
    token_sum = k_sum = s_sum = 0.0
    spatial_counts, temporal_counts = {}, {}
    try:
        for i, batch in enumerate(dl, 1):
            x, y, _ = e12.unpack_batch(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=args.amp and device.type == 'cuda'):
                logits = model(x).logits
            pred = logits.argmax(dim=1)
            good1 += int(pred.eq(y).sum().item())
            good5 += int(logits.topk(min(5, logits.shape[-1]), dim=1).indices.eq(y[:, None]).any(dim=1).sum().item())
            n = int(y.numel())
            total += n
            token_sum += float(model.last_compact_tokens) * n
            ks = model.last_k.detach().cpu().flatten().tolist()
            ss = model.last_s.detach().cpu().flatten().tolist()
            k_sum += sum(map(float, ks))
            s_sum += sum(map(float, ss))
            for k in ks:
                spatial_counts[str(int(k))] = spatial_counts.get(str(int(k)), 0) + 1
            for s in ss:
                temporal_counts[str(int(s))] = temporal_counts.get(str(int(s)), 0) + 1
            if i % args.print_every == 0 or i == len(dl):
                print(f'  {i}/{len(dl)} top1={100*good1/total:.3f}% tokens={token_sum/total:.1f}', flush=True)
    finally:
        for h in hs:
            h.remove()
    if total != len(dl.dataset):
        raise RuntimeError(f'Evaluation incomplete: {total} vs {len(dl.dataset)}')
    mean_tokens = token_sum / total
    return dict(spatial_bias=float(sb), temporal_bias=float(tb), samples=total,
                correct_top1=good1, correct_top5=good5,
                top1=100*good1/total, top5=100*good5/total,
                physical_compact_patch_tokens=mean_tokens,
                physical_token_reduction_pct=100*(1-mean_tokens/(196*args.num_frames)),
                K_per_frame=k_sum/total, S_mean=s_sum/total,
                spatial_k_counts=spatial_counts, temporal_s_counts=temporal_counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--e12-checkpoint', required=True)
    p.add_argument('--output', default=r'.\runs\e23_adaptive_calibration')
    p.add_argument('--model-name', default='facebook/timesformer-base-finetuned-k400')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--print-every', type=int, default=250)
    p.add_argument('--token-cap', type=float, default=750.0)
    p.add_argument('--spatial-biases', type=float, nargs='+', default=[0.0, 0.5, 1.5])
    p.add_argument('--temporal-biases', type=float, nargs='+', default=[0.0, 0.5, 1.5])
    a = p.parse_args()
    checkpoint = torch.load(a.e12_checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('args', {})
    for k in ('seed','val_fraction','split','num_frames','insert_block','temperature'):
        if k not in cfg:
            raise RuntimeError(f'Checkpoint missing required setting: {k}')
    if cfg['num_frames'] != 8 or cfg['split'] != 1:
        raise RuntimeError('This experiment is configured for UCF101 split-1 with 8 frames')
    a.num_frames = cfg['num_frames']
    a.seed = cfg['seed']
    e14.seed_all(a.seed)
    all_train, test = e6.read_split(Path(a.data_root), Path(a.split_root), cfg['split'])
    train, val = e6.stratified(all_train, cfg['val_fraction'], cfg['seed'])
    if (len(train), len(val), len(test)) != (8578, 959, 3783):
        raise RuntimeError('Unexpected E12 split sizes')
    if pathset(train) & pathset(val) or pathset(val) & pathset(test) or pathset(train) & pathset(test):
        raise RuntimeError('Partition overlap detected')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(checkpoint_sha256=hashlib.sha256(Path(a.e12_checkpoint).read_bytes()).hexdigest(),
                validation_sha256=fingerprint(val), seed=a.seed, val_fraction=cfg['val_fraction'],
                spatial_biases=a.spatial_biases, temporal_biases=a.temporal_biases,
                token_cap=a.token_cap, amp=a.amp, model_name=a.model_name,
                method='subtract beta * ordered budget-level index from each controller logit',
                selection_rule='max validation top1 with tokens <= cap; tie: fewer tokens',
                limitation='E12 checkpoint-selection validation set reused; not independent')
    manifest = out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != meta:
            raise RuntimeError('Manifest changed; choose a new output directory')
    else:
        dump(manifest, meta)
    print('[AUDIT] E12 train=8578, val=959, test=3783, no overlap')
    print('[SAFETY] Test loader is never constructed or evaluated.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(a.model_name)
    dl = loader(val, processor, a)
    backbone = TimesformerForVideoClassification.from_pretrained(
        a.model_name, num_labels=101, ignore_mismatched_sizes=True)
    model = e14.AdaSTE14(backbone, block=cfg['insert_block'],
                          T=cfg['num_frames'], temperature=cfg['temperature'])
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, a.e12_checkpoint)
    model.to(device).eval()
    rows_path = out/'validation_results.json'
    rows = json.loads(rows_path.read_text(encoding='utf-8')) if rows_path.exists() else []
    seen = {(r['spatial_bias'], r['temporal_bias']) for r in rows}
    for sb in a.spatial_biases:
        for tb in a.temporal_biases:
            if (sb, tb) in seen:
                print('[RESUME]', sb, tb)
                continue
            print(f'\n[VALIDATION] spatial_bias={sb}, temporal_bias={tb}', flush=True)
            row = evaluate(model, dl, device, a, sb, tb)
            rows.append(row)
            dump(rows_path, rows)
            write_csv(out/'validation_results.csv', rows)
    eligible = [r for r in rows if r['physical_compact_patch_tokens'] <= a.token_cap]
    if not eligible:
        raise RuntimeError('No calibrated policy meets token cap')
    selected = min(eligible, key=lambda r: (-r['top1'], r['physical_compact_patch_tokens'],
                                           r['spatial_bias'], r['temporal_bias']))
    original = next((r for r in rows if r['spatial_bias']==0 and r['temporal_bias']==0), None)
    if original is None:
        raise RuntimeError('Grid must include (0,0) original adaptive baseline')
    dump(out/'frozen_calibration.json', dict(selected=selected, original=original,
         checkpoint_selection_validation_reused=True, official_test_evaluated=False,
         selection_rule=meta['selection_rule']))
    print('\nFROZEN CALIBRATION:', selected['spatial_bias'], selected['temporal_bias'],
          'top1=', selected['top1'], 'tokens=', selected['physical_compact_patch_tokens'])
    print('Saved:', (out/'frozen_calibration.json').resolve())


if __name__ == '__main__':
    main()
