#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2-B test-only evaluation. Never trains, selects checkpoints, or changes weights.
Requires adast_e5_adaptive_consolidation.py, adast_hmdb51_h1_dense_v2.py,
and adast_hmdb51_h2a_transfer.py in the same directory.
Inherited H1 checkpoint was test-selected: results remain exploratory.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_h1_dense_v2 import HMDB, read_hmdb_split
from adast_hmdb51_h2a_transfer import load_checkpoint


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max())}


def main():
    p = argparse.ArgumentParser(description='HMDB51 H2-B test-only evaluation')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--checkpoint', default='runs/hmdb51_h2b_policy_only_s1/best_val.pt')
    p.add_argument('--split', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--insert-block', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output', default='runs/hmdb51_h2b_test_only_s1')
    a = p.parse_args()
    if a.num_frames != 8 or a.insert_block != 6:
        raise ValueError('Checkpoint architecture requires 8 frames and insert block 6')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(a.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'H2-B validation-selected checkpoint missing: {ckpt_path}')
    train, test, classes = read_hmdb_split(a.data_root, a.split_root, a.split)
    if len(classes) != 51 or not test:
        raise RuntimeError('Invalid HMDB51 split or empty test set')
    obj, sd = load_checkpoint(ckpt_path)
    if obj.get('classes') != classes or obj.get('split') != a.split or obj.get('num_frames') != a.num_frames:
        raise RuntimeError('Checkpoint class order / split / frame count mismatch')
    if obj.get('epoch') is None or not isinstance(obj.get('val'), dict):
        raise RuntimeError('Expected a validation-selected H2-B checkpoint')
    ck_args = obj.get('args', {})
    if int(ck_args.get('insert_block', -1)) != a.insert_block:
        raise RuntimeError('Checkpoint insertion block mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'H2-B TEST ONLY | checkpoint epoch={obj["epoch"]} | test={len(test)} | device={device}', flush=True)
    print('NOTE: exploratory results; inherited H1 backbone was selected on official test.', flush=True)
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME, num_frames=8, num_labels=51, ignore_mismatched_sizes=True)
    model = AdaSTE5(bb, block=6, T=8)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f'Strict checkpoint load verified: {len(sd)} tensors; classifier shape={tuple(model.backbone.classifier.weight.shape)}', flush=True)
    loader = DataLoader(HMDB(test, proc, 8, False), batch_size=a.batch_size,
                        shuffle=False, num_workers=a.workers,
                        pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0)
    n = c1 = c5 = 0
    loss_sum = 0.0
    spatial, temporal, ks, combined, gates, entropies = [], [], [], [], [], []
    spatial_counts, temporal_counts = Counter(), Counter()
    start = time.time()
    csv_path = out / 'per_video_test.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['video', 'label', 'prediction', 'correct', 'spatial_r',
                         'k_per_frame', 'temporal_s', 'combined_effective_retention',
                         'merge_gate', 'merge_assignment_entropy'])
        with torch.inference_mode():
            for x, y, paths in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast('cuda', enabled=a.amp and device.type == 'cuda', dtype=torch.float16):
                    logits = model(x).logits
                    loss = F.cross_entropy(logits, y)
                if logits.shape[1] != 51:
                    raise RuntimeError(f'Expected 51 logits, got {logits.shape[1]}')
                pred = logits.argmax(1)
                top5 = logits.topk(5, dim=1).indices
                bs = y.numel()
                n += bs
                c1 += int((pred == y).sum().item())
                c5 += int((top5 == y[:, None]).any(dim=1).sum().item())
                loss_sum += float(loss.item()) * bs
                rr = model.last_r.float().cpu().tolist()
                ss = model.last_s.float().cpu().tolist()
                kk = model.last_k.cpu().tolist()
                gg = model.last_merge_gate.float().cpu().tolist()
                ee = model.last_merge_entropy.float().cpu().tolist()
                for i, path in enumerate(paths):
                    joint = float(rr[i]) * float(ss[i]) / model.T
                    spatial.append(float(rr[i])); temporal.append(float(ss[i]))
                    ks.append(int(kk[i])); combined.append(joint)
                    gates.append(float(gg[i])); entropies.append(float(ee[i]))
                    spatial_counts[str(rr[i])] += 1
                    temporal_counts[str(ss[i])] += 1
                    writer.writerow([path, int(y[i].item()), int(pred[i].item()),
                                     int(pred[i].item() == y[i].item()), rr[i], kk[i],
                                     ss[i], joint, gg[i], ee[i]])
                if n % 100 < bs or n == len(test):
                    print(f'{n}/{len(test)} | Top-1={100*c1/n:.2f}% | Top-5={100*c5/n:.2f}% | elapsed={time.time()-start:.0f}s', flush=True)
    if n != len(test):
        raise RuntimeError(f'Incomplete test evaluation: {n}/{len(test)}')
    result = {
        'experiment': 'H2-B HMDB51 validation-selected policy-only test evaluation',
        'protocol_warning': 'EXPLORATORY: inherited H1 dense checkpoint was selected using official HMDB51 test accuracy; this is not an independent held-out result.',
        'checkpoint': str(ckpt_path), 'selected_epoch': int(obj['epoch']),
        'selection_val_top1': obj['val'].get('top1'),
        'split': a.split, 'n': n, 'top1': 100*c1/n, 'top5': 100*c5/n,
        'loss': loss_sum/n, 'spatial_retention': stats(spatial),
        'temporal_s': stats(temporal), 'k_per_frame': stats(ks),
        'combined_effective_retention': stats(combined),
        'merge_gate': stats(gates), 'merge_assignment_entropy': stats(entropies),
        'spatial_counts': dict(spatial_counts), 'temporal_counts': dict(temporal_counts),
        'physical_token_deletion': False, 'elapsed_seconds': time.time()-start,
        'args': vars(a),
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('FINAL H2-B TEST:', json.dumps(result, indent=2), flush=True)
    print('Saved:', out / 'summary.json', flush=True)
    print('Per-video:', csv_path, flush=True)


if __name__ == '__main__':
    main()
