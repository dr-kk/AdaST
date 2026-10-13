#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2-C diagnostic: four inference modes, same H2-B checkpoint and HMDB51 test set.
No training, no checkpoint selection. Exploratory: inherited H1 was test-selected.
Dense mode uses H2-B's frozen backbone (NOT the original H1 checkpoint).
"""
import argparse, csv, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME, LEVELS
from adast_hmdb51_h1_dense_v2 import HMDB, read_hmdb_split
from adast_hmdb51_h2a_transfer import load_checkpoint

class DiagnosticE5(AdaSTE5):
    mode = 'full'
    def _hook(self, module, args, kwargs):
        if self.mode == 'full':
            return super()._hook(module, args, kwargs)
        if self.mode == 'dense':
            return args, kwargs
        if args:
            h, rest, source = args[0], args[1:], 'args'
        else:
            h, rest, source = kwargs['hidden_states'], (), 'kwargs'
        cls, patch = h[:, :1], h[:, 1:]
        b, m, d = patch.shape
        if m % self.T:
            raise RuntimeError('Unexpected patch count for temporal grouping')
        n = m // self.T
        z = patch.reshape(b, self.T, n, d)
        scores, delta = self.scorer(patch, self.T)
        logits = self.budget(scores, delta, cls)
        idx = logits.argmax(1)
        r = self.levels[idx]
        k = torch.round(r * n).long().clamp(1, n)
        spatial = torch.zeros((b, self.T, n), device=patch.device, dtype=patch.dtype)
        sf = scores.reshape(b, self.T, n)
        for bi in range(b):
            indices = torch.topk(sf[bi], int(k[bi].item()), dim=1, sorted=False).indices
            spatial[bi].scatter_(1, indices, 1.0)
        if self.mode == 'spatial':
            s = torch.full((b,), self.T, device=patch.device, dtype=patch.dtype)
            joint = spatial.reshape(b, m)
        elif self.mode == 'spatiotemporal':
            temporal_logits = self.temporal_budget(z)
            s = self.temporal_levels[temporal_logits.argmax(1)]
            frame_feat = z.mean(2)
            novelty = torch.zeros((b, self.T), device=patch.device, dtype=patch.dtype)
            novelty[:, 0] = frame_feat[:, 0].pow(2).mean(-1).sqrt()
            if self.T > 1:
                novelty[:, 1:] = (frame_feat[:, 1:] - frame_feat[:, :-1]).pow(2).mean(-1).sqrt()
            temporal_mask = torch.zeros_like(novelty)
            for bi in range(b):
                ids = torch.topk(novelty[bi], int(s[bi].item()), sorted=False).indices
                temporal_mask[bi].scatter_(0, ids, 1.0)
            joint = (spatial * temporal_mask[:, :, None]).reshape(b, m)
        else:
            raise RuntimeError('Unknown diagnostic mode: ' + self.mode)
        self.last_r, self.last_s, self.last_k = r.detach(), s.detach(), k.detach()
        gh = torch.cat((cls, patch * joint.unsqueeze(-1)), dim=1)
        if source == 'args':
            return (gh,) + rest, kwargs
        new_kwargs = dict(kwargs)
        new_kwargs['hidden_states'] = gh
        return args, new_kwargs

def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))

def main():
    p = argparse.ArgumentParser(description='H2-C test-only component ablation')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--checkpoint', default='runs/hmdb51_h2b_policy_only_s1/best_val.pt')
    p.add_argument('--split', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--insert-block', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output', default='runs/hmdb51_h2c_diagnostic_s1')
    a = p.parse_args()
    if a.num_frames != 8 or a.insert_block != 6:
        raise ValueError('H2-B checkpoint requires 8 frames, insertion block 6')
    if a.batch_size < 1:
        raise ValueError('batch-size must be positive')
    ckpt = Path(a.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    _, test, classes = read_hmdb_split(a.data_root, a.split_root, a.split)
    obj, sd = load_checkpoint(ckpt)
    if obj.get('classes') != classes or obj.get('split') != a.split or obj.get('num_frames') != a.num_frames:
        raise RuntimeError('Checkpoint class order, split, or frames mismatch')
    if obj.get('epoch') is None or not isinstance(obj.get('val'), dict):
        raise RuntimeError('Expected validation-selected H2-B checkpoint')
    if int(obj.get('args', {}).get('insert_block', -1)) != a.insert_block:
        raise RuntimeError('Checkpoint insertion block mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME, num_frames=8, num_labels=51, ignore_mismatched_sizes=True)
    model = DiagnosticE5(backbone, block=6, T=8)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f'Strict checkpoint load: {len(sd)} tensors; classifier={tuple(model.backbone.classifier.weight.shape)}', flush=True)
    print('WARNING: Exploratory; inherited H1 backbone was selected using official test accuracy.', flush=True)
    loader = DataLoader(HMDB(test, processor, 8, False), batch_size=a.batch_size,
                        shuffle=False, num_workers=a.workers,
                        pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    modes = ('dense', 'spatial', 'spatiotemporal', 'full')
    for mode in modes:
        model.mode = mode
        n = c1 = c5 = 0
        spatial, temporal, effective = [], [], []
        counts = Counter()
        t0 = time.time()
        path = output / f'{mode}_per_video.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'label', 'prediction', 'correct', 'spatial_r', 'temporal_s', 'effective_retention'])
            with torch.inference_mode():
                for x, y, paths in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=a.amp and device.type == 'cuda'):
                        logits = model(x).logits
                    if logits.shape[1] != 51:
                        raise RuntimeError('Unexpected classifier output size')
                    predictions = logits.argmax(1)
                    top5 = logits.topk(5, dim=1).indices
                    bs = y.numel()
                    n += bs
                    c1 += int((predictions == y).sum().item())
                    c5 += int((top5 == y[:, None]).any(dim=1).sum().item())
                    rr = [1.0] * bs if mode == 'dense' else model.last_r.float().cpu().tolist()
                    ss = [8.0] * bs if mode == 'dense' else model.last_s.float().cpu().tolist()
                    for i, video in enumerate(paths):
                        er = float(rr[i]) * float(ss[i]) / 8.0
                        spatial.append(float(rr[i])); temporal.append(float(ss[i])); effective.append(er)
                        counts[str(ss[i])] += 1
                        writer.writerow([video, int(y[i]), int(predictions[i]),
                                         int(predictions[i] == y[i]), rr[i], ss[i], er])
                    if n % 300 < bs or n == len(test):
                        print(f'{mode}: {n}/{len(test)} Top-1={100*c1/n:.2f}% Top-5={100*c5/n:.2f}% elapsed={time.time()-t0:.0f}s', flush=True)
        if n != len(test):
            raise RuntimeError(f'Incomplete {mode} evaluation: {n}/{len(test)}')
        results[mode] = dict(n=n, top1=100*c1/n, top5=100*c5/n,
                             spatial_retention=summarize(spatial), temporal_s=summarize(temporal),
                             effective_retention=summarize(effective), temporal_counts=dict(counts),
                             elapsed_seconds=time.time()-t0, per_video_csv=str(path))
        print(f'FINAL {mode}: Top-1={results[mode]["top1"]:.4f}% Top-5={results[mode]["top5"]:.4f}%', flush=True)
    summary = dict(experiment='H2-C HMDB51 frozen-checkpoint diagnostic',
                   protocol_warning='EXPLORATORY: inherited H1 dense backbone was selected on the official HMDB51 test split.',
                   checkpoint=str(ckpt), selected_epoch=int(obj['epoch']), split=a.split,
                   modes=results,
                   note='Dense uses the H2-B checkpoint backbone; modes are inference interventions, not separately retrained models. Gating preserves sequence shape; no physical pruning or speedup claim.',
                   args=vars(a))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('H2-C COMPLETE:', json.dumps({k: {'top1': v['top1'], 'top5': v['top5'], 'effective_retention': v['effective_retention']['mean']} for k,v in results.items()}, indent=2), flush=True)
    print('Saved:', output / 'summary.json', flush=True)

if __name__ == '__main__':
    main()
