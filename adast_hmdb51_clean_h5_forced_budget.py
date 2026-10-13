#!/usr/bin/env python
"""Clean H5: validation-only forced temporal-budget sweep S=6,7,8.
No training, no official-test decoding, no physical token deletion.
Run alongside the original H1/H2/E5 Python modules.
"""
import argparse, csv, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest

@torch.no_grad()
def evaluate(model, loader, device, amp, budget):
    model.eval()
    n = c1 = c5 = 0
    loss_sum = 0.0
    retention_sum = 0.0
    for step, (x, y, _) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp and device.type == 'cuda', dtype=torch.float16):
            logits = model(x).logits
            loss = F.cross_entropy(logits, y)
        n += len(y)
        c1 += int((logits.argmax(1) == y).sum())
        c5 += int((logits.topk(5, 1).indices == y[:, None]).any(1).sum())
        loss_sum += float(loss) * len(y)
        actual = model.last_s.float().cpu()
        if not torch.all(actual == budget):
            raise RuntimeError(f'Forced S={budget} did not take effect: {actual.tolist()}')
        retention_sum += float((model.last_r.float().cpu() * actual / model.T).sum())
        if step % 100 == 0 or step == len(loader):
            print(f'S={budget}: {step}/{len(loader)} top1={100*c1/n:.2f}%', flush=True)
    return dict(forced_s=budget, n=n, top1=100*c1/n, top5=100*c5/n,
                loss=loss_sum/n, effective_retention=retention_sum/n,
                physical_token_deletion=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--dense-checkpoint', default=r'runs\hmdb51_clean_h1_dense_s1\best_val.pt')
    p.add_argument('--h2-checkpoint', default=r'runs\hmdb51_clean_h2_policy_s1_v2\best_val.pt')
    p.add_argument('--manifest', default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output', default=r'runs\hmdb51_clean_h5_forced_budget_s1')
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f'Output folder not empty: {out}; choose a new folder')
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _, val, classes, digest, test_n = validate_manifest(
        manifest, manifest_path, a.split_root, a.data_root, a.split)
    dense_obj, dense_sd = checkpoint(a.dense_checkpoint)
    h2_obj, h2_sd = checkpoint(a.h2_checkpoint)
    if dense_obj.get('manifest_sha256') != digest or h2_obj.get('manifest_sha256') != digest:
        raise RuntimeError('Checkpoint/manifest hash mismatch')
    if dense_obj.get('protocol') != 'Validation-selected; official test not evaluated':
        raise RuntimeError('Unexpected H1 checkpoint protocol')
    if h2_obj.get('protocol') != 'Clean H2 validation-selected; official test not evaluated':
        raise RuntimeError('Unexpected H2 checkpoint protocol')
    for obj in (dense_obj, h2_obj):
        if obj.get('classes') != classes or obj.get('split') != a.split or obj.get('num_frames') != 8:
            raise RuntimeError('Checkpoint class order/split/frames mismatch')
    seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME, num_frames=8, num_labels=51, ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd, strict=True)
    model = AdaSTE5(bb, block=6, T=8)
    model.load_state_dict(h2_sd, strict=True)
    for key, value in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[key].cpu(), value.cpu()):
            raise RuntimeError('H2 backbone differs from clean H1: ' + key)
    model.to(device).eval()
    loader = DataLoader(HMDB(val, proc, 8, False), batch_size=a.batch_size,
                        shuffle=False, num_workers=a.workers,
                        pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0)
    out.mkdir(parents=True, exist_ok=False)
    print(f'CLEAN H5 | val={len(val)} official test={test_n} NOT EVALUATED | '
          f'manifest={digest} | device={device}', flush=True)
    # Override only the controller's output; the original E5 hook, frame novelty,
    # spatial selection, consolidation and downstream backbone remain unchanged.
    original_forward = model.temporal_budget.forward
    selected = [8]
    def forced_forward(z):
        logits = torch.full((z.shape[0], 3), -100.0, dtype=z.dtype, device=z.device)
        logits[:, selected[0] - 6] = 100.0
        return logits
    model.temporal_budget.forward = forced_forward
    results = []
    try:
        for budget in (8, 7, 6):
            selected[0] = budget
            result = evaluate(model, loader, device, a.amp, budget)
            results.append(result)
            print('RESULT:', json.dumps(result), flush=True)
    finally:
        model.temporal_budget.forward = original_forward
    summary = dict(experiment='Clean HMDB51 H5 forced temporal-budget validation sweep',
                   manifest_sha256=digest, validation_n=len(val),
                   official_test_n_not_evaluated=test_n,
                   h1_checkpoint=str(Path(a.dense_checkpoint).resolve()),
                   h2_checkpoint=str(Path(a.h2_checkpoint).resolve()),
                   results=results, notes='Forced S=8,7,6; no training; shape-preserving gating, not physical pruning',
                   args=vars(a))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)
    print('CLEAN H5 COMPLETE:', out/'summary.json', flush=True)

if __name__ == '__main__':
    main()
