"""E25-B: frozen E12 adaptive vs validation-selected fixed FLOPs audit.
Uses the existing E9-v2 fvcore profiler and E14 E12 checkpoint/model loader.
No training, no checkpoint selection, no test-set tuning.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import adast_e9_v2_dataset_gflops as e9
import adast_e14_budget_sweep as e14


def dump_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def profile(model, x, adaptive):
    if adaptive:
        model.force_spatial_r = None
        model.force_temporal_s = None
    else:
        model.set_forced_budget(0.4375, 7)
    return e9.fvcore_flops(model, x, is_adast=True)


def main():
    p = argparse.ArgumentParser(description='E25-B final E12 counted-FLOPs audit')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--e12-checkpoint', required=True)
    p.add_argument('--dense-checkpoint', required=True)
    p.add_argument('--output', default='runs/e25b_e12_flops')
    p.add_argument('--max-videos', type=int, default=100)
    p.add_argument('--model-name', default='facebook/timesformer-base-finetuned-k400')
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--insert-block', type=int, default=6)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--print-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.max_videos < 1:
        p.error('--max-videos must be positive')
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    processor = e14.AutoImageProcessor.from_pretrained(args.model_name)
    _, test_items = e14.e6.read_split(Path(args.data_root), Path(args.split_root), args.split)
    if len(test_items) != 3783:
        raise RuntimeError(f'Expected 3783 official test videos, found {len(test_items)}')
    items = test_items[:args.max_videos]
    dataset = e14.e6.UCF(items, processor, args.num_frames)

    print('Building dense model and loading dense checkpoint...')
    dense = e9.make_backbone(args.num_frames)
    e9.load_dense(dense, args.dense_checkpoint)
    dense.to(device).eval()
    x0, _, _ = dataset[0]
    x0 = x0.unsqueeze(0).to(device)
    dense_flops, dense_unsupported = e9.fvcore_flops(dense, x0, is_adast=False)
    print('Dense counted GFLOPs:', dense_flops / 1e9)
    del dense, x0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print('Building frozen E12 model...')
    backbone = e14.TimesformerForVideoClassification.from_pretrained(
        args.model_name, num_labels=101, ignore_mismatched_sizes=True)
    model = e14.AdaSTE14(backbone, block=args.insert_block,
                           T=args.num_frames, temperature=args.temperature)
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, args.e12_checkpoint)
    model.to(device).eval()

    csv_path = out / 'per_video_flops.csv'
    fields = ['index','video','label','adaptive_gflops','adaptive_tokens',
              'fixed_gflops','fixed_tokens','dense_gflops',
              'adaptive_reduction_pct','fixed_reduction_pct']
    existing = {}
    if csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                existing[int(r['index'])] = r
    unsupported = {'adaptive': {}, 'fixed': {}}
    for i in range(len(dataset)):
        if i in existing:
            continue
        x, y, meta = dataset[i]
        x = x.unsqueeze(0).to(device)
        a_flops, a_uns = profile(model, x, True)
        a_tokens = int(model.last_compact_tokens)
        f_flops, f_uns = profile(model, x, False)
        f_tokens = int(model.last_compact_tokens)
        if f_tokens != 602:
            raise RuntimeError(f'Fixed budget should have 602 patch tokens; got {f_tokens} at {i}')
        for key, ops in [('adaptive', a_uns), ('fixed', f_uns)]:
            for op, count in ops.items():
                unsupported[key][op] = max(unsupported[key].get(op, 0), int(count))
        row = dict(index=i, video=str(meta), label=int(y),
                   adaptive_gflops=a_flops/1e9, adaptive_tokens=a_tokens,
                   fixed_gflops=f_flops/1e9, fixed_tokens=f_tokens,
                   dense_gflops=dense_flops/1e9,
                   adaptive_reduction_pct=100*(dense_flops-a_flops)/dense_flops,
                   fixed_reduction_pct=100*(dense_flops-f_flops)/dense_flops)
        existing[i] = row
        # Atomic rewrite to permit safe resume after interruption.
        temp = csv_path.with_suffix('.tmp')
        with temp.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for j in sorted(existing):
                w.writerow(existing[j])
        temp.replace(csv_path)
        if (i+1) % args.print_every == 0 or i == 0 or i+1 == len(dataset):
            print(f'{i+1}/{len(dataset)} | adaptive {a_flops/1e9:.3f} G, '
                  f'fixed {f_flops/1e9:.3f} G | tokens {a_tokens}/{f_tokens}', flush=True)
        del x

    rows = [existing[i] for i in range(len(dataset))]
    summary = {'experiment':'E25B_final_E12_fvcore_counted_FLOPs',
               'dataset':'UCF101 official Split-1 first N test videos',
               'n':len(rows), 'checkpoint':str(args.e12_checkpoint),
               'dense_checkpoint':str(args.dense_checkpoint),
               'dense_counted_gflops':dense_flops/1e9,
               'unsupported_ops_dense':dense_unsupported,
               'unsupported_ops_seen_in_current_run':unsupported,
               'note':'fvcore-counted operations only; not wall-clock speedup; '
                      'unsupported operators excluded. Model weights frozen.'}
    for key in ['adaptive','fixed']:
        gf = np.array([float(r[f'{key}_gflops']) for r in rows])
        tk = np.array([float(r[f'{key}_tokens']) for r in rows])
        summary[key] = {'counted_gflops_mean':float(gf.mean()),
                        'counted_gflops_std':float(gf.std()),
                        'counted_gflops_median':float(np.median(gf)),
                        'mean_patch_tokens':float(tk.mean()),
                        'counted_reduction_vs_dense_pct':float(100*(1-gf.mean()/(dense_flops/1e9)))}
    dump_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Saved:', csv_path, out/'summary.json')

if __name__ == '__main__':
    main()
