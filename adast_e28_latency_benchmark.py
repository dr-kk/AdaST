import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import TimesformerForVideoClassification

import adast_e14_budget_sweep as e14
import adast_e7_v3_final_benchmark as e7


def build_dense(args, device):
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_name,
        num_frames=8,
        num_labels=101,
        ignore_mismatched_sizes=True,
    )
    e7.load_dense(model, args.dense_checkpoint)
    return model.to(device).eval()


def build_adast(args, device):
    backbone = TimesformerForVideoClassification.from_pretrained(
        args.model_name,
        num_frames=8,
        num_labels=101,
        ignore_mismatched_sizes=True,
    )
    model = e14.AdaSTE14(
        backbone,
        block=6,
        T=8,
        temperature=1.0,
    )
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, args.e12_checkpoint)
    return model.to(device).eval()


def set_mode(model, mode):
    if mode == "adaptive":
        model.force_spatial_r = None
        model.force_temporal_s = None
    elif mode == "fixed602":
        model.set_forced_budget(0.4375, 7)
    elif mode == "fixed688":
        model.set_forced_budget(0.4375, 8)


@torch.inference_mode()
def measure(model, xs, device, warmup, runs):
    for i in range(warmup):
        model(xs[i % len(xs)])
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(runs):
        x = xs[i % len(xs)]
        start.record()
        model(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-root", required=True)
    p.add_argument("--dense-checkpoint", required=True)
    p.add_argument("--e12-checkpoint", required=True)
    p.add_argument(
        "--model-name",
        default="facebook/timesformer-base-finetuned-k400"
    )
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", default="runs/e28_latency")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("E28 requires CUDA.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    from transformers import AutoImageProcessor
    from torch.utils.data import DataLoader

    _, test_items = e7.read_split(
        args.data_root, args.split_root, 1
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model_name
    )
    ds = e7.UCF(test_items, processor, 8)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    xs = e7.cache_inputs(loader, args.runs, device)
    print("Cached videos:", len(xs))

    dense = build_dense(args, device)
    adaptive = build_adast(args, device)

    modes = ["dense", "adaptive", "fixed602", "fixed688"]
    results = {mode: [] for mode in modes}

    for trial in range(args.trials):
        order = modes[trial % 4:] + modes[:trial % 4]
        print("\nTrial", trial + 1, "order:", order)

        for mode in order:
            model = dense if mode == "dense" else adaptive

            if mode != "dense":
                set_mode(model, mode)

            times = measure(
                model, xs, device,
                args.warmup, args.runs
            )
            results[mode].extend(times)

            print(
                mode,
                "median:",
                round(float(np.median(times)), 3),
                "ms"
            )

    summary = {}

    for mode, times in results.items():
        a = np.asarray(times)
        summary[mode] = {
            "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "fps": float(1000 / a.mean()),
            "samples": len(a),
        }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nFINAL E28 RESULTS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
