import argparse
import json
from pathlib import Path

import torch
import adast_e14_budget_sweep as e14
import adast_e31_evaluate as e31

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-root", required=True)
    p.add_argument("--e12-checkpoint", required=True)
    p.add_argument("--output", default="runs/e31_statistical_validation")
    p.add_argument("--model-name", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--insert-block", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    e14.seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    processor = e14.AutoImageProcessor.from_pretrained(args.model_name)
    loader = e14.build_test_loader(args, processor)
    print("Test samples:", len(loader.dataset), flush=True)
    assert len(loader.dataset) == 3783, "Unexpected UCF101 Split-1 test size"

    backbone = e14.TimesformerForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=101,
        ignore_mismatched_sizes=True,
    )
    model = e14.AdaSTE14(
        backbone,
        block=args.insert_block,
        T=args.num_frames,
        temperature=args.temperature,
    )
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, args.e12_checkpoint)
    model.to(device).eval()

    configs = {
        "E30-A_spatial_adaptive": (None, 8),
        "E30-B_temporal_adaptive": (0.5, None),
        "E30-C_fully_adaptive": (None, None),
        "E30-D_fully_fixed": (0.5, 8),
    }

    results = {}
    for name, (rho, temporal_s) in configs.items():
        print("\n" + "=" * 70, flush=True)
        print(name, "spatial_r=", rho, "temporal_s=", temporal_s, flush=True)

        result = e31.evaluate_point_e31(
            model, loader, device, args, rho, temporal_s
        )
        assert result["samples"] == 3783
        predictions = result.pop("predictions")

        import csv

        pred_path = out / f"{name}_predictions.csv"

        with pred_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video", "target",
                    "prediction", "correct"
                ],
            )
            writer.writeheader()
            writer.writerows(predictions)

        print(
            f"[PREDICTIONS SAVED] {pred_path} "
            f"({len(predictions)} videos)",
            flush=True,
        )

        results[name] = result

        print(
            f"Top-1={result['top1']:.4f}% "
            f"Top-5={result['top5']:.4f}% "
            f"K={result['K_per_frame']:.2f} "
            f"Tokens={result['physical_compact_patch_tokens']:.2f}",
            flush=True,
        )

        (out / "results.json").write_text(
            json.dumps(results, indent=2, default=float),
            encoding="utf-8",
        )
        print("[SAVED]", out / "results.json", flush=True)

    print("\nE31 PREDICTION COLLECTION COMPLETE", flush=True)

if __name__ == "__main__":
    main()
