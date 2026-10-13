import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor

import adast_e6_v2_pretrained_compact as e6
import adast_e7_v3_final_benchmark as e7
from adast_e18_component_ablation import AdaSTE18


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def build_model(backbone, cls, variant=None):
    if variant is None:
        return cls(backbone, block=6, T=8)

    return cls(
        backbone,
        block=6,
        T=8,
        variant=variant,
    )


def load_model(model, dense_path, e12_path, device):
    e6.load_dense(model.backbone, dense_path)
    model.enable_physical_tail()

    obj = torch.load(
        e12_path,
        map_location="cpu",
        weights_only=False,
    )
    sd = e6.norm(e6.state(obj))

    status = model.load_state_dict(sd, strict=True)

    print(
        "Checkpoint:",
        "missing=", len(status.missing_keys),
        "unexpected=", len(status.unexpected_keys),
    )

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def run_one(model, x):
    logits = get_logits(model(x)).float().cpu()

    return {
        "logits": logits,
        "prediction": int(logits.argmax(1).item()),
        "k": scalar(model.last_k),
        "s": scalar(model.last_s),
        "tokens": scalar(model.last_compact_tokens),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-root", required=True)
    p.add_argument("--dense-checkpoint", required=True)
    p.add_argument("--e12-checkpoint", required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument(
        "--output",
        default="runs/e19_paired_parity",
    )
    args = p.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be positive")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Precision: FP32")
    print("Samples requested:", args.samples)

    _, test_items_e6 = e6.read_split(
        Path(args.data_root),
        Path(args.split_root),
        split=1,
    )

    _, test_items_e7 = e7.read_split(
        Path(args.data_root),
        Path(args.split_root),
        split=1,
    )

    def identities(items):
        return [
            (Path(str(path)).parent.name.casefold(),
             Path(str(path)).name.casefold(),
             int(label))
            for path, label in items
        ]

    if identities(test_items_e6) != identities(test_items_e7):
        raise RuntimeError(
            "E6/E7 official test lists are not identical."
        )

    if len(test_items_e6) != 3783:
        raise RuntimeError(
            f"Expected 3783 videos; got {len(test_items_e6)}"
        )

    processor = AutoImageProcessor.from_pretrained(
        e7.MODEL_NAME
    )

    dataset = e6.UCF(
        test_items_e6,
        processor,
        8,
    )

    # Both models use the same backbone constructor to isolate
    # differences caused by the original vs overridden hooks.
    original = build_model(
        e7.make_backbone(8),
        e6.AdaSTE5,
    )

    original = load_model(
        original,
        args.dense_checkpoint,
        args.e12_checkpoint,
        device,
    )

    modified = build_model(
        e7.make_backbone(8),
        AdaSTE18,
        variant="full",
    )

    modified = load_model(
        modified,
        args.dense_checkpoint,
        args.e12_checkpoint,
        device,
    )

    n = min(args.samples, len(dataset))
    indices = np.linspace(
        0,
        len(dataset) - 1,
        n,
        dtype=int,
    ).tolist()

    rows = []

    for count, index in enumerate(indices, 1):
        x, label, path = dataset[index]

        # Both models receive the exact same GPU tensor.
        x = x.unsqueeze(0).to(device)

        a = run_one(original, x)
        b = run_one(modified, x)

        max_diff = float(
            (a["logits"] - b["logits"])
            .abs()
            .max()
            .item()
        )

        row = {
            "index": index,
            "video": str(path),
            "label": int(label),
            "original_prediction": a["prediction"],
            "e18_prediction": b["prediction"],
            "prediction_equal":
                a["prediction"] == b["prediction"],
            "original_k": a["k"],
            "e18_k": b["k"],
            "k_equal": a["k"] == b["k"],
            "original_s": a["s"],
            "e18_s": b["s"],
            "s_equal": a["s"] == b["s"],
            "original_tokens": a["tokens"],
            "e18_tokens": b["tokens"],
            "tokens_equal":
                a["tokens"] == b["tokens"],
            "max_abs_logit_diff": max_diff,
        }

        rows.append(row)

        if count % 10 == 0 or count == n:
            print(
                f"[{count}/{n}] "
                f"prediction_equal={row['prediction_equal']} "
                f"K_equal={row['k_equal']} "
                f"S_equal={row['s_equal']} "
                f"tokens_equal={row['tokens_equal']} "
                f"max_logit_diff={max_diff:.8g}"
            )

    csv_path = outdir / "per_video_parity.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "E19-C paired model parity",
        "samples": len(rows),
        "official_test_size": len(dataset),
        "same_input_tensor": True,
        "same_checkpoint": True,
        "precision": "FP32",
        "prediction_mismatches": sum(
            not r["prediction_equal"] for r in rows
        ),
        "k_mismatches": sum(
            not r["k_equal"] for r in rows
        ),
        "s_mismatches": sum(
            not r["s_equal"] for r in rows
        ),
        "token_mismatches": sum(
            not r["tokens_equal"] for r in rows
        ),
        "max_abs_logit_diff": max(
            r["max_abs_logit_diff"] for r in rows
        ),
    }

    json_path = outdir / "summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("E19-C PAIRED PARITY RESULTS")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
