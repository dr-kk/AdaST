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


def video_key(path):
    p = Path(str(path).replace("\\", "/"))
    return (p.parent.name.lower(), p.name.lower())


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def load_historical(path):
    records = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "video",
            "label",
            "prediction",
            "k_per_frame",
            "temporal_S",
            "compact_patch_tokens",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"Historical CSV missing columns: {sorted(missing)}"
            )

        for row in reader:
            key = video_key(row["video"])

            if key in records:
                raise RuntimeError(
                    f"Duplicate historical video: {key}"
                )

            records[key] = row

    return records


def checkpoint_state(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    return e6.norm(e6.state(obj))


def load_model(dense_path, checkpoint_path, device):
    backbone = e7.make_backbone(8)

    e6.load_dense(backbone, dense_path)

    model = e6.AdaSTE5(
        backbone,
        block=6,
        T=8,
    )

    model.enable_physical_tail()

    sd = checkpoint_state(checkpoint_path)

    status = model.load_state_dict(
        sd,
        strict=True,
    )

    print(
        "E12 checkpoint:",
        f"missing={len(status.missing_keys)}",
        f"unexpected={len(status.unexpected_keys)}",
    )

    model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def evaluate_video(model, x, amp=False):
    device = x.device

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        output = model(x)
        logits = output.logits

    logits = logits.float().cpu()

    return {
        "prediction": int(logits.argmax(1).item()),
        "top5": logits.topk(5, dim=1).indices[0].tolist(),
        "k": as_float(model.last_k),
        "s": as_float(model.last_s),
        "tokens": as_float(model.last_compact_tokens),
    }


def close(a, b, atol=1e-5):
    return abs(float(a) - float(b)) <= atol


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--e12-checkpoint", required=True)
    parser.add_argument("--historical-csv", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output",
        default="runs/e19_historical_audit",
    )

    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    historical = load_historical(args.historical_csv)

    _, test_items = e6.read_split(
        Path(args.data_root),
        Path(args.split_root),
        1,
    )

    if len(test_items) != 3783:
        raise RuntimeError(
            f"Expected 3783 test videos, found {len(test_items)}"
        )

    processor = AutoImageProcessor.from_pretrained(
        e7.MODEL_NAME
    )

    dataset = e6.UCF(
        test_items,
        processor,
        8,
    )

    # Confirm historical and current video identities match
    # across the complete official test split.
    current_keys = set()

    for path, label in test_items:
        key = video_key(path)

        if key in current_keys:
            raise RuntimeError(
                f"Duplicate current video identity: {key}"
            )

        current_keys.add(key)

        if key not in historical:
            raise RuntimeError(
                f"Missing historical video: {key}"
            )

        if int(historical[key]["label"]) != int(label):
            raise RuntimeError(
                f"Historical label mismatch: {key}"
            )

    historical_keys = set(historical)

    if historical_keys != current_keys:
        raise RuntimeError(
            "Historical CSV does not exactly match "
            "the official current test split. "
            f"Historical={len(historical_keys)}, "
            f"current={len(current_keys)}"
        )

    print("Official test split:", len(dataset))
    print("Historical records:", len(historical))
    print("Video identities and labels: MATCH")
    print("Device:", device)
    print("AMP:", args.amp)

    model = load_model(
        args.dense_checkpoint,
        args.e12_checkpoint,
        device,
    )

    count = min(args.samples, len(dataset))

    # Deterministic coverage across the complete test split.
    indices = np.linspace(
        0,
        len(dataset) - 1,
        count,
        dtype=int,
    ).tolist()

    rows = []

    for number, index in enumerate(indices, 1):
        x, label, path = dataset[index]

        key = video_key(path)
        old = historical[key]

        x = x.unsqueeze(0).to(device)

        fresh = evaluate_video(
            model,
            x,
            amp=args.amp,
        )

        old_pred = int(old["prediction"])
        old_k = float(old["k_per_frame"])
        old_s = float(old["temporal_S"])
        old_tokens = float(old["compact_patch_tokens"])

        row = {
            "index": index,
            "video": str(path),
            "label": int(label),
            "historical_prediction": old_pred,
            "fresh_prediction": fresh["prediction"],
            "prediction_match": (
                old_pred == fresh["prediction"]
            ),
            "historical_k": old_k,
            "fresh_k": fresh["k"],
            "k_match": close(old_k, fresh["k"]),
            "historical_s": old_s,
            "fresh_s": fresh["s"],
            "s_match": close(old_s, fresh["s"]),
            "historical_tokens": old_tokens,
            "fresh_tokens": fresh["tokens"],
            "tokens_match": close(
                old_tokens,
                fresh["tokens"],
            ),
            "historical_correct": (
                old_pred == int(label)
            ),
            "fresh_correct": (
                fresh["prediction"] == int(label)
            ),
            "fresh_top5_correct": (
                int(label) in fresh["top5"]
            ),
        }

        rows.append(row)

        if number % 10 == 0 or number == count:
            print(
                f"[{number}/{count}] "
                f"Pred={row['prediction_match']} "
                f"K={row['k_match']} "
                f"S={row['s_match']} "
                f"Tokens={row['tokens_match']}"
            )

    csv_path = outdir / "per_video_audit.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    def mismatches(field):
        return sum(not r[field] for r in rows)

    n = len(rows)

    summary = {
        "experiment": "E19-D historical E12 audit",
        "samples": n,
        "official_test_size": len(dataset),
        "historical_records": len(historical),
        "identity_and_label_check": "PASS",
        "amp": args.amp,
        "prediction_mismatches": mismatches(
            "prediction_match"
        ),
        "k_mismatches": mismatches("k_match"),
        "s_mismatches": mismatches("s_match"),
        "token_mismatches": mismatches(
            "tokens_match"
        ),
        "historical_subset_top1_percent": (
            100 * sum(
                r["historical_correct"] for r in rows
            ) / n
        ),
        "fresh_subset_top1_percent": (
            100 * sum(
                r["fresh_correct"] for r in rows
            ) / n
        ),
        "fresh_subset_top5_percent": (
            100 * sum(
                r["fresh_top5_correct"] for r in rows
            ) / n
        ),
        "note": (
            "Subset results are diagnostic only. "
            "They are not full UCF101 test-set metrics."
        ),
    }

    summary_path = outdir / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("E19-D HISTORICAL E12 AUDIT")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print("Saved:", csv_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
