
#!/usr/bin/env python3
"""Frozen official HMDB51 split-1 evaluation of clean H1 and H2 checkpoints.

No training, checkpoint modification, validation evaluation, or sparse execution.
Each model is evaluated once and saved to a separate, new output directory.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, read_hmdb_split

EXPECTED_MANIFEST_SHA256 = (
    "8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec"
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint is not a dictionary")
    return obj


def verify_protocol(args, manifest):
    canonical = hashlib.sha256(
        json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if canonical != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Manifest SHA256 mismatch; STOP")

    train = [(str(p), int(y)) for p, y in manifest["train"]]
    val = [(str(p), int(y)) for p, y in manifest["val"]]

    official_train, official_test, classes = read_hmdb_split(
        args.data_root, args.split_root, 1
    )

    def key(item):
        return (str(Path(item[0]).resolve()).casefold(), int(item[1]))

    train_set = set(map(key, train))
    val_set = set(map(key, val))
    official_train_set = set(map(key, official_train))
    test_set = set(map(key, official_test))

    if (
        len(train) != 3213
        or len(val) != 357
        or len(official_train) != 3570
        or len(official_test) != 1530
        or len(train_set) != 3213
        or len(val_set) != 357
        or len(official_train_set) != 3570
        or len(test_set) != 1530
        or train_set & val_set
        or train_set | val_set != official_train_set
        or official_train_set & test_set
        or manifest["classes"] != classes
        or manifest["split"] != 1
    ):
        raise RuntimeError("Official split or manifest verification failed; STOP")

    for path, _ in official_test:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing official test video: {path}")

    return official_test, classes, canonical


def build_model(experiment, checkpoint, classes):
    obj = load_checkpoint(checkpoint)

    if (
        obj.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or obj.get("classes") != classes
        or obj.get("split") != 1
        or obj.get("num_frames") != 8
    ):
        raise RuntimeError(f"{experiment} checkpoint metadata mismatch; STOP")

    backbone = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_frames=8,
        num_labels=51,
        ignore_mismatched_sizes=True,
    )

    if experiment == "h1":
        if obj.get("protocol") != (
            "Validation-selected; official test not evaluated"
        ):
            raise RuntimeError("Not a clean H1 checkpoint; STOP")

        if not isinstance(obj.get("model"), dict):
            raise RuntimeError("H1 checkpoint has no 'model' state dictionary")

        backbone.load_state_dict(obj["model"], strict=True)
        return backbone

    if obj.get("protocol") != (
        "Clean H2 validation-selected; official test not evaluated"
    ):
        raise RuntimeError("Not a clean H2 checkpoint; STOP")

    if not isinstance(obj.get("model_state_dict"), dict):
        raise RuntimeError("H2 checkpoint has no 'model_state_dict'")

    model = AdaSTE5(backbone, block=6, T=8)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("h1", "h2"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument(
        "--manifest",
        default=r"runs\hmdb51_clean_h1_dense_s1\split_manifest.json",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.progress_every < 1:
        parser.error("--progress-every must be positive")

    output = Path(args.output).resolve()
    if output.exists():
        parser.error(
            "Output directory already exists. Use a NEW directory; STOP."
        )

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    official_test, classes, manifest_hash = verify_protocol(args, manifest)

    # Complete checkpoint and architecture checks BEFORE opening test videos.
    model = build_model(args.experiment, checkpoint, classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    loader = DataLoader(
        HMDB(official_test, processor, 8, False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint_hash = sha256_file(checkpoint)

    output.mkdir(parents=True, exist_ok=False)
    fields = [
        "index",
        "label",
        "prediction",
        "correct_top1",
        "correct_top5",
    ]

    n = correct1 = correct5 = 0
    start = time.perf_counter()

    with (output / "per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with torch.inference_mode():
            for index, (x, y, _) in enumerate(loader):
                x = x.to(device)

                if args.experiment == "h1":
                    logits = model(pixel_values=x).logits
                else:
                    logits = model(x).logits

                logits = logits.detach().float()

                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"Nonfinite logits at official test index {index}"
                    )

                target = int(y.item())
                prediction = int(logits.argmax(dim=1).item())
                hit1 = int(prediction == target)
                hit5 = int(
                    target in logits.topk(5, dim=1).indices[0].tolist()
                )

                writer.writerow(
                    {
                        "index": index,
                        "label": target,
                        "prediction": prediction,
                        "correct_top1": hit1,
                        "correct_top5": hit5,
                    }
                )
                f.flush()

                n += 1
                correct1 += hit1
                correct5 += hit5

                if n % args.progress_every == 0 or n == 1530:
                    print(
                        f"{args.experiment.upper()} PROGRESS: {n}/1530 "
                        f"Top-1={100 * correct1 / n:.2f}% "
                        f"Top-5={100 * correct5 / n:.2f}%",
                        flush=True,
                    )

    if n != 1530:
        raise RuntimeError(f"Incomplete official test: {n}/1530; STOP")

    result = {
        "experiment": args.experiment.upper(),
        "dataset": "HMDB51",
        "split": 1,
        "evaluation": "Frozen validation-selected checkpoint; official test",
        "status": "PASS",
        "n": n,
        "top1_percent": 100 * correct1 / n,
        "top5_percent": 100 * correct5 / n,
        "top1_correct": correct1,
        "top5_correct": correct5,
        "manifest_canonical_sha256": manifest_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_loaded_strictly": True,
        "device": str(device),
        "amp": False,
        "training_performed": False,
        "validation_evaluated": False,
        "official_test_evaluated": True,
        "sparse_path_used": False,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "limitations": [
            "One frozen checkpoint evaluated on HMDB51 official split-1.",
            "No test-set tuning or checkpoint reselection is permitted.",
            "No physical token deletion, FLOP reduction, or speedup established.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(
        f"{args.experiment.upper()} OFFICIAL TEST RESULT:",
        json.dumps(result, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()