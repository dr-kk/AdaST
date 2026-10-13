import argparse
import json
import random
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import adast_e14_budget_sweep as e14
from adast_e33_direct_drop import direct_drop_forward_compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--e12-checkpoint", required=True)
    parser.add_argument("--output", default="runs/e33_direct_drop")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model-name",
                        default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--insert-block", type=int, default=6)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    assert args.batch_size == 1
    assert args.num_frames == 8

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    processor = e14.AutoImageProcessor.from_pretrained(args.model_name)
    loader = e14.build_test_loader(args, processor)

    backbone = e14.TimesformerForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=101,
        ignore_mismatched_sizes=True
    )

    model = e14.AdaSTE14(
        backbone,
        block=args.insert_block,
        T=args.num_frames,
        temperature=args.temperature
    )
    model.enable_physical_tail()
    e14.load_e12_checkpoint(model, args.e12_checkpoint)
    model.to(device).eval()

    # Fixed K=86 out of 196 patches per frame.
    rho = 86.0 / 196.0
    model.set_forced_budget(rho, 8)

    consolidator = model.consolidator
    original = consolidator.forward_compact

    records = []
    verified_indices = 0

    # Capture the retained indices for each variant.
    captured = {"adast": None, "drop": None}

    def original_wrapper(self, pristine_patch, keep_mask):
        out, idx = original(pristine_patch, keep_mask)
        captured["adast"] = idx.detach().cpu().clone()
        return out, idx

    def dropping_wrapper(self, pristine_patch, keep_mask):
        out, idx = direct_drop_forward_compact(
            pristine_patch, keep_mask
        )
        captured["drop"] = idx.detach().cpu().clone()
        return out, idx

    try:
        with torch.no_grad():
            for step, batch in enumerate(loader, 1):
                if args.max_samples > 0 and step > args.max_samples:
                    break

                x, y, paths = e14.e12.unpack_batch(batch, device)
                predictions = {}

                for variant in ("adast", "drop"):
                    captured[variant] = None

                    if variant == "adast":
                        consolidator.forward_compact = MethodType(
                            original_wrapper, consolidator
                        )
                    else:
                        consolidator.forward_compact = MethodType(
                            dropping_wrapper, consolidator
                        )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=args.amp and device.type == "cuda"
                    ):
                        logits = model(x).logits

                    predictions[variant] = {
                        "pred": int(logits.argmax(1).item()),
                        "top5": logits.topk(5, dim=1).indices[0]
                                     .detach().cpu().tolist(),
                        "tokens": int(model.last_compact_tokens)
                    }

                ia = captured["adast"]
                ib = captured["drop"]

                if ia is None or ib is None:
                    raise RuntimeError("Token indices not captured")

                if not torch.equal(ia, ib):
                    raise RuntimeError(
                        f"Selection mismatch at sample {step}"
                    )

                if predictions["adast"]["tokens"] != 688:
                    raise RuntimeError("Unexpected AdaST token count")

                if predictions["drop"]["tokens"] != 688:
                    raise RuntimeError("Unexpected dropping token count")

                verified_indices += 1
                target = int(y.item())

                record = {
                    "sample": step,
                    "target": target,
                    "adast": predictions["adast"],
                    "drop": predictions["drop"],
                }
                records.append(record)

                if step % args.print_every == 0 or (
                    args.max_samples > 0 and step == args.max_samples
                ):
                    print(
                        f"{step} samples | indices matched: "
                        f"{verified_indices}",
                        flush=True
                    )
    finally:
        consolidator.forward_compact = original

    def summarize(variant):
        n = len(records)
        return {
            "samples": n,
            "top1": 100.0 * sum(
                r[variant]["pred"] == r["target"]
                for r in records
            ) / n,
            "top5": 100.0 * sum(
                r["target"] in r[variant]["top5"]
                for r in records
            ) / n,
            "mean_tokens": sum(
                r[variant]["tokens"] for r in records
            ) / n,
        }

    result = {
        "experiment": "E33",
        "precision": "FP16" if args.amp else "FP32",
        "budget": {"K": 86, "S": 8},
        "verified_matching_indices": verified_indices,
        "adast": summarize("adast"),
        "direct_drop": summarize("drop"),
        "records": records,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    suffix = "full" if args.max_samples == 0 else f"smoke_{args.max_samples}"
    path = output / f"e33_{suffix}.json"
    path.write_text(json.dumps(result, indent=2))

    print("\n=== E33 RESULTS ===")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "records"},
        indent=2
    ))
    print("Saved:", path)


if __name__ == "__main__":
    main()
