import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import adast_e6_v2_pretrained_compact as e6


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise RuntimeError("Unsupported checkpoint")


def normalize_state_dict(sd):
    out = {}
    for k, v in sd.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_e12(model, path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"E12 checkpoint not found:\n{path}")

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    sd = normalize_state_dict(
        extract_state(ckpt)
    )

    z = model.load_state_dict(
        sd,
        strict=True,
    )

    print("[OK] E12 checkpoint loaded:")
    print("    ", path)
    print(
        f"     missing={len(z.missing_keys)}, "
        f"unexpected={len(z.unexpected_keys)}"
    )


class AdaSTE15HardDrop(e6.AdaSTE5):
    """
    E12 inference ablation.

    SAME:
      spatial controller
      temporal controller
      importance ranking
      novelty selection
      K(X), S(X)
      joint keep mask
      compact tail
      classifier

    CHANGED:
      representative consolidation -> hard token dropping
    """

    def _hook(self, module, args, kwargs):

        if args:
            h = args[0]
            rest = args[1:]
            src = "args"
        else:
            h = kwargs["hidden_states"]
            rest = ()
            src = "kwargs"

        cls = h[:, :1]
        patch = h[:, 1:]

        b, m, d = patch.shape

        if m % self.T:
            raise RuntimeError(
                f"{m} patch tokens not divisible by T={self.T}"
            )

        n = m // self.T

        # =====================================================
        # Spatial policy — identical to E12
        # =====================================================

        pristine_patch = patch

        pristine_z = pristine_patch.reshape(
            b,
            self.T,
            n,
            d,
        )

        scores, delta = self.scorer(
            pristine_patch,
            self.T,
        )

        logits = self.budget(
            scores,
            delta,
            cls,
        )

        idx = logits.argmax(1)

        choice = F.one_hot(
            idx,
            num_classes=len(e6.LEVELS),
        ).to(logits.dtype)

        probs = torch.softmax(
            logits,
            dim=1,
        )

        r = (
            choice
            * self.levels[None, :]
        ).sum(1)

        k = torch.round(
            r * n
        ).long().clamp(
            1,
            n,
        )

        sf = scores.reshape(
            b,
            self.T,
            n,
        )

        spatial_mask = torch.zeros_like(
            sf
        )

        for bi in range(b):

            ids = torch.topk(
                sf[bi],
                int(k[bi]),
                dim=1,
                largest=True,
                sorted=False,
            ).indices

            spatial_mask[bi].scatter_(
                1,
                ids,
                1.0,
            )

        spatial_hard = spatial_mask.reshape(
            b,
            m,
        )

        # =====================================================
        # Temporal policy — identical to E12
        # =====================================================

        t_logits = self.temporal_budget(
            pristine_z
        )

        t_probs = torch.softmax(
            t_logits,
            dim=1,
        )

        t_idx_eval = t_logits.argmax(1)

        t_choice = F.one_hot(
            t_idx_eval,
            num_classes=len(self.temporal_levels),
        ).to(t_logits.dtype)

        s_budget = (
            t_choice
            * self.temporal_levels[None, :]
        ).sum(1)

        t_idx = t_choice.argmax(1)

        expected_s = (
            t_probs
            * self.temporal_levels[None, :]
        ).sum(1)

        frame_feat = pristine_z.mean(2)

        novelty = torch.zeros(
            (b, self.T),
            device=pristine_z.device,
            dtype=pristine_z.dtype,
        )

        novelty[:, 0] = (
            frame_feat[:, 0]
            .pow(2)
            .mean(-1)
            .sqrt()
        )

        if self.T > 1:

            novelty[:, 1:] = (
                (
                    frame_feat[:, 1:]
                    - frame_feat[:, :-1]
                )
                .pow(2)
                .mean(-1)
                .sqrt()
            )

        frame_mask = torch.zeros_like(
            novelty
        )

        for bi in range(b):

            si = int(
                s_budget[bi].item()
            )

            ids = torch.topk(
                novelty[bi],
                si,
                largest=True,
                sorted=False,
            ).indices

            frame_mask[bi].scatter_(
                0,
                ids,
                1.0,
            )

        temporal_gate = (
            frame_mask[:, :, None]
            .expand(-1, -1, n)
            .reshape(b, m)
        )

        joint_keep = (
            spatial_hard
            * temporal_gate
        )

        # =====================================================
        # E15-A ABLATION
        # =====================================================

        if not self.physical_mode:
            raise RuntimeError(
                "E15-A requires physical_mode=True"
            )

        if b != 1:
            raise RuntimeError(
                "E15-A requires batch_size=1"
            )

        keep_idx = torch.nonzero(
            joint_keep[0] > 0.5,
            as_tuple=False,
        ).flatten()

        if keep_idx.numel() == 0:
            raise RuntimeError(
                "Empty keep set"
            )

        # IMPORTANT:
        # No consolidator.forward_compact().
        # Selected tokens are packed directly.
        compact_patch = pristine_patch[
            :,
            keep_idx,
            :
        ]

        self.last_compact_tokens = int(
            compact_patch.size(1)
        )

        gh = torch.cat(
            [
                cls,
                compact_patch,
            ],
            dim=1,
        )

        # =====================================================
        # Diagnostics
        # =====================================================

        self.last_probs = probs

        self.last_r = r.detach()

        self.last_expected_r = (
            probs
            * self.levels[None, :]
        ).sum(1)

        self.last_actual_r = (
            spatial_hard
            .mean(1)
            .detach()
        )

        self.last_k = k.detach()
        self.last_idx = idx.detach()

        self.last_t_probs = t_probs
        self.last_s = s_budget.detach()
        self.last_expected_s = expected_s
        self.last_t_idx = t_idx.detach()

        self.last_merge_gate = None
        self.last_merge_entropy = None

        if src == "args":
            return (
                (gh,) + rest,
                kwargs,
            )

        kw = dict(kwargs)
        kw["hidden_states"] = gh

        return args, kw


def unpack_batch(batch, device):

    if isinstance(batch, dict):

        x = batch.get(
            "pixel_values",
            batch.get("video"),
        )

        y = batch.get(
            "labels",
            batch.get("label"),
        )

        path = batch.get(
            "path",
            batch.get("video_path"),
        )

    else:

        x = batch[0]
        y = batch[1]

        path = (
            batch[2]
            if len(batch) > 2
            else None
        )

    x = x.to(
        device,
        non_blocking=True,
    )

    y = torch.as_tensor(
        y,
        dtype=torch.long,
        device=device,
    )

    return x, y, path


def build_test_loader(args, processor):

    _, test_items = e6.read_split(
        Path(args.data_root),
        Path(args.split_root),
        args.split,
    )

    dataset = e6.UCF(
        test_items,
        processor,
        args.num_frames,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[OK] Official UCF101 Split-{args.split}: "
        f"{len(dataset)} test videos"
    )

    return loader


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    args,
):

    model.eval()

    total = 0
    correct1 = 0
    correct5 = 0

    rs = []
    actual_rs = []
    ks = []
    ss = []
    effs = []
    tokens = []

    rows = []

    start = time.time()

    for step, batch in enumerate(
        loader,
        1,
    ):

        x, y, paths = unpack_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                args.amp
                and device.type == "cuda"
            ),
        ):

            out = model(x)

            logits = out.logits

        pred = logits.argmax(1)

        total += y.numel()

        correct1 += (
            pred.eq(y)
            .sum()
            .item()
        )

        top5 = logits.topk(
            min(
                5,
                logits.shape[-1],
            ),
            dim=1,
        ).indices

        correct5 += (
            top5.eq(
                y[:, None]
            )
            .any(1)
            .sum()
            .item()
        )

        r = float(
            model.last_r.item()
        )

        actual_r = float(
            model.last_actual_r.item()
        )

        k = float(
            model.last_k.item()
        )

        s = float(
            model.last_s.item()
        )

        compact = float(
            model.last_compact_tokens
        )

        eff = (
            r
            * s
            / args.num_frames
        )

        rs.append(r)
        actual_rs.append(actual_r)
        ks.append(k)
        ss.append(s)
        effs.append(eff)
        tokens.append(compact)

        video = ""

        if paths is not None:

            if isinstance(
                paths,
                (list, tuple),
            ):
                video = str(
                    paths[0]
                )
            else:
                video = str(
                    paths
                )

        rows.append(
            {
                "video":
                    video,

                "label":
                    int(
                        y[0].item()
                    ),

                "prediction":
                    int(
                        pred[0].item()
                    ),

                "correct":
                    int(
                        pred[0].item()
                        ==
                        y[0].item()
                    ),

                "spatial_r":
                    r,

                "actual_spatial_retention":
                    actual_r,

                "k_per_frame":
                    k,

                "temporal_S":
                    s,

                "effective_retention":
                    eff,

                "compact_patch_tokens":
                    compact,
            }
        )

        if (
            step % args.print_every == 0
            or
            step == len(loader)
        ):

            print(
                f"{step:5d}/"
                f"{len(loader)} | "
                f"Top1="
                f"{100*correct1/total:.2f}% | "
                f"Top5="
                f"{100*correct5/total:.2f}% | "
                f"Tokens="
                f"{np.mean(tokens):.2f}"
            )

    elapsed = (
        time.time()
        - start
    )

    mean_eff = float(
        np.mean(effs)
    )

    mean_tokens = float(
        np.mean(tokens)
    )

    dense_tokens = float(
        args.num_frames
        * args.patches_per_frame
    )

    return {

        "samples":
            total,

        "top1":
            100.0
            * correct1
            / total,

        "top5":
            100.0
            * correct5
            / total,

        "mean_spatial_retention":
            float(
                np.mean(rs)
            ),

        "mean_actual_spatial_retention":
            float(
                np.mean(actual_rs)
            ),

        "mean_spatial_k":
            float(
                np.mean(ks)
            ),

        "mean_temporal_S":
            float(
                np.mean(ss)
            ),

        "mean_effective_retention":
            mean_eff,

        "mean_effective_reduction_pct":
            100.0
            * (
                1.0
                - mean_eff
            ),

        "mean_physical_compact_patch_tokens":
            mean_tokens,

        "mean_physical_token_reduction_pct":
            100.0
            * (
                1.0
                -
                mean_tokens
                / dense_tokens
            ),

        "seconds":
            elapsed,

    }, rows


def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=r"D:\ST-EdgeViT\datasets\UCF101",
    )

    parser.add_argument(
        "--split-root",
        default=r"D:\AdaST\splits\ucfTrainTestlist",
    )

    parser.add_argument(
        "--e12-checkpoint",
        default=r"D:\AdaST\runs\e12_rep_preserve\best.pt",
    )

    parser.add_argument(
        "--output",
        default=r"D:\AdaST\runs\e15a_hard_drop",
    )

    parser.add_argument(
        "--model-name",
        default=(
            "facebook/"
            "timesformer-base-"
            "finetuned-k400"
        ),
    )

    parser.add_argument(
        "--split",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--insert-block",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--patches-per-frame",
        type=int,
        default=196,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    args = parser.parse_args()

    for name, path in (

        (
            "Dataset",
            args.data_root,
        ),

        (
            "Split root",
            args.split_root,
        ),

        (
            "E12 checkpoint",
            args.e12_checkpoint,
        ),

    ):

        if not Path(path).exists():

            raise FileNotFoundError(
                f"{name} not found:\n"
                f"{path}"
            )

    seed_all(
        args.seed
    )

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 90
    )

    print(
        "AdaST E15-A | "
        "CONSOLIDATION vs HARD DROP"
    )

    print(
        "=" * 90
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "E12 checkpoint:",
        args.e12_checkpoint,
    )

    print(
        "No retraining."
    )

    print(
        "Same K(X), S(X), selected tokens, "
        "compact tail and classifier."
    )

    print(
        "ONLY change: consolidation -> hard drop."
    )

    print(
        "=" * 90
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            args.model_name
        )
    )

    loader = build_test_loader(
        args,
        processor,
    )

    backbone = (
        TimesformerForVideoClassification
        .from_pretrained(
            args.model_name,
            num_labels=101,
            ignore_mismatched_sizes=True,
        )
    )

    model = AdaSTE15HardDrop(
        backbone,
        block=args.insert_block,
        T=args.num_frames,
        temperature=args.temperature,
    )

    model.enable_physical_tail()

    load_e12(
        model,
        args.e12_checkpoint,
    )

    for p in model.parameters():
        p.requires_grad = False

    model.to(
        device
    )

    model.eval()

    print()

    print(
        "OFFICIAL UCF101 SPLIT-1 TEST"
    )

    result, rows = evaluate(
        model,
        loader,
        device,
        args,
    )

    E12_TOP1 = 90.8538
    E12_TOP5 = 98.7047

    summary = {

        "experiment":
            "E15A_hard_drop_no_consolidation",

        "retraining":
            False,

        "same_e12_policy":
            True,

        "consolidation":
            False,

        "hard_drop":
            True,

        "test":
            result,

        "e12_reference": {

            "top1":
                E12_TOP1,

            "top5":
                E12_TOP5,

            "mean_physical_compact_tokens":
                742.28,
        },

        "delta_top1_vs_e12_pp":
            result["top1"]
            - E12_TOP1,

        "delta_top5_vs_e12_pp":
            result["top5"]
            - E12_TOP5,
    }

    with open(
        output / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    save_csv(
        output / "result.csv",
        [result],
    )

    save_csv(
        output / "per_video_test.csv",
        rows,
    )

    print()

    print(
        "=" * 90
    )

    print(
        "E15-A FINAL RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"Samples                      : "
        f"{result['samples']}"
    )

    print(
        f"Top-1                        : "
        f"{result['top1']:.4f}%"
    )

    print(
        f"Top-5                        : "
        f"{result['top5']:.4f}%"
    )

    print(
        f"Mean spatial retention       : "
        f"{result['mean_spatial_retention']:.6f}"
    )

    print(
        f"Mean spatial K/frame         : "
        f"{result['mean_spatial_k']:.4f}"
    )

    print(
        f"Mean temporal S              : "
        f"{result['mean_temporal_S']:.4f}"
    )

    print(
        f"Mean effective reduction     : "
        f"{result['mean_effective_reduction_pct']:.4f}%"
    )

    print(
        f"Mean physical compact tokens : "
        f"{result['mean_physical_compact_patch_tokens']:.2f}"
    )

    print(
        f"Physical token reduction     : "
        f"{result['mean_physical_token_reduction_pct']:.4f}%"
    )

    print(
        "-" * 90
    )

    print(
        "E12 Full AdaST reference     : "
        "90.8538 / 98.7047"
    )

    print(
        f"Delta Top-1                  : "
        f"{summary['delta_top1_vs_e12_pp']:+.4f} pp"
    )

    print(
        f"Delta Top-5                  : "
        f"{summary['delta_top5_vs_e12_pp']:+.4f} pp"
    )

    print(
        "=" * 90
    )

    print(
        "Saved:",
        output / "summary.json",
    )

    print(
        "Saved:",
        output / "result.csv",
    )

    print(
        "Saved:",
        output / "per_video_test.csv",
    )


if __name__ == "__main__":
    main()