#!/usr/bin/env python
"""
AdaST E14
Controlled Spatio-Temporal Budget Sweep
with automatic resume and schema-safe saving.
"""

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
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
)

import adast_e6_v2_pretrained_compact as e6
import adast_e12_rep_preserve as e12


# ============================================================
# OPERATING POINTS
# ============================================================

SPATIAL_LEVELS = [
    0.375,
    0.4375,
    0.5000,
    0.5625,
    0.6250,
]

TEMPORAL_LEVELS = [
    6,
    7,
    8,
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# E14 MODEL
# ============================================================

class AdaSTE14(e6.AdaSTE5):

    def __init__(
        self,
        backbone,
        block=6,
        T=8,
        temperature=1.0,
    ):

        super().__init__(
            backbone,
            block=block,
            T=T,
            temperature=temperature,
        )

        self.force_spatial_r = None
        self.force_temporal_s = None


    def set_forced_budget(
        self,
        rho,
        temporal_s,
    ):

        self.force_spatial_r = float(rho)
        self.force_temporal_s = int(temporal_s)


    def _hook(
        self,
        module,
        args,
        kwargs,
    ):

        h = args[0]

        cls = h[:, :1]
        patch = h[:, 1:]

        b, m, d = patch.shape

        if m % self.T != 0:

            raise RuntimeError(
                f"{m} patch tokens are not divisible "
                f"by T={self.T}"
            )

        n = m // self.T

        pristine_patch = patch

        pristine_z = pristine_patch.reshape(
            b,
            self.T,
            n,
            d,
        )

        # ====================================================
        # LEARNED SPATIAL IMPORTANCE
        # ====================================================

        scores, delta = self.scorer(
            pristine_patch,
            self.T,
        )

        logits = self.budget(
            scores,
            delta,
            cls,
        )

        spatial_probs = torch.softmax(
            logits,
            dim=1,
        )

        learned_spatial_idx = logits.argmax(
            dim=1,
        )

        # ====================================================
        # FIXED SPATIAL BUDGET
        # ====================================================

        if self.force_spatial_r is None:

            choice = F.one_hot(
                learned_spatial_idx,
                num_classes=len(self.levels),
            ).to(
                logits.dtype
            )

            r = (
                choice
                * self.levels[None, :]
            ).sum(
                dim=1
            )

        else:

            r = torch.full(
                (b,),
                self.force_spatial_r,
                device=patch.device,
                dtype=patch.dtype,
            )

        k = torch.round(
            r * n
        ).long()

        k = k.clamp(
            min=1,
            max=n,
        )

        # ====================================================
        # SPATIAL RANKING
        # ====================================================

        spatial_scores = scores.reshape(
            b,
            self.T,
            n,
        )

        spatial_mask = torch.zeros_like(
            spatial_scores
        )

        for bi in range(b):

            ki = int(
                k[bi].item()
            )

            ids = torch.topk(
                spatial_scores[bi],
                ki,
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

        actual_r = spatial_hard.mean(
            dim=1
        )

        # ====================================================
        # TEMPORAL CONTROLLER
        # ====================================================

        temporal_logits = self.temporal_budget(
            pristine_z
        )

        temporal_probs = torch.softmax(
            temporal_logits,
            dim=1,
        )

        learned_temporal_idx = temporal_logits.argmax(
            dim=1
        )

        expected_s = (
            temporal_probs
            * self.temporal_levels[None, :]
        ).sum(
            dim=1
        )

        # ====================================================
        # FIXED TEMPORAL BUDGET
        # ====================================================

        if self.force_temporal_s is None:

            temporal_choice = F.one_hot(
                learned_temporal_idx,
                num_classes=len(
                    self.temporal_levels
                ),
            ).to(
                temporal_logits.dtype
            )

            s_budget = (
                temporal_choice
                * self.temporal_levels[None, :]
            ).sum(
                dim=1
            )

        else:

            s_budget = torch.full(
                (b,),
                float(
                    self.force_temporal_s
                ),
                device=patch.device,
                dtype=patch.dtype,
            )

        # ====================================================
        # TEMPORAL NOVELTY
        # ====================================================

        frame_features = pristine_z.mean(
            dim=2
        )

        novelty = torch.zeros(
            (b, self.T),
            device=patch.device,
            dtype=patch.dtype,
        )

        novelty[:, 0] = (
            frame_features[:, 0]
            .pow(2)
            .mean(dim=-1)
            .sqrt()
        )

        if self.T > 1:

            novelty[:, 1:] = (
                frame_features[:, 1:]
                - frame_features[:, :-1]
            ).pow(
                2
            ).mean(
                dim=-1
            ).sqrt()

        frame_mask = torch.zeros_like(
            novelty
        )

        for bi in range(b):

            si = int(
                s_budget[bi].item()
            )

            si = max(
                1,
                min(
                    si,
                    self.T,
                ),
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
            .expand(
                -1,
                -1,
                n,
            )
            .reshape(
                b,
                m,
            )
        )

        # ====================================================
        # JOINT MASK
        # ====================================================

        joint_keep = (
            spatial_hard
            * temporal_gate
        )

        # ====================================================
        # PHYSICAL CONSOLIDATION
        # ====================================================

        compact_patch, _ = (
            self.consolidator.forward_compact(
                pristine_patch,
                joint_keep,
            )
        )

        # Same straight-through temporal factor used by AdaST.
        # Forward value is exactly 1.
        temporal_st = (
            1.0
            + expected_s / self.T
            - (
                expected_s / self.T
            ).detach()
        )

        compact_patch = (
            compact_patch
            * temporal_st[:, None, None]
        )

        compact_hidden = torch.cat(
            [
                cls,
                compact_patch,
            ],
            dim=1,
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        self.last_probs = spatial_probs

        self.last_r = r.detach()

        self.last_expected_r = (
            spatial_probs
            * self.levels[None, :]
        ).sum(
            dim=1
        ).detach()

        self.last_actual_r = (
            actual_r.detach()
        )

        self.last_k = (
            k.detach()
        )

        self.last_idx = (
            learned_spatial_idx.detach()
        )

        self.last_t_probs = (
            temporal_probs
        )

        self.last_s = (
            s_budget.detach()
        )

        self.last_expected_s = (
            expected_s.detach()
        )

        self.last_t_idx = (
            learned_temporal_idx.detach()
        )

        self.last_compact_tokens = int(
            compact_patch.shape[1]
        )

        return (
            (compact_hidden,)
            + args[1:],
            kwargs,
        )


# ============================================================
# CHECKPOINT
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            if (
                key in checkpoint
                and isinstance(
                    checkpoint[key],
                    dict,
                )
            ):

                return checkpoint[key]

    return checkpoint


def normalize_state_dict(
    state_dict,
):

    normalized = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        normalized[key] = value

    return normalized


def load_e12_checkpoint(
    model,
    checkpoint_path,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"E12 checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = normalize_state_dict(
        extract_state_dict(
            checkpoint
        )
    )

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "[OK] E12 checkpoint loaded:"
    )

    print(
        "    ",
        checkpoint_path,
    )

    print(
        f"     missing="
        f"{len(result.missing_keys)}, "
        f"unexpected="
        f"{len(result.unexpected_keys)}"
    )


# ============================================================
# DATASET
# ============================================================

def build_test_loader(
    args,
    processor,
):

    root = Path(
        args.data_root
    )

    split_root = Path(
        args.split_root
    )

    if not root.exists():

        raise FileNotFoundError(
            f"UCF101 root does not exist: "
            f"{root}"
        )

    if not split_root.exists():

        raise FileNotFoundError(
            f"Split root does not exist: "
            f"{split_root}"
        )

    _, test_items = e6.read_split(
        root,
        split_root,
        args.split,
    )

    print(
        f"Official test entries: "
        f"{len(test_items)}"
    )

    # ========================================================
    # PREFLIGHT FILE CHECK
    # ========================================================

    missing = []

    for item in test_items:

        if isinstance(
            item,
            (tuple, list),
        ):

            path = Path(
                item[0]
            )

        else:

            path = Path(
                item
            )

        if not path.exists():

            missing.append(
                str(path)
            )

    if missing:

        print(
            f"[ERROR] Missing test videos: "
            f"{len(missing)}"
        )

        for path in missing[:20]:

            print(
                "   ",
                path,
            )

        raise RuntimeError(
            "UCF101 test set is incomplete."
        )

    print(
        "[OK] Every official test video exists."
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

    return loader


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_point(
    model,
    loader,
    device,
    args,
    rho,
    temporal_s,
):

    model.eval()

    model.set_forced_budget(
        rho,
        temporal_s,
    )

    total = 0
    correct1 = 0
    correct5 = 0

    actual_r_values = []
    k_values = []
    s_values = []
    compact_values = []

    start_time = time.time()

    for step, batch in enumerate(
        loader,
        start=1,
    ):

        x, y, _ = e12.unpack_batch(
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

            logits = model(
                x
            ).logits

        # Top-1
        prediction = logits.argmax(
            dim=1
        )

        correct1 += (
            prediction.eq(y)
            .sum()
            .item()
        )

        # Top-5
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
            .any(
                dim=1
            )
            .sum()
            .item()
        )

        total += y.numel()

        actual_r_values.extend(
            model.last_actual_r
            .float()
            .cpu()
            .tolist()
        )

        k_values.extend(
            model.last_k
            .float()
            .cpu()
            .tolist()
        )

        s_values.extend(
            model.last_s
            .float()
            .cpu()
            .tolist()
        )

        compact_values.append(
            float(
                model.last_compact_tokens
            )
        )

        if (
            step % args.print_every == 0
            or step == len(loader)
        ):

            print(
                f"{step:6d}/"
                f"{len(loader)} | "
                f"Top1="
                f"{100.0 * correct1 / total:.2f}% | "
                f"Top5="
                f"{100.0 * correct5 / total:.2f}%"
            )

    mean_actual_r = float(
        np.mean(
            actual_r_values
        )
    )

    mean_k = float(
        np.mean(
            k_values
        )
    )

    mean_s = float(
        np.mean(
            s_values
        )
    )

    mean_compact = float(
        np.mean(
            compact_values
        )
    )

    nominal_retention = (
        float(rho)
        * float(temporal_s)
        / float(args.num_frames)
    )

    actual_retention = (
        mean_actual_r
        * mean_s
        / float(args.num_frames)
    )

    # TimeSformer-B/16:
    # 14 x 14 = 196 patches/frame
    dense_patch_tokens = (
        196
        * args.num_frames
    )

    physical_retention = (
        mean_compact
        / float(
            dense_patch_tokens
        )
    )

    return {

        "rho_requested":
            float(rho),

        "S_requested":
            int(temporal_s),

        "samples":
            int(total),

        "top1":
            100.0
            * correct1
            / total,

        "top5":
            100.0
            * correct5
            / total,

        "K_per_frame":
            mean_k,

        "requested_spatial_retention":
            float(rho),

        "actual_spatial_retention":
            mean_actual_r,

        "physical_compact_patch_tokens":
            mean_compact,

        "nominal_effective_retention":
            nominal_retention,

        "nominal_effective_reduction_pct":
            100.0
            * (
                1.0
                - nominal_retention
            ),

        "actual_effective_retention":
            actual_retention,

        "actual_effective_reduction_pct":
            100.0
            * (
                1.0
                - actual_retention
            ),

        "physical_token_retention":
            physical_retention,

        "physical_token_reduction_pct":
            100.0
            * (
                1.0
                - physical_retention
            ),

        "seconds":
            time.time()
            - start_time,
    }


# ============================================================
# CANONICAL CSV SCHEMA
# ============================================================

CSV_FIELDS = [

    "rho_requested",

    "S_requested",

    "samples",

    "top1",

    "top5",

    "K_per_frame",

    "requested_spatial_retention",

    "actual_spatial_retention",

    "physical_compact_patch_tokens",

    "nominal_effective_retention",

    "nominal_effective_reduction_pct",

    "actual_effective_retention",

    "actual_effective_reduction_pct",

    "physical_token_retention",

    "physical_token_reduction_pct",

    "seconds",
]


# ============================================================
# OLD-RESULT COMPATIBILITY
# ============================================================

def normalize_result_row(
    row,
):

    row = dict(
        row
    )

    # First E14 result was produced before this field
    # was added to the schema.
    if (
        "requested_spatial_retention"
        not in row
        and "rho_requested"
        in row
    ):

        row[
            "requested_spatial_retention"
        ] = float(
            row[
                "rho_requested"
            ]
        )

    return row


# ============================================================
# SCHEMA-SAFE CSV
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    rows = [
        normalize_result_row(
            row
        )
        for row in rows
    ]

    # Canonical fields first.
    fieldnames = list(
        CSV_FIELDS
    )

    # Automatically support any future extra field.
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
        - set(
            fieldnames
        )
    )

    fieldnames.extend(
        extra_fields
    )

    with Path(
        path
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# ATOMIC JSON SAVE
# ============================================================

def atomic_json_write(
    path,
    data,
):

    path = Path(
        path
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    temporary.replace(
        path
    )


# ============================================================
# PARTIAL SAVE
# ============================================================

def save_partial(
    output_dir,
    results,
):

    results = [
        normalize_result_row(
            row
        )
        for row in results
    ]

    # IMPORTANT:
    # Save JSON first.
    # Even if CSV ever fails, completed experiment
    # is already preserved.
    atomic_json_write(
        output_dir
        / "budget_sweep_partial.json",
        results,
    )

    save_csv(
        output_dir
        / "budget_sweep_partial.csv",
        results,
    )


# ============================================================
# RESUME
# ============================================================

def load_partial(
    output_dir,
):

    partial_json = (
        output_dir
        / "budget_sweep_partial.json"
    )

    if not partial_json.exists():

        print(
            "[RESUME] No previous "
            "partial results found."
        )

        return []

    # utf-8-sig handles PowerShell-created JSON safely.
    with partial_json.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        results = json.load(
            file
        )

    if not isinstance(
        results,
        list,
    ):

        raise RuntimeError(
            "Partial JSON must contain a list."
        )

    results = [
        normalize_result_row(
            row
        )
        for row in results
    ]

    print(
        f"[RESUME] Loaded "
        f"{len(results)} completed "
        f"operating point(s)."
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AdaST E14 controlled "
            "spatio-temporal budget sweep "
            "with automatic resume"
        )
    )

    parser.add_argument(
        "--data-root",
        default=(
            r"D:\ST-EdgeViT"
            r"\datasets\UCF101"
        ),
    )

    parser.add_argument(
        "--split-root",
        default=(
            r"C:\Users\abc\AdaST"
            r"\splits\ucfTrainTestlist"
        ),
    )

    parser.add_argument(
        "--e12-checkpoint",
        default=(
            r"C:\Users\abc\AdaST"
            r"\runs\e12_rep_preserve"
            r"\best.pt"
        ),
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
        "--print-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=(
            r"C:\Users\abc\AdaST"
            r"\runs\e14_budget_sweep"
        ),
    )

    args = parser.parse_args()

    seed_all(
        args.seed
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 92
    )

    print(
        "AdaST E14 | CONTROLLED "
        "SPATIO-TEMPORAL BUDGET SWEEP"
    )

    print(
        "=" * 92
    )

    print(
        "Device        :",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU           :",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Data root     :",
        args.data_root,
    )

    print(
        "Split root    :",
        args.split_root,
    )

    print(
        "E12 checkpoint:",
        args.e12_checkpoint,
    )

    print(
        "Output        :",
        output_dir,
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

    processor = (
        AutoImageProcessor
        .from_pretrained(
            args.model_name
        )
    )

    # ========================================================
    # TEST SET
    # ========================================================

    test_loader = build_test_loader(
        args,
        processor,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding AdaST E14 model..."
    )

    backbone = (
        TimesformerForVideoClassification
        .from_pretrained(
            args.model_name,
            num_labels=101,
            ignore_mismatched_sizes=True,
        )
    )

    model = AdaSTE14(
        backbone,
        block=args.insert_block,
        T=args.num_frames,
        temperature=args.temperature,
    )

    model.enable_physical_tail()

    load_e12_checkpoint(
        model,
        args.e12_checkpoint,
    )

    model.to(
        device
    )

    model.eval()

    # ========================================================
    # LOAD PREVIOUS RESULTS
    # ========================================================

    results = load_partial(
        output_dir
    )

    completed = {

        (
            round(
                float(
                    row[
                        "rho_requested"
                    ]
                ),
                8,
            ),

            int(
                row[
                    "S_requested"
                ]
            ),
        )

        for row in results
    }

    total_points = (
        len(
            SPATIAL_LEVELS
        )
        * len(
            TEMPORAL_LEVELS
        )
    )

    print()

    print(
        "Spatial rho levels:",
        SPATIAL_LEVELS,
    )

    print(
        "Temporal S levels:",
        TEMPORAL_LEVELS,
    )

    print(
        "Total operating points:",
        total_points,
    )

    print(
        "Already completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        total_points
        - len(
            completed
        ),
    )

    # ========================================================
    # SWEEP
    # ========================================================

    for temporal_s in TEMPORAL_LEVELS:

        for rho in SPATIAL_LEVELS:

            key = (

                round(
                    float(
                        rho
                    ),
                    8,
                ),

                int(
                    temporal_s
                ),
            )

            # ================================================
            # RESUME
            # ================================================

            if key in completed:

                print(
                    f"[SKIP] "
                    f"rho={rho:.4f}, "
                    f"S={temporal_s} "
                    f"already completed."
                )

                continue

            print()

            print(
                "=" * 92
            )

            print(
                f"E14 | "
                f"rho={rho:.4f} | "
                f"S={temporal_s}"
            )

            print(
                "=" * 92
            )

            # ================================================
            # EVALUATE
            # ================================================

            result = evaluate_point(
                model,
                test_loader,
                device,
                args,
                rho,
                temporal_s,
            )

            # ================================================
            # PRINT BEFORE SAVE
            # ================================================

            print()

            print(
                "Samples                     : "
                f"{result['samples']}"
            )

            print(
                "Top-1                       : "
                f"{result['top1']:.4f}%"
            )

            print(
                "Top-5                       : "
                f"{result['top5']:.4f}%"
            )

            print(
                "K/frame                     : "
                f"{result['K_per_frame']:.2f}"
            )

            print(
                "Actual spatial retention    : "
                f"{result['actual_spatial_retention']:.6f}"
            )

            print(
                "Physical compact tokens     : "
                f"{result['physical_compact_patch_tokens']:.2f}"
            )

            print(
                "Nominal effective reduction : "
                f"{result['nominal_effective_reduction_pct']:.4f}%"
            )

            print(
                "Actual effective reduction  : "
                f"{result['actual_effective_reduction_pct']:.4f}%"
            )

            print(
                "Physical token reduction    : "
                f"{result['physical_token_reduction_pct']:.4f}%"
            )

            print(
                "Evaluation time             : "
                f"{result['seconds']:.1f} s"
            )

            # ================================================
            # SAVE
            # ================================================

            results.append(
                result
            )

            completed.add(
                key
            )

            save_partial(
                output_dir,
                results,
            )

            print(
                "[SAVE] Partial JSON/CSV "
                "updated safely."
            )

    # ========================================================
    # FINAL SORT
    # ========================================================

    results = [

        normalize_result_row(
            row
        )

        for row in results
    ]

    results = sorted(

        results,

        key=lambda row: (

            int(
                row[
                    "S_requested"
                ]
            ),

            float(
                row[
                    "rho_requested"
                ]
            ),
        ),
    )

    # ========================================================
    # FINAL CSV
    # ========================================================

    final_csv = (
        output_dir
        / "budget_sweep.csv"
    )

    save_csv(
        final_csv,
        results,
    )

    # ========================================================
    # FINAL JSON
    # ========================================================

    final_json = (
        output_dir
        / "budget_sweep.json"
    )

    atomic_json_write(

        final_json,

        {

            "experiment":
                "E14_controlled_budget_sweep",

            "dataset":
                "UCF101 Split-1",

            "checkpoint":
                str(
                    args.e12_checkpoint
                ),

            "spatial_levels":
                SPATIAL_LEVELS,

            "temporal_levels":
                TEMPORAL_LEVELS,

            "num_operating_points":
                len(
                    results
                ),

            "results":
                results,
        },
    )

    # Synchronize partial files.
    save_partial(
        output_dir,
        results,
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()

    print(
        "=" * 112
    )

    print(
        "E14 COMPLETE"
    )

    print(
        "=" * 112
    )

    print(
        f"{'rho':>8} "
        f"{'S':>3} "
        f"{'Top-1':>10} "
        f"{'Top-5':>10} "
        f"{'K':>7} "
        f"{'Tokens':>10} "
        f"{'Eff.Red%':>11} "
        f"{'Phy.Red%':>11}"
    )

    print(
        "-" * 112
    )

    for row in results:

        print(

            f"{row['rho_requested']:8.4f} "

            f"{row['S_requested']:3d} "

            f"{row['top1']:10.4f} "

            f"{row['top5']:10.4f} "

            f"{row['K_per_frame']:7.2f} "

            f"{row['physical_compact_patch_tokens']:10.2f} "

            f"{row['actual_effective_reduction_pct']:11.4f} "

            f"{row['physical_token_reduction_pct']:11.4f}"
        )

    print(
        "-" * 112
    )

    print(
        f"Completed points: "
        f"{len(results)}/"
        f"{total_points}"
    )

    print(
        "Final CSV :",
        final_csv,
    )

    print(
        "Final JSON:",
        final_json,
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":

    main()