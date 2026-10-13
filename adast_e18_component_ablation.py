#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AdaST E18 — Frozen Component Intervention

Purpose
-------
Evaluate the final E12 checkpoint on official UCF101 Split-1 under:

1. spatial_only
   - learned spatial K(X)
   - temporal selection disabled: S = T = 8

2. temporal_only
   - spatial selection disabled: K = N = 196
   - learned temporal S(X)

3. full
   - learned K(X)
   - learned S(X)

IMPORTANT
---------
This is a frozen component intervention.
No retraining is performed.
Do NOT describe spatial_only or temporal_only as independently trained models.

The physical compact E12 tail is used in all variants.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

import adast_e7_v3_final_benchmark as base


# ---------------------------------------------------------------------
# E18 MODEL
# ---------------------------------------------------------------------

class AdaSTE18(base.AdaSTE5):

    def __init__(
        self,
        backbone,
        block=6,
        T=8,
        temperature=1.0,
        variant="full",
    ):
        self.e18_variant = variant

        if variant not in {
            "full",
            "spatial_only",
            "temporal_only",
        }:
            raise ValueError(
                f"Unknown E18 variant: {variant}"
            )

        super().__init__(
            backbone,
            block=block,
            T=T,
            temperature=temperature,
        )

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

        pristine_patch = patch
        pristine_z = pristine_patch.reshape(
            b,
            self.T,
            n,
            d,
        )

        # =============================================================
        # SPATIAL POLICY
        # =============================================================

        scores, delta = self.scorer(
            pristine_patch,
            self.T,
        )

        logits = self.budget(
            scores,
            delta,
            cls,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        if self.e18_variant == "temporal_only":

            # Disable spatial selection.
            # Every patch from every retained frame is available.
            spatial_hard = torch.ones(
                (b, m),
                device=patch.device,
                dtype=patch.dtype,
            )

            k = torch.full(
                (b,),
                n,
                device=patch.device,
                dtype=torch.long,
            )

            r = torch.ones(
                (b,),
                device=patch.device,
                dtype=patch.dtype,
            )

            idx = torch.full(
                (b,),
                len(base.LEVELS) - 1,
                device=patch.device,
                dtype=torch.long,
            )

        else:

            # Original learned E12 spatial policy.
            idx = logits.argmax(1)

            choice = F.one_hot(
                idx,
                num_classes=len(base.LEVELS),
            ).to(logits.dtype)

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

        # =============================================================
        # TEMPORAL POLICY
        # =============================================================

        t_logits = self.temporal_budget(
            pristine_z
        )

        t_probs = torch.softmax(
            t_logits,
            dim=1,
        )

        if self.e18_variant == "spatial_only":

            # Disable temporal selection.
            # Keep all T frames.
            s_budget = torch.full(
                (b,),
                float(self.T),
                device=patch.device,
                dtype=patch.dtype,
            )

            frame_mask = torch.ones(
                (b, self.T),
                device=patch.device,
                dtype=patch.dtype,
            )

            # Index 2 corresponds to S=8 in {6,7,8}.
            t_idx = torch.full(
                (b,),
                2,
                device=patch.device,
                dtype=torch.long,
            )

            expected_s = s_budget

        else:

            # Frozen deterministic E12 temporal decision.
            t_idx = t_logits.argmax(1)

            t_choice = F.one_hot(
                t_idx,
                num_classes=len(
                    self.temporal_levels
                ),
            ).to(
                t_logits.dtype
            )

            s_budget = (
                t_choice
                * self.temporal_levels[None, :]
            ).sum(1)

            expected_s = (
                t_probs
                * self.temporal_levels[None, :]
            ).sum(1)

            # Original E12 temporal novelty.
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

        # =============================================================
        # JOINT RETENTION
        # =============================================================

        temporal_gate = (
            frame_mask[:, :, None]
            .expand(-1, -1, n)
            .reshape(b, m)
        )

        joint_keep = (
            spatial_hard
            * temporal_gate
        )

        # Evaluation only.
        # No straight-through training surrogate is required.
        t_st = torch.ones(
            (b,),
            device=patch.device,
            dtype=patch.dtype,
        )

        # =============================================================
        # CONSOLIDATION / PHYSICAL COMPACTION
        # =============================================================

        if self.physical_mode:

            compact_patch, _ = (
                self.consolidator.forward_compact(
                    pristine_patch,
                    joint_keep,
                )
            )

            self.last_compact_tokens = int(
                compact_patch.size(1)
            )

            joint_patch = (
                compact_patch
                * t_st[:, None, None]
            )

        else:

            joint_patch = self.consolidator(
                pristine_patch,
                joint_keep,
            )

            self.last_compact_tokens = (
                int(
                    joint_keep[0]
                    .sum()
                    .item()
                )
                if b == 1
                else None
            )

        gh = torch.cat(
            [
                cls,
                joint_patch,
            ],
            dim=1,
        )

        # =============================================================
        # STATISTICS
        # =============================================================

        self.last_probs = probs

        self.last_r = r.detach()

        self.last_expected_r = r.detach()

        self.last_actual_r = (
            spatial_hard
            .mean(1)
            .detach()
        )

        self.last_k = k.detach()

        self.last_idx = idx.detach()

        self.last_t_probs = t_probs

        self.last_s = (
            s_budget.detach()
        )

        self.last_expected_s = (
            expected_s.detach()
        )

        self.last_t_idx = (
            t_idx.detach()
        )

        self.last_merge_gate = (
            self.consolidator
            .last_gate
            .detach()
        )

        self.last_merge_entropy = (
            self.consolidator
            .last_assignment_entropy
            .detach()
        )

        if src == "args":
            return (
                (gh,) + rest,
                kwargs,
            )

        kw = dict(kwargs)
        kw["hidden_states"] = gh

        return args, kw


# ---------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------

def load_final_e12(
    model,
    dense_checkpoint,
    e12_checkpoint,
):

    base.load_dense(
        model.backbone,
        dense_checkpoint,
    )

    depth = model.enable_physical_tail()

    print(
        f"Physical compact-tail depth: {depth}"
    )

    sd = base.norm(
        base.state(
            torch.load(
                e12_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        )
    )

    z = model.load_state_dict(
        sd,
        strict=False,
    )

    if (
        z.missing_keys
        or z.unexpected_keys
    ):

        raise RuntimeError(
            "E12 checkpoint mismatch:\n"
            f"missing={z.missing_keys[:30]}\n"
            f"unexpected={z.unexpected_keys[:30]}"
        )

    print(
        "E12 checkpoint load: "
        "missing=0, unexpected=0"
    )


# ---------------------------------------------------------------------
# OFFICIAL TEST
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp=False,
):

    model.eval()

    total = 0
    correct1 = 0
    correct5 = 0

    compact_tokens = []
    spatial_r = []
    ks = []
    temporal_s = []

    for step, (x, y, _) in enumerate(
        loader,
        1,
    ):

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=(
                amp
                and device.type == "cuda"
            ),
        ):

            out = model(x)

        logits = out.logits

        _, pred = logits.topk(
            min(
                5,
                logits.size(1),
            ),
            dim=1,
        )

        correct1 += (
            pred[:, :1]
            .eq(y[:, None])
            .sum()
            .item()
        )

        correct5 += (
            pred
            .eq(y[:, None])
            .any(1)
            .sum()
            .item()
        )

        total += y.size(0)

        compact_tokens.append(
            float(
                model.last_compact_tokens
            )
        )

        spatial_r.extend(
            model.last_actual_r
            .float()
            .cpu()
            .tolist()
        )

        ks.extend(
            model.last_k
            .float()
            .cpu()
            .tolist()
        )

        temporal_s.extend(
            model.last_s
            .float()
            .cpu()
            .tolist()
        )

        if (
            step % 250 == 0
            or step == len(loader)
        ):

            print(
                f"[{step:4d}/{len(loader)}] "
                f"Top1={100*correct1/total:.3f}% | "
                f"Top5={100*correct5/total:.3f}% | "
                f"tokens={np.mean(compact_tokens):.2f}"
            )

    c = np.asarray(
        compact_tokens,
        dtype=float,
    )

    r = np.asarray(
        spatial_r,
        dtype=float,
    )

    k = np.asarray(
        ks,
        dtype=float,
    )

    ss = np.asarray(
        temporal_s,
        dtype=float,
    )

    retention = (
        c.mean()
        / 1568.0
    )

    return {
        "n": int(total),

        "top1_percent":
            100.0
            * correct1
            / total,

        "top5_percent":
            100.0
            * correct5
            / total,

        "spatial_retention_mean":
            float(r.mean()),

        "k_per_frame_mean":
            float(k.mean()),

        "temporal_S_mean":
            float(ss.mean()),

        "physical_patch_tokens_mean":
            float(c.mean()),

        "physical_patch_tokens_std":
            float(c.std()),

        "physical_patch_tokens_min":
            int(c.min()),

        "physical_patch_tokens_max":
            int(c.max()),

        "physical_patch_retention":
            float(retention),

        "physical_patch_reduction_percent":
            float(
                100.0
                * (1.0 - retention)
            ),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    p = argparse.ArgumentParser(
        "AdaST E18 component ablation"
    )

    p.add_argument(
        "--data-root",
        required=True,
    )

    p.add_argument(
        "--split-root",
        required=True,
    )

    p.add_argument(
        "--dense-checkpoint",
        required=True,
    )

    p.add_argument(
        "--e12-checkpoint",
        required=True,
    )

    p.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    p.add_argument(
        "--insert-block",
        type=int,
        default=6,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--amp",
        action="store_true",
    )

    p.add_argument(
        "--output",
        default=(
            "runs/"
            "e18_component_ablation"
        ),
    )

    a = p.parse_args()

    os.makedirs(
        a.output,
        exist_ok=True,
    )

    base.seed_all(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 88)
    print(
        "AdaST E18 — "
        "FROZEN COMPONENT INTERVENTION"
    )
    print("=" * 88)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "CPU",
    )

    print(
        "AMP:",
        a.amp,
    )

    _, test_items = base.read_split(
        a.data_root,
        a.split_root,
        split=1,
    )

    if len(test_items) != 3783:

        raise RuntimeError(
            "Expected 3783 official "
            "UCF101 Split-1 test videos, "
            f"found {len(test_items)}"
        )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            base.MODEL_NAME
        )
    )

    ds = base.UCF(
        test_items,
        processor,
        a.num_frames,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "Official test videos:",
        len(ds),
    )

    variants = [
        (
            "spatial_only",
            "adaptive K(X), force S=8",
        ),
        (
            "temporal_only",
            "force K=196, adaptive S(X)",
        ),
        (
            "full",
            "adaptive K(X) + adaptive S(X)",
        ),
    ]

    results = {}

    for variant, description in variants:

        print(
            "\n"
            + "=" * 88
        )

        print(
            variant,
            "—",
            description,
        )

        print(
            "=" * 88
        )

        model = AdaSTE18(
            base.make_backbone(
                a.num_frames
            ),
            block=a.insert_block,
            T=a.num_frames,
            variant=variant,
        )

        load_final_e12(
            model,
            a.dense_checkpoint,
            a.e12_checkpoint,
        )

        model.to(
            device
        )

        q = evaluate(
            model,
            loader,
            device,
            amp=a.amp,
        )

        q["description"] = (
            description
        )

        results[variant] = q

        print(
            "\nRESULT:",
            variant,
        )

        print(
            f"Top-1            : "
            f"{q['top1_percent']:.4f}%"
        )

        print(
            f"Top-5            : "
            f"{q['top5_percent']:.4f}%"
        )

        print(
            f"K/frame          : "
            f"{q['k_per_frame_mean']:.4f}"
        )

        print(
            f"Temporal S       : "
            f"{q['temporal_S_mean']:.4f}"
        )

        print(
            f"Physical tokens  : "
            f"{q['physical_patch_tokens_mean']:.2f}"
        )

        print(
            f"Token reduction  : "
            f"{q['physical_patch_reduction_percent']:.4f}%"
        )

        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

    full = results["full"]

    sp = results[
        "spatial_only"
    ]

    tp = results[
        "temporal_only"
    ]

    results[
        "controlled_deltas_vs_full"
    ] = {

        "spatial_only_minus_full_top1_pp":
            sp["top1_percent"]
            - full["top1_percent"],

        "temporal_only_minus_full_top1_pp":
            tp["top1_percent"]
            - full["top1_percent"],

        "spatial_only_minus_full_top5_pp":
            sp["top5_percent"]
            - full["top5_percent"],

        "temporal_only_minus_full_top5_pp":
            tp["top5_percent"]
            - full["top5_percent"],
    }

    results["protocol"] = {

        "training_performed":
            False,

        "checkpoint":
            "final E12",

        "weights_frozen":
            True,

        "dataset":
            "UCF101",

        "split":
            1,

        "official_test_n":
            3783,

        "amp":
            bool(a.amp),

        "interpretation":
            (
                "Frozen component "
                "intervention; not "
                "independently retrained "
                "ablations."
            ),
    }

    out = os.path.join(
        a.output,
        "summary.json",
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "E18 — FINAL COMPONENT ABLATION"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Variant':<16}"
        f"{'Top-1':>10}"
        f"{'Top-5':>10}"
        f"{'K/frame':>12}"
        f"{'S':>10}"
        f"{'Tokens':>12}"
        f"{'Reduction':>12}"
    )

    for key in (
        "spatial_only",
        "temporal_only",
        "full",
    ):

        q = results[key]

        print(
            f"{key:<16}"
            f"{q['top1_percent']:>10.4f}"
            f"{q['top5_percent']:>10.4f}"
            f"{q['k_per_frame_mean']:>12.2f}"
            f"{q['temporal_S_mean']:>10.4f}"
            f"{q['physical_patch_tokens_mean']:>12.2f}"
            f"{q['physical_patch_reduction_percent']:>11.2f}%"
        )

    print(
        "\nTop-1 deltas vs Full E12:"
    )

    print(
        "Spatial-only - Full : "
        f"{sp['top1_percent'] - full['top1_percent']:+.4f} pp"
    )

    print(
        "Temporal-only - Full: "
        f"{tp['top1_percent'] - full['top1_percent']:+.4f} pp"
    )

    print(
        "\nIMPORTANT: "
        "No retraining was performed."
    )

    print(
        "These are frozen "
        "component interventions."
    )

    print(
        "\nSummary:",
        out,
    )


if __name__ == "__main__":
    main()