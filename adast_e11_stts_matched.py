#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
E11 — STTS-style Matched Controlled Baseline
TimeSformer / UCF101 Split-1

IMPORTANT:
This is a mechanism-level STTS-style controlled baseline adapted to the
TimeSformer/AdaST evaluation pipeline. It is NOT an exact reproduction
of the original STTS implementation.

Matched budget:
    8 frames x 93 selected spatial tokens = 744 patch tokens
    Dense = 8 x 196 = 1568 patch tokens
    Reduction = 52.55%
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
)

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"


# ============================================================
# Utilities
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_class_index(root):
    p = Path(root) / "classInd.txt"

    if not p.exists():
        raise FileNotFoundError(p)

    d = {}

    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, c = line.split(maxsplit=1)
            d[c] = int(i) - 1

    return d


def read_split(data_root, split_root, split=1):

    dr = Path(data_root)
    sr = Path(split_root)

    cmap = read_class_index(sr)

    train_file = sr / f"trainlist{split:02d}.txt"
    test_file = sr / f"testlist{split:02d}.txt"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    train = []
    test = []

    for line in train_file.read_text(
        encoding="utf-8"
    ).splitlines():

        z = line.strip().split()

        if z:
            rel = z[0].replace("/", os.sep)

            train.append(
                (
                    dr / rel,
                    cmap[Path(rel).parts[0]],
                )
            )

    for line in test_file.read_text(
        encoding="utf-8"
    ).splitlines():

        rel = line.strip().replace("/", os.sep)

        if rel:
            test.append(
                (
                    dr / rel,
                    cmap[Path(rel).parts[0]],
                )
            )

    return train, test


def stratified(items, frac=0.10, seed=42):

    groups = defaultdict(list)

    for x in items:
        groups[x[1]].append(x)

    rng = random.Random(seed)

    train = []
    val = []

    for y in sorted(groups):

        a = groups[y][:]

        rng.shuffle(a)

        nv = max(
            1,
            min(
                len(a) - 1,
                round(len(a) * frac),
            ),
        )

        val += a[:nv]
        train += a[nv:]

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


def decode(path, T):

    try:
        container = av.open(str(path))

    except Exception as e:
        raise RuntimeError(
            f"Could not open {path}\n{e}"
        ) from e

    frames = []

    try:

        stream = container.streams.video[0]

        stream.thread_type = "AUTO"

        for f in container.decode(stream):

            frames.append(
                f.to_ndarray(format="rgb24")
            )

    finally:
        container.close()

    if not frames:
        raise RuntimeError(
            f"No frames decoded: {path}"
        )

    ids = np.linspace(
        0,
        len(frames) - 1,
        T,
    ).round().astype(int)

    return [frames[i] for i in ids]


# ============================================================
# Dataset
# ============================================================

class UCF(Dataset):

    def __init__(
        self,
        items,
        processor,
        T,
    ):

        self.items = items
        self.processor = processor
        self.T = T

    def __len__(self):

        return len(self.items)

    def __getitem__(self, i):

        p, y = self.items[i]

        frames = decode(
            p,
            self.T,
        )

        x = self.processor(
            images=frames,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return x, int(y), str(p)


def hits(logits, y):

    _, pred = logits.topk(
        min(5, logits.size(1)),
        1,
    )

    top1 = (
        pred[:, :1]
        .eq(y[:, None])
        .sum()
        .item()
    )

    top5 = (
        pred.eq(y[:, None])
        .any(1)
        .sum()
        .item()
    )

    return top1, top5


# ============================================================
# Checkpoint loading
# ============================================================

def state(obj):

    if isinstance(obj, dict):

        for k in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            if (
                k in obj
                and isinstance(obj[k], dict)
            ):

                return obj[k]

        return obj

    raise RuntimeError(
        "Unsupported checkpoint format"
    )


def norm(sd):

    out = {}

    for k, v in sd.items():

        while k.startswith("module."):
            k = k[7:]

        out[k] = v

    return out


def load_dense(backbone, path):

    sd = norm(
        state(
            torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        )
    )

    if any(
        k.startswith("backbone.")
        for k in sd
    ):

        sd = {
            (
                k[9:]
                if k.startswith("backbone.")
                else k
            ): v
            for k, v in sd.items()
        }

    result = backbone.load_state_dict(
        sd,
        strict=False,
    )

    print(
        "Dense checkpoint:",
        f"missing={len(result.missing_keys)},",
        f"unexpected={len(result.unexpected_keys)}",
    )


# ============================================================
# STTS-style scorer
# ============================================================

class STTSScorer(nn.Module):

    def __init__(
        self,
        d=768,
        hidden=256,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                3 * d,
                hidden,
            ),

            nn.GELU(),

            nn.LayerNorm(hidden),

            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        patch,
        T,
    ):

        B, M, D = patch.shape

        if M % T != 0:

            raise RuntimeError(
                f"{M} tokens not divisible by T={T}"
            )

        N = M // T

        z = patch.reshape(
            B,
            T,
            N,
            D,
        )

        # Temporal difference
        delta = torch.zeros_like(z)

        delta[:, 1:] = (
            z[:, 1:]
            - z[:, :-1]
        )

        # Global clip context
        ctx = (
            z.mean(
                dim=(1, 2),
                keepdim=True,
            )
            .expand(
                -1,
                T,
                N,
                -1,
            )
        )

        features = torch.cat(
            [
                z,
                delta,
                ctx,
            ],
            dim=-1,
        )

        scores = (
            self.net(features)
            .squeeze(-1)
        )

        return scores


# ============================================================
# Compact tail
# Same generic compact-tail principle used in E6-v2
# ============================================================

class CompactTailBlock(nn.Module):

    def __init__(
        self,
        d=768,
        heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):

        super().__init__()

        self.n1 = nn.LayerNorm(d)

        self.attn = nn.MultiheadAttention(
            d,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.n2 = nn.LayerNorm(d)

        hidden = int(
            d * mlp_ratio
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                d,
                hidden,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden,
                d,
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):

        q = self.n1(x)

        a, _ = self.attn(
            q,
            q,
            q,
            need_weights=False,
        )

        x = x + a

        return (
            x
            + self.mlp(
                self.n2(x)
            )
        )


class PhysicalCompactTail(nn.Module):

    def __init__(
        self,
        d=768,
        depth=6,
        heads=12,
    ):

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                CompactTailBlock(
                    d,
                    heads,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(d)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


# ============================================================
# STTS-style matched model
# ============================================================

class STTSMatched(nn.Module):

    def __init__(
        self,
        backbone,
        block=6,
        T=8,
        target_k=93,
        tau=0.25,
    ):

        super().__init__()

        self.backbone = backbone

        self.block = block

        self.T = T

        self.target_k = target_k

        self.tau = tau

        d = (
            backbone
            .config
            .hidden_size
        )

        self.scorer = STTSScorer(
            d=d,
            hidden=256,
        )

        depth = (
            len(self.layers())
            - block
        )

        self.compact_tail = (
            PhysicalCompactTail(
                d=d,
                depth=depth,
                heads=12,
            )
        )

        self.last_compact_tokens = None

        self.last_score_mean = None

        self.last_score_std = None

    def layers(self):

        return (
            self.backbone
            .timesformer
            .encoder
            .layer
        )

    def select(self, h):

        cls = h[:, :1]

        patch = h[:, 1:]

        B, M, D = patch.shape

        if M % self.T != 0:

            raise RuntimeError(
                f"{M} tokens not divisible by "
                f"T={self.T}"
            )

        N = M // self.T

        if self.target_k > N:

            raise RuntimeError(
                f"K={self.target_k} > N={N}"
            )

        z = patch.reshape(
            B,
            self.T,
            N,
            D,
        )

        scores = self.scorer(
            patch,
            self.T,
        )

        # ----------------------------------------
        # Hard Top-K forward selection
        # ----------------------------------------

        idx = torch.topk(
            scores,
            self.target_k,
            dim=2,
            largest=True,
            sorted=False,
        ).indices

        gather_idx = (
            idx
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                -1,
                D,
            )
        )

        selected = torch.gather(
            z,
            2,
            gather_idx,
        )

        # ----------------------------------------
        # Straight-through score surrogate
        # ----------------------------------------

        selected_scores = torch.gather(
            scores,
            2,
            idx,
        )

        threshold = (
            selected_scores
            .min(
                dim=2,
                keepdim=True,
            )
            .values
            .detach()
        )

        soft = torch.sigmoid(
            (
                scores
                - threshold
            )
            / self.tau
        )

        hard = torch.zeros_like(
            scores
        )

        hard.scatter_(
            2,
            idx,
            1.0,
        )

        st_gate = (
            hard
            + soft
            - soft.detach()
        )

        selected_gate = torch.gather(
            st_gate,
            2,
            idx,
        )

        selected = (
            selected
            * selected_gate
            .unsqueeze(-1)
        )

        # ----------------------------------------
        # Physical packed sequence
        # ----------------------------------------

        compact_patch = (
            selected.reshape(
                B,
                self.T
                * self.target_k,
                D,
            )
        )

        compact = torch.cat(
            [
                cls,
                compact_patch,
            ],
            dim=1,
        )

        self.last_compact_tokens = (
            self.T
            * self.target_k
        )

        self.last_score_mean = (
            scores
            .detach()
            .mean()
        )

        self.last_score_std = (
            scores
            .detach()
            .std(
                unbiased=False
            )
        )

        return compact

    def forward(self, x):

        ts = (
            self.backbone
            .timesformer
        )

        h = ts.embeddings(x)

        if isinstance(
            h,
            (tuple, list),
        ):

            h = h[0]

        # Dense prefix
        for i in range(
            self.block
        ):

            layer = self.layers()[i]

            try:

                z = layer(
                    h,
                    output_attentions=False,
                )

            except TypeError:

                z = layer(h)

            h = (
                z[0]
                if isinstance(
                    z,
                    (tuple, list),
                )
                else z
            )

        # STTS-style selection
        h = self.select(h)

        # Compact tail
        h = self.compact_tail(h)

        logits = (
            self.backbone
            .classifier(
                h[:, 0]
            )
        )

        return logits


# ============================================================
# Transfer compatible TimeSformer weights
# ============================================================

def initialize_compact_tail_from_timesformer(
    model
):

    copied = []

    src_layers = model.layers()

    for j, dst in enumerate(
        model.compact_tail.blocks
    ):

        si = model.block + j

        if si >= len(src_layers):
            break

        src = src_layers[si]

        # LayerNorm
        for name, dn in [

            (
                "layernorm_before",
                dst.n1,
            ),

            (
                "layernorm_after",
                dst.n2,
            ),
        ]:

            sn = getattr(
                src,
                name,
                None,
            )

            if (
                sn is not None
                and hasattr(
                    sn,
                    "weight",
                )
                and
                sn.weight.shape
                == dn.weight.shape
            ):

                dn.weight.data.copy_(
                    sn.weight.data
                )

                dn.bias.data.copy_(
                    sn.bias.data
                )

                copied.append(
                    f"{si}:{name}"
                )

        # MLP
        inter = getattr(
            getattr(
                src,
                "intermediate",
                None,
            ),
            "dense",
            None,
        )

        outdense = getattr(
            getattr(
                src,
                "output",
                None,
            ),
            "dense",
            None,
        )

        if (
            inter is not None
            and outdense is not None
        ):

            if (
                inter.weight.shape
                ==
                dst.mlp[0].weight.shape
            ):

                dst.mlp[0].weight.data.copy_(
                    inter.weight.data
                )

                dst.mlp[0].bias.data.copy_(
                    inter.bias.data
                )

                copied.append(
                    f"{si}:mlp_fc1"
                )

            if (
                outdense.weight.shape
                ==
                dst.mlp[3].weight.shape
            ):

                dst.mlp[3].weight.data.copy_(
                    outdense.weight.data
                )

                dst.mlp[3].bias.data.copy_(
                    outdense.bias.data
                )

                copied.append(
                    f"{si}:mlp_fc2"
                )

        # Attention
        att = getattr(
            src,
            "attention",
            None,
        )

        core = getattr(
            att,
            "attention",
            None,
        )

        qkv = getattr(
            core,
            "qkv",
            None,
        )

        proj = getattr(
            getattr(
                att,
                "output",
                None,
            ),
            "dense",
            None,
        )

        if (
            qkv is not None
            and
            qkv.weight.shape
            ==
            dst.attn.in_proj_weight.shape
        ):

            dst.attn.in_proj_weight.data.copy_(
                qkv.weight.data
            )

            if qkv.bias is not None:

                dst.attn.in_proj_bias.data.copy_(
                    qkv.bias.data
                )

            copied.append(
                f"{si}:attn_qkv"
            )

        if (
            proj is not None
            and
            proj.weight.shape
            ==
            dst.attn.out_proj.weight.shape
        ):

            dst.attn.out_proj.weight.data.copy_(
                proj.weight.data
            )

            dst.attn.out_proj.bias.data.copy_(
                proj.bias.data
            )

            copied.append(
                f"{si}:attn_proj"
            )

    print(
        f"Transferred {len(copied)} "
        "compatible TimeSformer "
        "tail groups."
    )


# ============================================================
# Trainable parameters
# ============================================================

def configure(model):

    for p in model.parameters():
        p.requires_grad = False

    # Train learned selector
    for p in model.scorer.parameters():
        p.requires_grad = True

    # Same compact-tail adaptation principle
    for p in model.compact_tail.parameters():
        p.requires_grad = True

    # Classifier
    for p in (
        model.backbone
        .classifier
        .parameters()
    ):
        p.requires_grad = True


# ============================================================
# Epoch
# ============================================================

def epoch(
    model,
    loader,
    device,
    amp,
    opt=None,
    scaler=None,
    accum=4,
):

    training = (
        opt is not None
    )

    model.train(training)

    ce = nn.CrossEntropyLoss()

    if training:
        opt.zero_grad(
            set_to_none=True
        )

    total = 0

    c1 = 0

    c5 = 0

    loss_sum = 0.0

    context = (
        torch.enable_grad
        if training
        else torch.no_grad
    )

    with context():

        for step, (
            x,
            y,
            _,
        ) in enumerate(
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
                    and
                    device.type
                    == "cuda"
                ),
            ):

                logits = model(x)

                loss = ce(
                    logits,
                    y,
                )

            if training:

                q = (
                    loss
                    / accum
                )

                if scaler:

                    scaler.scale(
                        q
                    ).backward()

                else:

                    q.backward()

                if (
                    step % accum == 0
                    or
                    step == len(loader)
                ):

                    pars = [
                        p
                        for p
                        in model.parameters()
                        if p.requires_grad
                    ]

                    if scaler:

                        scaler.unscale_(
                            opt
                        )

                        torch.nn.utils.clip_grad_norm_(
                            pars,
                            1.0,
                        )

                        scaler.step(
                            opt
                        )

                        scaler.update()

                    else:

                        torch.nn.utils.clip_grad_norm_(
                            pars,
                            1.0,
                        )

                        opt.step()

                    opt.zero_grad(
                        set_to_none=True
                    )

            a, b = hits(
                logits.detach(),
                y,
            )

            bs = y.size(0)

            total += bs

            c1 += a

            c5 += b

            loss_sum += (
                float(loss.item())
                * bs
            )

            if (
                training
                and
                (
                    step % 200 == 0
                    or
                    step == len(loader)
                )
            ):

                print(
                    f"  batch "
                    f"{step}/"
                    f"{len(loader)}"
                    f" | loss "
                    f"{loss_sum/total:.4f}"
                    f" | top1 "
                    f"{100*c1/total:.2f}%"
                    f" | K/frame "
                    f"{model.target_k}"
                )

    return {

        "loss":
            loss_sum / total,

        "top1":
            100.0 * c1 / total,

        "top5":
            100.0 * c5 / total,
    }


# ============================================================
# Final test
# ============================================================

@torch.no_grad()
def final_test(
    model,
    loader,
    device,
    amp,
    path,
):

    model.eval()

    total = 0

    c1 = 0

    c5 = 0

    rows = []

    for step, (
        x,
        y,
        paths,
    ) in enumerate(
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
                and
                device.type
                == "cuda"
            ),
        ):

            logits = model(x)

        a, b = hits(
            logits,
            y,
        )

        bs = y.size(0)

        total += bs

        c1 += a

        c5 += b

        pred = (
            logits.argmax(1)
            .cpu()
            .numpy()
        )

        yy = (
            y.cpu()
            .numpy()
        )

        for j in range(bs):

            rows.append(
                [
                    paths[j],
                    int(yy[j]),
                    int(pred[j]),
                    int(
                        pred[j]
                        == yy[j]
                    ),
                    model.target_k,
                    model.last_compact_tokens,
                ]
            )

        if (
            step % 250 == 0
            or
            step == len(loader)
        ):

            print(
                f"  test "
                f"{step}/"
                f"{len(loader)}"
                f" | Top-1 "
                f"{100*c1/total:.2f}%"
                f" | Top-5 "
                f"{100*c5/total:.2f}%"
            )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "video",
                "label",
                "prediction",
                "correct",
                "k_per_frame",
                "compact_patch_tokens",
            ]
        )

        w.writerows(rows)

    return {

        "samples": total,

        "top1":
            100.0 * c1 / total,

        "top5":
            100.0 * c5 / total,
    }


# ============================================================
# Main
# ============================================================

def main():

    p = argparse.ArgumentParser()

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
        "--split",
        type=int,
        default=1,
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
        "--target-k",
        type=int,
        default=93,
    )

    p.add_argument(
        "--tau",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    p.add_argument(
        "--accum-steps",
        type=int,
        default=4,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--amp",
        action="store_true",
    )

    p.add_argument(
        "--output",
        default="runs/e11_stts_matched",
    )

    a = p.parse_args()

    seed_all(a.seed)

    out = Path(a.output)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dense_tokens = (
        196
        * a.num_frames
    )

    compact_tokens = (
        a.target_k
        * a.num_frames
    )

    retention = (
        compact_tokens
        / dense_tokens
    )

    reduction = (
        1.0
        - retention
    )

    print("=" * 88)

    print(
        "E11 — STTS-STYLE "
        "MATCHED CONTROLLED BASELINE"
    )

    print("=" * 88)

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
        "Frames:",
        a.num_frames,
    )

    print(
        "Insertion block:",
        a.insert_block,
    )

    print(
        "K/frame:",
        a.target_k,
    )

    print(
        "Dense patch tokens:",
        dense_tokens,
    )

    print(
        "Compact patch tokens:",
        compact_tokens,
    )

    print(
        f"Retention: "
        f"{100*retention:.2f}%"
    )

    print(
        f"Reduction: "
        f"{100*reduction:.2f}%"
    )

    print("=" * 88)

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    original_train, test = (
        read_split(
            a.data_root,
            a.split_root,
            a.split,
        )
    )

    train, val = stratified(
        original_train,
        a.val_fraction,
        a.seed,
    )

    print(
        f"Train={len(train)}, "
        f"Val={len(val)}, "
        f"Test={len(test)}"
    )

    kwargs = dict(

        batch_size=
            a.batch_size,

        num_workers=
            a.workers,

        pin_memory=
            device.type == "cuda",

        persistent_workers=
            a.workers > 0,
    )

    train_loader = DataLoader(
        UCF(
            train,
            processor,
            a.num_frames,
        ),
        shuffle=True,
        **kwargs,
    )

    val_loader = DataLoader(
        UCF(
            val,
            processor,
            a.num_frames,
        ),
        shuffle=False,
        **kwargs,
    )

    test_loader = DataLoader(
        UCF(
            test,
            processor,
            a.num_frames,
        ),
        shuffle=False,
        **kwargs,
    )

    backbone = (
        TimesformerForVideoClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=101,
            ignore_mismatched_sizes=True,
        )
    )

    load_dense(
        backbone,
        a.dense_checkpoint,
    )

    model = STTSMatched(

        backbone,

        block=
            a.insert_block,

        T=
            a.num_frames,

        target_k=
            a.target_k,

        tau=
            a.tau,
    )

    initialize_compact_tail_from_timesformer(
        model
    )

    configure(model)

    model.to(device)

    trainable = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable/1e6:.3f} M"
    )

    optimizer = torch.optim.AdamW(

        [
            p
            for p
            in model.parameters()
            if p.requires_grad
        ],

        lr=a.lr,

        weight_decay=
            a.weight_decay,
    )

    scheduler = (
        torch.optim
        .lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=a.epochs,
        )
    )

    scaler = (

        torch.amp.GradScaler(
            "cuda"
        )

        if (
            a.amp
            and
            device.type == "cuda"
        )

        else None
    )

    best = -1.0

    best_path = (
        out / "best.pt"
    )

    metrics_path = (
        out / "metrics.csv"
    )

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        csv.writer(f).writerow(
            [
                "epoch",
                "train_top1",
                "train_top5",
                "val_top1",
                "val_top5",
            ]
        )

    for ep in range(
        1,
        a.epochs + 1,
    ):

        print(
            f"\nEPOCH "
            f"{ep}/"
            f"{a.epochs}"
        )

        t0 = time.time()

        trm = epoch(

            model,
            train_loader,
            device,
            a.amp,

            opt=optimizer,

            scaler=scaler,

            accum=
                a.accum_steps,
        )

        vam = epoch(

            model,
            val_loader,
            device,
            a.amp,
        )

        print(
            f"Train Top-1="
            f"{trm['top1']:.2f}%"
            f" | Top-5="
            f"{trm['top5']:.2f}%"
        )

        print(
            f"Val Top-1="
            f"{vam['top1']:.2f}%"
            f" | Top-5="
            f"{vam['top5']:.2f}%"
        )

        print(
            f"Epoch time="
            f"{time.time()-t0:.1f}s"
        )

        with metrics_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            csv.writer(f).writerow(
                [
                    ep,
                    trm["top1"],
                    trm["top5"],
                    vam["top1"],
                    vam["top5"],
                ]
            )

        if (
            vam["top1"]
            > best
        ):

            best = vam["top1"]

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        ep,

                    "val":
                        vam,

                    "args":
                        vars(a),
                },

                best_path,
            )

            print(
                "Saved new best "
                "E11 checkpoint."
            )

        scheduler.step()

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)

    print(
        "\nRunning official "
        "UCF101 Split-1 test..."
    )

    test_metrics = final_test(

        model,

        test_loader,

        device,

        a.amp,

        out
        / "per_video_test.csv",
    )

    summary = {

        "experiment":
            "E11 STTS-style matched controlled baseline",

        "dataset":
            "UCF101 Split-1",

        "samples":
            test_metrics["samples"],

        "test_top1":
            test_metrics["top1"],

        "test_top5":
            test_metrics["top5"],

        "frames":
            a.num_frames,

        "insertion_block":
            a.insert_block,

        "spatial_k_per_frame":
            a.target_k,

        "dense_patch_tokens":
            dense_tokens,

        "compact_patch_tokens":
            compact_tokens,

        "patch_token_retention":
            retention,

        "patch_token_reduction":
            reduction,

        "physical_sequence_reduction":
            True,

        "method_note":
            "STTS-style learned Top-K controlled baseline adapted to TimeSformer; not exact original STTS reproduction.",

        "checkpoint":
            str(
                best_path.resolve()
            ),
    }

    (
        out
        / "summary.json"
    ).write_text(

        json.dumps(
            summary,
            indent=2,
        ),

        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "E11 FINAL TEST RESULT"
    )

    print(
        "=" * 88
    )

    print(
        f"Samples : "
        f"{test_metrics['samples']}"
    )

    print(
        f"Top-1   : "
        f"{test_metrics['top1']:.4f}%"
    )

    print(
        f"Top-5   : "
        f"{test_metrics['top5']:.4f}%"
    )

    print(
        f"K/frame : "
        f"{a.target_k}"
    )

    print(
        f"Dense patch tokens   : "
        f"{dense_tokens}"
    )

    print(
        f"Compact patch tokens : "
        f"{compact_tokens}"
    )

    print(
        f"Retention : "
        f"{100*retention:.2f}%"
    )

    print(
        f"Reduction : "
        f"{100*reduction:.2f}%"
    )

    print(
        "Physical compaction: YES"
    )

    print(
        "Report as: STTS-style "
        "controlled baseline"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()