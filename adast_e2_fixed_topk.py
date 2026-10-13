#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaST E2 — Learned Spatio-Temporal Importance + Fixed Top-K Gating
==================================================================

Purpose
-------
Train a learned spatio-temporal importance scorer on top of an already fine-tuned TimeSformer UCF101 checkpoint while keeping a FIXED retention
ratio. Adaptive budget, token consolidation and preservation loss are OFF.

Important implementation note
-----------------------------
Hugging Face TimeSformer uses divided space-time attention and expects the
regular T x N token grid in later layers. Physically deleting arbitrary
mid-network tokens breaks that structure. Therefore E2 uses a HARD
straight-through token gate at an intermediate block:
    forward pass  -> selected tokens kept, rejected patch tokens zeroed
    backward pass -> gradients flow through the learned importance scores

This is the clean ablation for "does the learned scorer identify useful
spatio-temporal tokens?". It DOES NOT claim wall-clock/FLOP reduction because
the sequence shape is intentionally preserved. Physical sequence
consolidation/reduction belongs to E4/E5.

Expected environment
--------------------
Python 3.11
PyTorch
transformers
torchvision
PyAV
numpy

Example
-------
python e2_adast_fixed_topk.py ^
  --data-root "C:\\Users\\abc\\AdaST\\datasets\\UCF101" ^
  --split-root "C:\\Users\\abc\\AdaST\\splits\\ucfTrainTestlist" ^
  --checkpoint "C:\\Users\\abc\\AdaST\\checkpoints\\timesformer_ucf101_best.pt" ^
  --retention 0.50 ^
  --insert-block 6 ^
  --epochs 3 ^
  --batch-size 1 ^
  --accum-steps 4 ^
  --workers 2 ^
  --amp ^
  --output "runs\\adast_e2_r050"
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, TimesformerForVideoClassification


MODEL_NAME = "facebook/timesformer-base-finetuned-k400"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# UCF101
# ---------------------------------------------------------------------
def read_class_index(split_root: Path):
    class_file = split_root / "classInd.txt"
    if not class_file.exists():
        raise FileNotFoundError(f"Missing: {class_file}")

    name_to_zero = {}
    with class_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(maxsplit=1)
            name_to_zero[name] = int(idx) - 1
    return name_to_zero


def read_ucf_split(data_root: Path, split_root: Path, split: int = 1):
    class_map = read_class_index(split_root)

    train_file = split_root / f"trainlist{split:02d}.txt"
    test_file = split_root / f"testlist{split:02d}.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing: {test_file}")

    train_items = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rel = parts[0].replace("/", os.sep)
            class_name = Path(rel).parts[0]
            label = class_map[class_name]
            train_items.append((data_root / rel, label))

    test_items = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip().replace("/", os.sep)
            if not rel:
                continue
            class_name = Path(rel).parts[0]
            label = class_map[class_name]
            test_items.append((data_root / rel, label))

    return train_items, test_items


def stratified_train_val(items, val_fraction=0.10, seed=42):
    by_class = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    train_items, val_items = [], []

    for label in sorted(by_class):
        group = list(by_class[label])
        rng.shuffle(group)

        n_val = max(1, int(round(len(group) * val_fraction)))
        n_val = min(n_val, max(1, len(group) - 1))

        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def decode_uniform_frames(video_path: Path, num_frames: int):
    """
    Decode a video with PyAV and uniformly sample num_frames frames.
    Returns a list of RGB numpy arrays.
    """
    try:
        container = av.open(str(video_path))
    except Exception as e:
        raise RuntimeError(f"Could not open video: {video_path}\n{e}") from e

    frames = []
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded: {video_path}")

    if len(frames) >= num_frames:
        ids = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    else:
        ids = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)

    return [frames[int(i)] for i in ids]


class UCF101TimesformerDataset(Dataset):
    def __init__(self, items, processor, num_frames=8):
        self.items = items
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        try:
            frames = decode_uniform_frames(path, self.num_frames)
            pixel_values = self.processor(
                images=frames,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

            # HF video processors normally return [T, C, H, W].
            if pixel_values.ndim != 4:
                raise RuntimeError(
                    f"Expected [T,C,H,W], got {tuple(pixel_values.shape)}"
                )

            return pixel_values, int(label), str(path)

        except Exception as e:
            # Fail loudly: silently replacing videos would corrupt evaluation.
            raise RuntimeError(f"Dataset failure at index {index}: {path}\n{e}") from e


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def topk_correct(logits, targets, ks=(1, 5)):
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    out = []
    for k in ks:
        kk = min(k, logits.size(1))
        out.append(correct[:kk].reshape(-1).float().sum().item())
    return out


# ---------------------------------------------------------------------
# AdaST E2 scorer
# ---------------------------------------------------------------------
class STImportanceScorer(nn.Module):
    """
    s_{t,i} = sigmoid( f_phi([z_{t,i}; delta z_{t,i}; c]) )

    hidden patch tokens are expected as [B, T*N, D].
    """
    def __init__(self, dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        # Slightly non-saturated start.
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, patch_tokens, num_frames):
        b, m, d = patch_tokens.shape
        if m % num_frames != 0:
            raise RuntimeError(
                f"Patch token count {m} is not divisible by frames={num_frames}."
            )

        n = m // num_frames
        z = patch_tokens.reshape(b, num_frames, n, d)

        delta = torch.zeros_like(z)
        delta[:, 1:] = z[:, 1:] - z[:, :-1]

        context = z.mean(dim=(1, 2), keepdim=True)
        context = context.expand(-1, num_frames, n, -1)

        feat = torch.cat((z, delta, context), dim=-1)
        scores = torch.sigmoid(self.net(feat)).squeeze(-1)

        return scores.reshape(b, m)


class AdaSTE2Wrapper(nn.Module):
    """
    Wrap a HF TimesformerForVideoClassification.

    A persistent pre-hook on encoder layer `insert_block` performs
    straight-through hard Top-K gating while preserving sequence shape.
    """

    def __init__(
        self,
        backbone,
        retention=0.50,
        insert_block=6,
        num_frames=8,
        scorer_hidden=256,
    ):
        super().__init__()

        self.backbone = backbone
        self.retention = float(retention)
        self.insert_block = int(insert_block)
        self.num_frames = int(num_frames)

        hidden_size = int(backbone.config.hidden_size)
        self.scorer = STImportanceScorer(
            dim=hidden_size,
            hidden_dim=scorer_hidden
        )

        self.last_scores = None
        self.last_hard_mask = None
        self.last_soft_mask = None
        self.last_k = None
        self.last_total_patch_tokens = None

        layers = self._encoder_layers()
        if not (0 <= self.insert_block < len(layers)):
            raise ValueError(
                f"insert_block={self.insert_block}, "
                f"but TimeSformer has {len(layers)} layers."
            )

        self._hook_handle = layers[self.insert_block].register_forward_pre_hook(
            self._gate_pre_hook,
            with_kwargs=True,
        )

    def _encoder_layers(self):
        # HF TimeSformer path used by current transformers releases.
        if hasattr(self.backbone, "timesformer"):
            ts = self.backbone.timesformer
            if hasattr(ts, "encoder") and hasattr(ts.encoder, "layer"):
                return ts.encoder.layer

        raise AttributeError(
            "Could not locate TimeSformer encoder layers at "
            "model.timesformer.encoder.layer. "
            "Check your transformers version/model implementation."
        )

    def _gate_pre_hook(self, module, args, kwargs):
        # First positional argument should be hidden_states.
        if len(args) > 0:
            hidden = args[0]
            remaining_args = args[1:]
            source = "args"
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
            remaining_args = ()
            source = "kwargs"
        else:
            raise RuntimeError(
                "Could not locate hidden_states in TimeSformer layer input."
            )

        if hidden.ndim != 3:
            raise RuntimeError(
                f"Expected hidden states [B,S,D], got {tuple(hidden.shape)}."
            )

        # TimeSformer sequence = CLS token + T*N patch tokens.
        cls_token = hidden[:, :1, :]
        patch_tokens = hidden[:, 1:, :]

        b, m, d = patch_tokens.shape
        if m % self.num_frames != 0:
            raise RuntimeError(
                f"At insert block {self.insert_block}, patch tokens={m}; "
                f"not divisible by num_frames={self.num_frames}. "
                "Use the same frame count as the backbone input."
            )

        scores = self.scorer(patch_tokens, self.num_frames)

        k = max(1, min(m, int(round(m * self.retention))))

        topk_idx = torch.topk(
            scores,
            k=k,
            dim=1,
            largest=True,
            sorted=False
        ).indices

        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)

        # Straight-through gate:
        # forward value == hard mask
        # backward derivative flows through scores
        st_gate = hard_mask + scores - scores.detach()

        gated_patch_tokens = patch_tokens * st_gate.unsqueeze(-1)
        gated_hidden = torch.cat((cls_token, gated_patch_tokens), dim=1)

        self.last_scores = scores.detach()
        self.last_hard_mask = hard_mask.detach()
        self.last_soft_mask = scores.detach()
        self.last_k = k
        self.last_total_patch_tokens = m

        if source == "args":
            new_args = (gated_hidden,) + remaining_args
            return new_args, kwargs

        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = gated_hidden
        return args, new_kwargs

    def forward(self, pixel_values, labels=None):
        return self.backbone(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True
        )

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------
# Model / checkpoint
# ---------------------------------------------------------------------
def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        candidates = [
            "model_state_dict",
            "state_dict",
            "model",
            "global_model_state_dict",
        ]
        for key in candidates:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint does not contain a recognizable state_dict.")


def normalize_state_dict_keys(state):
    out = {}
    for k, v in state.items():
        nk = k

        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        out[nk] = v
    return out


def load_finetuned_checkpoint(backbone, checkpoint_path: Path):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )
    state = normalize_state_dict_keys(unwrap_state_dict(ckpt))

    # Sometimes checkpoints were saved from a wrapper that prefixes backbone.
    if any(k.startswith("backbone.") for k in state):
        state = {
            (k[len("backbone."):] if k.startswith("backbone.") else k): v
            for k, v in state.items()
        }

    incompatible = backbone.load_state_dict(state, strict=False)

    print("")
    print("Checkpoint load report")
    print("-" * 80)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Missing    : {len(incompatible.missing_keys)}")
    print(f"Unexpected : {len(incompatible.unexpected_keys)}")

    if incompatible.missing_keys:
        print("First missing keys:")
        for k in incompatible.missing_keys[:10]:
            print(f"  {k}")

    if incompatible.unexpected_keys:
        print("First unexpected keys:")
        for k in incompatible.unexpected_keys[:10]:
            print(f"  {k}")

    print("-" * 80)


def configure_trainable_parameters(
    model: AdaSTE2Wrapper,
    train_last_blocks=2,
    train_classifier=True,
):
    # Freeze complete backbone.
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Scorer is always trainable.
    for p in model.scorer.parameters():
        p.requires_grad = True

    layers = model._encoder_layers()
    n_layers = len(layers)

    if train_last_blocks > 0:
        start = max(0, n_layers - train_last_blocks)
        for layer_idx in range(start, n_layers):
            for p in layers[layer_idx].parameters():
                p.requires_grad = True

    if train_classifier:
        if hasattr(model.backbone, "classifier"):
            for p in model.backbone.classifier.parameters():
                p.requires_grad = True

        # Final layer norm can adapt cheaply.
        if hasattr(model.backbone.timesformer, "layernorm"):
            for p in model.backbone.timesformer.layernorm.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return total, trainable


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------
def evaluate(model, loader, device, use_amp=True):
    model.eval()

    total = 0
    total_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total_active = 0.0
    total_possible = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for videos, labels, _paths in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_amp and device.type == "cuda"),
            ):
                outputs = model(videos)
                logits = outputs.logits
                loss = criterion(logits, labels)

            bsz = labels.size(0)
            c1, c5 = topk_correct(logits, labels, ks=(1, 5))

            total += bsz
            total_loss += float(loss.item()) * bsz
            correct1 += c1
            correct5 += c5

            if model.last_hard_mask is not None:
                total_active += float(model.last_hard_mask.sum().item())
                total_possible += float(model.last_hard_mask.numel())

    retention = total_active / max(total_possible, 1.0)

    return {
        "loss": total_loss / max(total, 1),
        "top1": 100.0 * correct1 / max(total, 1),
        "top5": 100.0 * correct5 / max(total, 1),
        "actual_retention": retention,
        "token_reduction": 1.0 - retention,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    accum_steps=4,
    use_amp=True,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)

    seen = 0
    running_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0

    for step, (videos, labels, _paths) in enumerate(loader, start=1):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(use_amp and device.type == "cuda"),
        ):
            outputs = model(videos)
            logits = outputs.logits
            loss = criterion(logits, labels)
            scaled_loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        bsz = labels.size(0)
        c1, c5 = topk_correct(logits.detach(), labels, ks=(1, 5))

        seen += bsz
        running_loss += float(loss.item()) * bsz
        correct1 += c1
        correct5 += c5

        if step % 20 == 0 or step == len(loader):
            scorer_mean = (
                float(model.last_scores.mean().item())
                if model.last_scores is not None
                else float("nan")
            )
            print(
                f"  batch {step:4d}/{len(loader):4d}"
                f" | loss {running_loss / max(seen,1):.4f}"
                f" | top1 {100.0 * correct1 / max(seen,1):.2f}%"
                f" | score_mean {scorer_mean:.4f}"
            )

    return {
        "loss": running_loss / max(seen, 1),
        "top1": 100.0 * correct1 / max(seen, 1),
        "top5": 100.0 * correct5 / max(seen, 1),
    }


def profile_inference(model, loader, device, use_amp=True, warmup=10, steps=100):
    """
    Measures current gated-model latency. Sequence shape is not physically
    shortened in E2, so this must NOT be interpreted as token-reduction speedup.
    """
    model.eval()

    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch

    with torch.no_grad():
        for _ in range(warmup):
            videos, _labels, _ = next_batch()
            videos = videos.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_amp and device.type == "cuda"),
            ):
                _ = model(videos)
            if device.type == "cuda":
                torch.cuda.synchronize()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        latencies = []

        for _ in range(steps):
            videos, _labels, _ = next_batch()
            videos = videos.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_amp and device.type == "cuda"),
            ):
                _ = model(videos)

            if device.type == "cuda":
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            latencies.append(dt / videos.size(0))

        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

    mean_sec = float(np.mean(latencies))
    return {
        "latency_ms_per_video": 1000.0 * mean_sec,
        "fps": 1.0 / mean_sec if mean_sec > 0 else 0.0,
        "peak_vram_mb": peak_mb,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AdaST E2: learned ST importance + fixed Top-K gating"
    )

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--retention", type=float, default=0.50)
    parser.add_argument("--insert-block", type=int, default=6)
    parser.add_argument("--scorer-hidden", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--lr-scorer", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--train-last-blocks", type=int, default=2)

    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default="runs/e2_fixed_topk_r050")

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --resume and evaluate without training."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional E2 checkpoint generated by this script."
    )

    args = parser.parse_args()

    if not (0.0 < args.retention <= 1.0):
        raise ValueError("--retention must be in (0, 1].")

    seed_everything(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("AdaST E2 — Learned ST Importance + Fixed Top-K Gating")
    print("=" * 88)
    print(f"Device             : {device}")
    if torch.cuda.is_available():
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"Dataset            : UCF101 Split-{args.split}")
    print(f"Frames             : {args.num_frames}")
    print(f"Retention          : {args.retention:.4f}")
    print(f"Token reduction    : {(1.0-args.retention)*100:.2f}% (effective)")
    print(f"Insert block       : {args.insert_block}")
    print(f"Train last blocks  : {args.train_last_blocks}")
    print(f"Batch / accum      : {args.batch_size} / {args.accum_steps}")
    print(f"AMP                : {args.amp}")
    print("=" * 88)

    print("\nLoading image processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Reading UCF101 split...")
    official_train, official_test = read_ucf_split(
        data_root,
        split_root,
        split=args.split
    )

    train_items, val_items = stratified_train_val(
        official_train,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    print(f"Train clips         : {len(train_items)}")
    print(f"Validation clips    : {len(val_items)}")
    print(f"Official test clips : {len(official_test)}")

    train_ds = UCF101TimesformerDataset(
        train_items, processor, num_frames=args.num_frames
    )
    val_ds = UCF101TimesformerDataset(
        val_items, processor, num_frames=args.num_frames
    )
    test_ds = UCF101TimesformerDataset(
        official_test, processor, num_frames=args.num_frames
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print("\nLoading TimeSformer...")
    backbone = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=101,
        ignore_mismatched_sizes=True,
    )

    load_finetuned_checkpoint(backbone, checkpoint_path)

    model = AdaSTE2Wrapper(
        backbone=backbone,
        retention=args.retention,
        insert_block=args.insert_block,
        num_frames=args.num_frames,
        scorer_hidden=args.scorer_hidden,
    )

    total_params, trainable_params = configure_trainable_parameters(
        model,
        train_last_blocks=args.train_last_blocks,
        train_classifier=True,
    )

    print(f"Total parameters    : {total_params/1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params/1e6:.3f} M")

    model.to(device)

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        saved = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False
        )
        state = saved["model_state_dict"] if "model_state_dict" in saved else saved
        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"Loaded E2 resume: {resume_path} "
            f"(missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})"
        )

    scorer_params = list(model.scorer.parameters())

    backbone_trainable = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("scorer.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": scorer_params,
                "lr": args.lr_scorer,
            },
            {
                "params": backbone_trainable,
                "lr": args.lr_backbone,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if args.amp and device.type == "cuda"
        else None
    )

    metrics_csv = output_dir / "metrics.csv"

    if not metrics_csv.exists():
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
                "actual_retention",
                "token_reduction",
                "lr_scorer",
                "lr_backbone",
                "epoch_sec",
                "peak_vram_mb",
            ])

    best_val_top1 = -1.0
    best_path = output_dir / "best.pt"

    if not args.eval_only:
        total_start = time.time()

        for epoch in range(1, args.epochs + 1):
            print("")
            print("=" * 88)
            print(f"EPOCH {epoch}/{args.epochs}")
            print("=" * 88)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                accum_steps=args.accum_steps,
                use_amp=args.amp,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=args.amp,
            )

            epoch_sec = time.time() - epoch_start
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )

            print("")
            print(f"Train loss       : {train_metrics['loss']:.4f}")
            print(f"Train Top-1      : {train_metrics['top1']:.2f}%")
            print(f"Train Top-5      : {train_metrics['top5']:.2f}%")
            print(f"Validation loss  : {val_metrics['loss']:.4f}")
            print(f"Validation Top-1 : {val_metrics['top1']:.2f}%")
            print(f"Validation Top-5 : {val_metrics['top5']:.2f}%")
            print(f"Actual retention : {val_metrics['actual_retention']:.4f}")
            print(f"Token reduction  : {100*val_metrics['token_reduction']:.2f}%")
            print(f"Epoch time       : {epoch_sec:.1f} sec")
            print(f"Peak VRAM        : {peak_mb:.0f} MB")

            with metrics_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch,
                    train_metrics["loss"],
                    train_metrics["top1"],
                    train_metrics["top5"],
                    val_metrics["loss"],
                    val_metrics["top1"],
                    val_metrics["top5"],
                    val_metrics["actual_retention"],
                    val_metrics["token_reduction"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    epoch_sec,
                    peak_mb,
                ])

            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_val_top1": best_val_top1,
                        "retention": args.retention,
                        "insert_block": args.insert_block,
                        "num_frames": args.num_frames,
                        "args": vars(args),
                    },
                    best_path,
                )

                print(
                    f"Saved new best E2 checkpoint "
                    f"(Val Top-1={best_val_top1:.2f}%)"
                )

            scheduler.step()

        total_train_sec = time.time() - total_start
        print(f"\nTraining complete in {total_train_sec/3600:.2f} h")

    if not best_path.exists():
        if args.resume is not None:
            best_path = Path(args.resume)
        else:
            raise RuntimeError(
                "No E2 best checkpoint exists. Train first or provide --resume."
            )

    print("\nLoading best E2 checkpoint for official test evaluation...")
    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False
    )
    state = best["model_state_dict"] if "model_state_dict" in best else best
    model.load_state_dict(state, strict=False)
    model.to(device)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=args.amp,
    )

    profile = profile_inference(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=args.amp,
        warmup=10,
        steps=min(100, max(20, len(test_loader))),
    )

    total_patch_tokens = (
        model.last_total_patch_tokens
        if model.last_total_patch_tokens is not None
        else -1
    )
    retained_tokens = (
        model.last_k
        if model.last_k is not None
        else -1
    )

    summary = {
        "experiment": "AdaST E2 - Learned ST Importance + Fixed Top-K Gating",
        "dataset": f"UCF101 Split-{args.split}",
        "backbone": "TimeSformer-Base K400 -> UCF101",
        "dense_reference_top1": 92.55,
        "dense_reference_top5": 98.97,
        "retention_target": args.retention,
        "actual_retention": test_metrics["actual_retention"],
        "token_reduction_percent": 100.0 * test_metrics["token_reduction"],
        "dense_patch_tokens": total_patch_tokens,
        "retained_patch_tokens": retained_tokens,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "latency_ms_per_video_masked_model": profile["latency_ms_per_video"],
        "fps_masked_model": profile["fps"],
        "peak_vram_mb": profile["peak_vram_mb"],
        "physical_sequence_reduction": False,
        "note": (
            "E2 preserves TimeSformer sequence shape and uses hard "
            "straight-through gating. Runtime is not a physical token "
            "reduction speedup measurement."
        ),
        "checkpoint": str(best_path.resolve()),
        "config": vars(args),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "retention_target",
            "actual_retention",
            "test_top1",
            "test_top5",
            "dense_tokens",
            "retained_tokens",
            "token_reduction_percent",
            "latency_ms_masked_model",
            "fps_masked_model",
            "peak_vram_mb",
        ])
        writer.writerow([
            args.retention,
            test_metrics["actual_retention"],
            test_metrics["top1"],
            test_metrics["top5"],
            total_patch_tokens,
            retained_tokens,
            100.0 * test_metrics["token_reduction"],
            profile["latency_ms_per_video"],
            profile["fps"],
            profile["peak_vram_mb"],
        ])

    print("")
    print("=" * 88)
    print("E2 FIXED TOP-K — FINAL TEST RESULT")
    print("=" * 88)
    print(f"Dataset              : UCF101 Split-{args.split}")
    print(f"Backbone             : TimeSformer")
    print(f"Frames               : {args.num_frames}")
    print(f"Insert block         : {args.insert_block}")
    print(f"Target retention     : {args.retention:.4f}")
    print(f"Actual retention     : {test_metrics['actual_retention']:.4f}")
    print(f"Dense patch tokens   : {total_patch_tokens}")
    print(f"Effective kept tokens: {retained_tokens}")
    print(f"Token reduction      : {100*test_metrics['token_reduction']:.2f}%")
    print("")
    print(f"Test Top-1           : {test_metrics['top1']:.4f}%")
    print(f"Test Top-5           : {test_metrics['top5']:.4f}%")
    print("")
    print(f"Masked-model latency : {profile['latency_ms_per_video']:.3f} ms/video")
    print(f"Masked-model FPS     : {profile['fps']:.3f}")
    print(f"Peak VRAM            : {profile['peak_vram_mb']:.1f} MB")
    print("")
    print("IMPORTANT: E2 uses structured-shape-preserving hard gating.")
    print("Do NOT report its latency as physical token-pruning speedup.")
    print(f"Best checkpoint      : {best_path.resolve()}")
    print(f"Summary JSON         : {summary_path.resolve()}")
    print(f"Summary CSV          : {summary_csv.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
