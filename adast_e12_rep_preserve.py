#!/usr/bin/env python
"""
AdaST E12: Representation-Preserving Physical Compaction

Student:
  - Exact E6-v2 AdaST architecture
  - Initialized from E6-v2 best checkpoint
  - K(X), S(X), scorer and consolidator frozen
  - Compact tail + classifier trainable

Teacher:
  - Dense UCF101 TimeSformer
  - Fully frozen

Loss:
  L = CE
      + lambda_cls * [1 - cosine(student_CLS, teacher_CLS)]
      + lambda_kd * T^2 * KL(teacher || student)

Default:
  lambda_cls = 0.5
  lambda_kd  = 0.5
  T          = 2.0
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
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import adast_e6_v2_pretrained_compact as e6


# ============================================================
# Reproducibility
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Checkpoint helpers
# ============================================================

def extract_state(obj):

    if isinstance(obj, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

    return obj


def normalize_state_dict(sd):

    result = {}

    for key, value in sd.items():

        if key.startswith("module."):
            key = key[7:]

        result[key] = value

    return result


def load_e6_checkpoint(model, checkpoint_path):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nE6-v2 checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Locate it using PowerShell:\n"
            'Get-ChildItem "C:\\Users\\abc\\AdaST" '
            '-Recurse -Filter "best.pt" | '
            "Select-Object -ExpandProperty FullName\n"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = extract_state(checkpoint)
    state = normalize_state_dict(state)

    result = model.load_state_dict(
        state,
        strict=True,
    )

    print("[OK] E6-v2 checkpoint loaded")
    print("     ", checkpoint_path)

    print(
        f"     missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}"
    )

    return checkpoint


# ============================================================
# CLS representation capture
# ============================================================

class FeatureCapture:

    def __init__(self, module):

        self.value = None

        self.handle = (
            module.register_forward_pre_hook(
                self._hook
            )
        )

    def _hook(self, module, inputs):

        self.value = inputs[0]

    def close(self):

        self.handle.remove()


# ============================================================
# Student configuration
# ============================================================

def configure_student(model):

    # Freeze complete model first
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train compact attention tail
    for parameter in model.compact_tail.parameters():
        parameter.requires_grad = True

    # Train classification head
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad = True

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[OK] Total parameters     : "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"[OK] Trainable parameters : "
        f"{trainable_params / 1e6:.3f} M"
    )

    print(
        "[OK] AdaST policy + consolidator frozen"
    )

    print(
        "[OK] Compact tail + classifier trainable"
    )


def set_student_train_mode(model):
    """
    Keep the overall AdaST wrapper in eval mode.

    This makes the already learned/frozen adaptive policy
    deterministic during E12.

    eval() does not disable gradients.

    Compact tail and classifier remain trainable.
    """

    model.eval()

    model.compact_tail.train()

    model.backbone.classifier.train()


# ============================================================
# Dataset
# ============================================================

def build_loaders(args, processor):

    train_all, test_items = e6.read_split(
        Path(args.data_root),
        Path(args.split_root),
        args.split,
    )

    train_items, val_items = e6.stratified(
        train_all,
        args.val_fraction,
        args.seed,
    )

    # IMPORTANT:
    # Exact E6 UCF constructor has no train= argument.

    train_dataset = e6.UCF(
        train_items,
        processor,
        args.num_frames,
    )

    val_dataset = e6.UCF(
        val_items,
        processor,
        args.num_frames,
    )

    test_dataset = e6.UCF(
        test_items,
        processor,
        args.num_frames,
    )

    common = dict(
        batch_size=1,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )

    print(
        f"Train={len(train_dataset)} | "
        f"Val={len(val_dataset)} | "
        f"Test={len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


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


# ============================================================
# E12 training
# ============================================================

def train_one_epoch(
    student,
    teacher,
    loader,
    optimizer,
    scaler,
    device,
    student_capture,
    teacher_capture,
    args,
):

    set_student_train_mode(student)

    teacher.eval()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_ce = 0.0
    total_cls = 0.0
    total_kd = 0.0

    total_correct = 0
    total_samples = 0

    start_time = time.time()

    for step, batch in enumerate(
        loader,
        start=1,
    ):

        x, y, _ = unpack_batch(
            batch,
            device,
        )

        # ====================================================
        # Frozen dense teacher
        # ====================================================

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    args.amp
                    and device.type == "cuda"
                ),
            ):

                teacher_output = teacher(
                    pixel_values=x,
                    return_dict=True,
                )

                teacher_logits = (
                    teacher_output.logits
                )

                teacher_cls = (
                    teacher_capture.value
                )

        if teacher_cls is None:
            raise RuntimeError(
                "Teacher CLS feature was not captured."
            )

        teacher_logits = teacher_logits.detach()
        teacher_cls = teacher_cls.detach()

        # ====================================================
        # AdaST compact student
        # ====================================================

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                args.amp
                and device.type == "cuda"
            ),
        ):

            student_output = student(x)

            student_logits = (
                student_output.logits
            )

            student_cls = (
                student_capture.value
            )

            if student_cls is None:
                raise RuntimeError(
                    "Student CLS feature was not captured."
                )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            ce_loss = F.cross_entropy(
                student_logits,
                y,
            )

            # ------------------------------------------------
            # Representation preservation
            # ------------------------------------------------

            cls_loss = (
                1.0
                -
                F.cosine_similarity(
                    student_cls.float(),
                    teacher_cls.float(),
                    dim=-1,
                ).mean()
            )

            # ------------------------------------------------
            # Logit distillation
            # ------------------------------------------------

            temperature = (
                args.distill_temp
            )

            student_log_prob = (
                F.log_softmax(
                    student_logits.float()
                    / temperature,
                    dim=-1,
                )
            )

            teacher_prob = (
                F.softmax(
                    teacher_logits.float()
                    / temperature,
                    dim=-1,
                )
            )

            kd_loss = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction="batchmean",
            )

            kd_loss = (
                kd_loss
                *
                temperature
                *
                temperature
            )

            # ------------------------------------------------
            # Total E12 loss
            # ------------------------------------------------

            loss = (
                ce_loss
                +
                args.lambda_cls
                * cls_loss
                +
                args.lambda_kd
                * kd_loss
            )

            backward_loss = (
                loss
                /
                args.accum_steps
            )

        scaler.scale(
            backward_loss
        ).backward()

        if (
            step % args.accum_steps == 0
            or
            step == len(loader)
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_size = y.numel()

        total_loss += (
            loss.item()
            * batch_size
        )

        total_ce += (
            ce_loss.item()
            * batch_size
        )

        total_cls += (
            cls_loss.item()
            * batch_size
        )

        total_kd += (
            kd_loss.item()
            * batch_size
        )

        prediction = (
            student_logits.argmax(
                dim=1
            )
        )

        total_correct += (
            prediction.eq(y)
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

        if (
            step % args.print_every == 0
            or
            step == len(loader)
        ):

            print(
                f"step "
                f"{step:5d}/"
                f"{len(loader)} | "
                f"loss="
                f"{total_loss/total_samples:.4f} | "
                f"CE="
                f"{total_ce/total_samples:.4f} | "
                f"CLS="
                f"{total_cls/total_samples:.4f} | "
                f"KD="
                f"{total_kd/total_samples:.4f} | "
                f"Top1="
                f"{100*total_correct/total_samples:.2f}%"
            )

    elapsed = (
        time.time()
        -
        start_time
    )

    return {
        "loss":
            total_loss
            / total_samples,

        "ce":
            total_ce
            / total_samples,

        "cls":
            total_cls
            / total_samples,

        "kd":
            total_kd
            / total_samples,

        "top1":
            100.0
            * total_correct
            / total_samples,

        "seconds":
            elapsed,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    args,
    collect=False,
):

    model.eval()

    total = 0
    correct1 = 0
    correct5 = 0

    spatial_r_list = []
    spatial_k_list = []
    temporal_s_list = []

    effective_retention_list = []
    compact_token_list = []

    rows = []

    for batch in loader:

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

            output = model(x)

            logits = (
                output.logits
            )

        prediction = logits.argmax(
            dim=1
        )

        total += y.numel()

        correct1 += (
            prediction.eq(y)
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
            .any(dim=1)
            .sum()
            .item()
        )

        spatial_r = float(
            getattr(
                model,
                "last_r",
                np.nan,
            )
        )

        spatial_k = float(
            getattr(
                model,
                "last_k",
                np.nan,
            )
        )

        temporal_s = float(
            getattr(
                model,
                "last_s",
                np.nan,
            )
        )

        compact_tokens = float(
            getattr(
                model,
                "last_compact_tokens",
                np.nan,
            )
        )

        if np.isfinite(
            spatial_r
        ):
            spatial_r_list.append(
                spatial_r
            )

        if np.isfinite(
            spatial_k
        ):
            spatial_k_list.append(
                spatial_k
            )

        if np.isfinite(
            temporal_s
        ):
            temporal_s_list.append(
                temporal_s
            )

        if np.isfinite(
            compact_tokens
        ):
            compact_token_list.append(
                compact_tokens
            )

        effective_retention = (
            spatial_r
            *
            (
                temporal_s
                /
                args.num_frames
            )
        )

        if np.isfinite(
            effective_retention
        ):
            effective_retention_list.append(
                effective_retention
            )

        if collect:

            video_path = ""

            if paths is not None:

                if isinstance(
                    paths,
                    (list, tuple),
                ):
                    video_path = str(
                        paths[0]
                    )

                else:
                    video_path = str(
                        paths
                    )

            rows.append(
                {
                    "video":
                        video_path,

                    "label":
                        int(
                            y[0].item()
                        ),

                    "prediction":
                        int(
                            prediction[0]
                            .item()
                        ),

                    "correct":
                        int(
                            prediction[0]
                            .item()
                            ==
                            y[0].item()
                        ),

                    "spatial_r":
                        spatial_r,

                    "k_per_frame":
                        spatial_k,

                    "temporal_S":
                        temporal_s,

                    "combined_effective_retention":
                        effective_retention,

                    "compact_patch_tokens":
                        compact_tokens,
                }
            )

    result = {

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
                np.mean(
                    spatial_r_list
                )
            ),

        "mean_spatial_k":
            float(
                np.mean(
                    spatial_k_list
                )
            ),

        "mean_temporal_S":
            float(
                np.mean(
                    temporal_s_list
                )
            ),

        "mean_effective_retention":
            float(
                np.mean(
                    effective_retention_list
                )
            ),

        "mean_effective_reduction_pct":
            100.0
            *
            (
                1.0
                -
                float(
                    np.mean(
                        effective_retention_list
                    )
                )
            ),

        "mean_physical_compact_patch_tokens":
            float(
                np.mean(
                    compact_token_list
                )
            ),
    }

    return result, rows


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--split-root",
        required=True,
    )

    parser.add_argument(
        "--dense-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--e6-checkpoint",
        required=True,
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
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--distill-temp",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=(
            "runs/"
            "e12_rep_preserve"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Early validation
    # --------------------------------------------------------

    if args.batch_size != 1:

        raise ValueError(
            "Physical AdaST E6-v2/E12 "
            "requires --batch-size 1"
        )

    for name, path in (
        (
            "Dense checkpoint",
            args.dense_checkpoint,
        ),
        (
            "E6 checkpoint",
            args.e6_checkpoint,
        ),
    ):

        if not Path(path).exists():

            raise FileNotFoundError(
                f"{name} does not exist:\n"
                f"{path}"
            )

    seed_all(
        args.seed
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 76
    )

    print(
        "AdaST E12 | "
        "Representation-Preserving "
        "Physical Compaction"
    )

    print(
        "=" * 76
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        f"Loss = CE + "
        f"{args.lambda_cls}*CLS + "
        f"{args.lambda_kd}*KD"
        f"(T={args.distill_temp})"
    )

    # ========================================================
    # Data
    # ========================================================

    processor = (
        AutoImageProcessor
        .from_pretrained(
            args.model_name
        )
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders(
        args,
        processor,
    )

    # ========================================================
    # Student
    # ========================================================

    print()
    print(
        "Building E6-v2 student..."
    )

    student_backbone = (
        TimesformerForVideoClassification
        .from_pretrained(
            args.model_name,
            num_labels=101,
            ignore_mismatched_sizes=True,
        )
    )

    student = e6.AdaSTE5(
        student_backbone,
        block=args.insert_block,
        T=args.num_frames,
        temperature=args.temperature,
    )

    student.enable_physical_tail()

    load_e6_checkpoint(
        student,
        args.e6_checkpoint,
    )

    configure_student(
        student
    )

    student.to(
        device
    )

    # ========================================================
    # Teacher
    # ========================================================

    print()
    print(
        "Building dense teacher..."
    )

    teacher = (
        TimesformerForVideoClassification
        .from_pretrained(
            args.model_name,
            num_labels=101,
            ignore_mismatched_sizes=True,
        )
    )

    e6.load_dense(
        teacher,
        args.dense_checkpoint,
    )

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    teacher.eval()

    teacher.to(
        device
    )

    print(
        "[OK] Dense teacher frozen"
    )

    # ========================================================
    # Capture teacher/student final CLS representations
    # ========================================================

    student_capture = FeatureCapture(
        student.backbone.classifier
    )

    teacher_capture = FeatureCapture(
        teacher.classifier
    )

    # ========================================================
    # Optimizer
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter
        in student.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            args.amp
            and device.type == "cuda"
        ),
    )

    best_val_top1 = -1.0

    history = []

    best_path = (
        output_dir
        /
        "best.pt"
    )

    try:

        # ====================================================
        # Training
        # ====================================================

        for epoch in range(
            1,
            args.epochs + 1,
        ):

            print()
            print(
                "=" * 76
            )

            print(
                f"EPOCH "
                f"{epoch}/"
                f"{args.epochs}"
            )

            print(
                "=" * 76
            )

            train_result = (
                train_one_epoch(
                    student,
                    teacher,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    student_capture,
                    teacher_capture,
                    args,
                )
            )

            validation_result, _ = (
                evaluate(
                    student,
                    val_loader,
                    device,
                    args,
                    collect=False,
                )
            )

            row = {

                "epoch":
                    epoch,

                "train_loss":
                    train_result[
                        "loss"
                    ],

                "train_ce":
                    train_result[
                        "ce"
                    ],

                "train_cls":
                    train_result[
                        "cls"
                    ],

                "train_kd":
                    train_result[
                        "kd"
                    ],

                "train_top1":
                    train_result[
                        "top1"
                    ],

                "val_top1":
                    validation_result[
                        "top1"
                    ],

                "val_top5":
                    validation_result[
                        "top5"
                    ],

                "seconds":
                    train_result[
                        "seconds"
                    ],
            }

            history.append(
                row
            )

            print()
            print(
                f"Epoch {epoch}: "
                f"Val Top-1 = "
                f"{validation_result['top1']:.4f}%"
            )

            print(
                f"Epoch {epoch}: "
                f"Val Top-5 = "
                f"{validation_result['top5']:.4f}%"
            )

            if (
                validation_result[
                    "top1"
                ]
                >
                best_val_top1
            ):

                best_val_top1 = (
                    validation_result[
                        "top1"
                    ]
                )

                torch.save(
                    {
                        "model_state_dict":
                            student.state_dict(),

                        "epoch":
                            epoch,

                        "val":
                            validation_result,

                        "args":
                            vars(args),
                    },
                    best_path,
                )

                print(
                    "[BEST] Saved:",
                    best_path,
                )

        # ====================================================
        # Save training history
        # ====================================================

        save_csv(
            output_dir
            /
            "metrics.csv",
            history,
        )

        # ====================================================
        # Reload best validation checkpoint
        # ====================================================

        best_checkpoint = torch.load(
            best_path,
            map_location="cpu",
        )

        student.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        student.to(
            device
        )

        student.eval()

        # ====================================================
        # Official UCF101 test
        # ====================================================

        print()
        print(
            "=" * 76
        )

        print(
            "OFFICIAL UCF101 "
            "SPLIT-1 TEST"
        )

        print(
            "=" * 76
        )

        test_result, test_rows = (
            evaluate(
                student,
                test_loader,
                device,
                args,
                collect=True,
            )
        )

        save_csv(
            output_dir
            /
            "per_video_test.csv",
            test_rows,
        )

        # ====================================================
        # Summary
        # ====================================================

        summary = {

            "experiment":
                "E12_representation_"
                "preserving_"
                "physical_compaction",

            "student_checkpoint":
                str(
                    args.e6_checkpoint
                ),

            "dense_teacher_checkpoint":
                str(
                    args.dense_checkpoint
                ),

            "loss": {

                "lambda_cls":
                    args.lambda_cls,

                "lambda_kd":
                    args.lambda_kd,

                "distill_temperature":
                    args.distill_temp,
            },

            "best_val_top1":
                best_val_top1,

            "test":
                test_result,

            "policy_frozen":
                True,

            "consolidator_frozen":
                True,

            "physical_sequence_compaction":
                True,

            "trainable_modules":
                (
                    "compact_tail "
                    "+ classifier"
                ),
        }

        summary_path = (
            output_dir
            /
            "summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
            )

        # ====================================================
        # Final terminal report
        # ====================================================

        print()
        print(
            "Samples                      :",
            test_result[
                "samples"
            ],
        )

        print(
            "Top-1                        : "
            f"{test_result['top1']:.4f}%"
        )

        print(
            "Top-5                        : "
            f"{test_result['top5']:.4f}%"
        )

        print(
            "Mean spatial retention       : "
            f"{test_result['mean_spatial_retention']:.6f}"
        )

        print(
            "Mean spatial K/frame         : "
            f"{test_result['mean_spatial_k']:.4f}"
        )

        print(
            "Mean temporal S              : "
            f"{test_result['mean_temporal_S']:.4f}"
        )

        print(
            "Mean effective retention     : "
            f"{test_result['mean_effective_retention']:.6f}"
        )

        print(
            "Mean effective reduction     : "
            f"{test_result['mean_effective_reduction_pct']:.4f}%"
        )

        print(
            "Mean physical compact tokens : "
            f"{test_result['mean_physical_compact_patch_tokens']:.2f}"
        )

        print()
        print(
            "Best checkpoint:",
            best_path,
        )

        print(
            "Summary:",
            summary_path,
        )

        print(
            "=" * 76
        )

    finally:

        student_capture.close()

        teacher_capture.close()


if __name__ == "__main__":
    main()