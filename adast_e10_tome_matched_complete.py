import os
import math
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
)


# ============================================================
# AdaST E10
# ToMe-Style Controlled Matched Baseline
# UCF101 Split-1
# ============================================================

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"

NUM_CLASSES = 101
NUM_FRAMES = 8
NUM_PATCHES = 196
DIM = 768

PARTITION = 6

# ------------------------------------------------------------
# IMPORTANT
# Original ST-EdgeViT controlled baseline used 74 trajectories.
# Keep this value for reproducing that controlled setting.
# We can subsequently run a budget-matched AdaST comparison.
# ------------------------------------------------------------

TARGET_TOKENS = 93

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\abc\ST-EdgeViT\datasets\UCF101"

SPLIT_DIR = r"C:\Users\abc\AdaST\splits\ucfTrainTestlist"

CLASS_FILE = os.path.join(
    SPLIT_DIR,
    "classInd.txt"
)

TEST_FILE = os.path.join(
    SPLIT_DIR,
    "testlist01.txt"
)

CHECKPOINT = r"C:\Users\abc\AdaST\checkpoints\timesformer_ucf101_best.pt"

TEST_FILE = os.path.join(
    SPLIT_DIR,
    "testlist01.txt"
)


# ============================================================
# IMPORTANT:
# Put SAME dense TimeSformer checkpoint used by AdaST here.
#
# Change ONLY this path if your checkpoint has another name.
# ============================================================

CHECKPOINT = r"C:\Users\abc\AdaST\checkpoints\timesformer_ucf101_best.pt"


# ============================================================
# UCF101 DATASET
# Same basic protocol used in AdaST:
#
# - official UCF101 Split-1
# - PyAV decoding
# - uniform frame sampling
# - 8 frames
# - HuggingFace TimeSformer processor
# ============================================================

class UCF101Dataset(Dataset):

    def __init__(
        self,
        root_dir,
        split_file,
        class_index_file,
        model_name,
        num_frames=8
    ):

        self.root_dir = Path(root_dir)

        self.num_frames = num_frames

        self.processor = (
            AutoImageProcessor
            .from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # Class mapping
        # ----------------------------------------------------

        self.class_map = {}

        with open(
            class_index_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                idx, class_name = (
                    line.split(
                        maxsplit=1
                    )
                )

                self.class_map[
                    class_name
                ] = int(idx) - 1

        # ----------------------------------------------------
        # Read official test split
        # ----------------------------------------------------

        self.samples = []

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                rel = (
                    line
                    .split()[0]
                    .replace(
                        "/",
                        os.sep
                    )
                )

                class_name = (
                    Path(rel)
                    .parts[0]
                )

                label = (
                    self.class_map[
                        class_name
                    ]
                )

                path = (
                    self.root_dir
                    /
                    rel
                )

                self.samples.append(
                    (
                        path,
                        label
                    )
                )

        print(
            f"Loaded test samples : "
            f"{len(self.samples)}"
        )


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def decode_video(
        self,
        path
    ):

        try:

            container = av.open(
                str(path)
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not open:\n"
                f"{path}\n"
                f"{e}"
            ) from e

        frames = []

        try:

            stream = (
                container
                .streams
                .video[0]
            )

            stream.thread_type = (
                "AUTO"
            )

            for frame in container.decode(
                stream
            ):

                frames.append(
                    frame.to_ndarray(
                        format="rgb24"
                    )
                )

        finally:

            container.close()

        if not frames:

            raise RuntimeError(
                f"No frames decoded: "
                f"{path}"
            )

        # ----------------------------------------------------
        # Uniform sampling
        # ----------------------------------------------------

        indices = np.linspace(
            0,
            len(frames) - 1,
            self.num_frames
        )

        indices = (
            indices
            .round()
            .astype(int)
        )

        selected = [
            frames[i]
            for i in indices
        ]

        return selected


    def __getitem__(
        self,
        index
    ):

        path, label = (
            self.samples[
                index
            ]
        )

        frames = (
            self.decode_video(
                path
            )
        )

        pixel_values = (
            self.processor(
                images=frames,
                return_tensors="pt"
            )
            ["pixel_values"]
            .squeeze(0)
        )

        return {
            "pixel_values":
                pixel_values,

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long
                ),

            "path":
                str(path)
        }


# ============================================================
# Accuracy
# ============================================================

def topk_correct(
    logits,
    labels,
    topk=(1, 5)
):

    maxk = max(
        topk
    )

    _, pred = logits.topk(
        maxk,
        dim=1,
        largest=True,
        sorted=True
    )

    pred = pred.t()

    correct = pred.eq(
        labels
        .view(
            1,
            -1
        )
        .expand_as(
            pred
        )
    )

    values = []

    for k in topk:

        value = (
            correct[:k]
            .reshape(-1)
            .float()
            .sum()
            .item()
        )

        values.append(
            value
        )

    return values


# ============================================================
# Load checkpoint robustly
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    print(
        "\nLoading checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Common checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif (
            "model"
            in checkpoint
            and
            isinstance(
                checkpoint["model"],
                dict
            )
        ):

            state = (
                checkpoint[
                    "model"
                ]
            )

        else:

            state = checkpoint

    else:

        state = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned = {}

    for key, value in state.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = (
                new_key[
                    len("module.") :
                ]
            )

        cleaned[
            new_key
        ] = value

    result = (
        model.load_state_dict(
            cleaned,
            strict=False
        )
    )

    print(
        "Missing keys    :",
        len(
            result.missing_keys
        )
    )

    print(
        "Unexpected keys :",
        len(
            result.unexpected_keys
        )
    )

    return model


# ============================================================
# Edge forward
# First P TimeSformer blocks
# ============================================================

def edge_forward(
    model,
    pixel_values,
    partition
):

    hidden = (
        model
        .timesformer
        .embeddings(
            pixel_values
        )
    )

    for layer_idx in range(
        partition
    ):

        layer = (
            model
            .timesformer
            .encoder
            .layer[
                layer_idx
            ]
        )

        outputs = layer(
            hidden,
            output_attentions=False
        )

        hidden = (
            outputs[0]
        )

    return hidden


# ============================================================
# ONE ToMe-STYLE BIPARTITE MERGING ROUND
#
# trajectories:
#
# [B, N, T, D]
#
# Alternating source/destination trajectories.
# Similarity calculated from temporally pooled descriptors.
# ============================================================

def tome_merge_round(
    trajectories,
    sizes,
    target_count
):

    batch_size = (
        trajectories.shape[0]
    )

    num_tokens = (
        trajectories.shape[1]
    )

    if (
        target_count
        >=
        num_tokens
    ):

        return (
            trajectories,
            sizes
        )

    num_merges = (
        num_tokens
        -
        target_count
    )

    # --------------------------------------------------------
    # Alternating bipartite partition
    # --------------------------------------------------------

    src_indices = torch.arange(
        0,
        num_tokens,
        2,
        device=trajectories.device
    )

    dst_indices = torch.arange(
        1,
        num_tokens,
        2,
        device=trajectories.device
    )

    # Safety
    max_merges = min(
        num_merges,
        src_indices.numel()
    )

    if (
        max_merges <= 0
    ):

        return (
            trajectories,
            sizes
        )

    outputs = []

    output_sizes = []

    # --------------------------------------------------------
    # Batch-wise merge
    # --------------------------------------------------------

    for b in range(
        batch_size
    ):

        x = trajectories[
            b
        ]

        sz = sizes[
            b
        ]

        src = x[
            src_indices
        ]

        dst = x[
            dst_indices
        ]

        src_size = sz[
            src_indices
        ]

        dst_size = sz[
            dst_indices
        ]

        # ----------------------------------------------------
        # Temporal trajectory descriptor
        # ----------------------------------------------------

        src_descriptor = (
            src.mean(
                dim=1
            )
        )

        dst_descriptor = (
            dst.mean(
                dim=1
            )
        )

        src_descriptor = (
            torch.nn.functional
            .normalize(
                src_descriptor,
                dim=-1
            )
        )

        dst_descriptor = (
            torch.nn.functional
            .normalize(
                dst_descriptor,
                dim=-1
            )
        )

        similarity = (
            src_descriptor
            @
            dst_descriptor.t()
        )

        best_similarity, best_dst = (
            similarity.max(
                dim=1
            )
        )

        # ----------------------------------------------------
        # Highest similarity sources are merged
        # ----------------------------------------------------

        merge_src = torch.topk(
            best_similarity,
            k=max_merges,
            largest=True
        ).indices

        merge_src_set = set(
            merge_src
            .detach()
            .cpu()
            .tolist()
        )

        # Clone destinations because they receive merges
        new_dst = (
            dst.clone()
        )

        new_dst_size = (
            dst_size.clone()
        )

        # ----------------------------------------------------
        # Size-weighted merging
        # ----------------------------------------------------

        for source_local_idx in (
            merge_src.tolist()
        ):

            destination_local_idx = int(
                best_dst[
                    source_local_idx
                ].item()
            )

            s_size = (
                src_size[
                    source_local_idx
                ]
            )

            d_size = (
                new_dst_size[
                    destination_local_idx
                ]
            )

            total_size = (
                s_size
                +
                d_size
            )

            merged = (
                new_dst[
                    destination_local_idx
                ]
                *
                d_size
                +
                src[
                    source_local_idx
                ]
                *
                s_size
            ) / total_size

            new_dst[
                destination_local_idx
            ] = merged

            new_dst_size[
                destination_local_idx
            ] = total_size

        # ----------------------------------------------------
        # Keep unmerged source trajectories
        # ----------------------------------------------------

        keep_src_local = [
            i
            for i in range(
                src.shape[0]
            )
            if i not in merge_src_set
        ]

        if len(
            keep_src_local
        ) > 0:

            keep_idx = torch.tensor(
                keep_src_local,
                device=x.device,
                dtype=torch.long
            )

            kept_src = src[
                keep_idx
            ]

            kept_src_size = (
                src_size[
                    keep_idx
                ]
            )

            merged_tokens = torch.cat(
                [
                    new_dst,
                    kept_src
                ],
                dim=0
            )

            merged_sizes = torch.cat(
                [
                    new_dst_size,
                    kept_src_size
                ],
                dim=0
            )

        else:

            merged_tokens = (
                new_dst
            )

            merged_sizes = (
                new_dst_size
            )

        # ----------------------------------------------------
        # Odd token not included in destination partition
        #
        # With N odd, last even-index token belongs to src,
        # therefore it is already handled above.
        # ----------------------------------------------------

        outputs.append(
            merged_tokens
        )

        output_sizes.append(
            merged_sizes
        )

    trajectories = (
        torch.stack(
            outputs,
            dim=0
        )
    )

    sizes = (
        torch.stack(
            output_sizes,
            dim=0
        )
    )

    return (
        trajectories,
        sizes
    )


# ============================================================
# ToMe-style progressive trajectory merging
#
# 196 -> 98 -> target
# ============================================================

def tome_style_merge(
    hidden
):

    batch_size = (
        hidden.shape[0]
    )

    hidden_dim = (
        hidden.shape[-1]
    )

    cls_token = hidden[
        :,
        :1,
        :
    ]

    patches = hidden[
        :,
        1:,
        :
    ]

    expected = (
        NUM_PATCHES
        *
        NUM_FRAMES
    )

    if (
        patches.shape[1]
        !=
        expected
    ):

        raise RuntimeError(
            f"Unexpected patch count "
            f"{patches.shape[1]}, "
            f"expected {expected}"
        )

    # --------------------------------------------------------
    # Same trajectory organization used by controlled baseline
    # [B, N, T, D]
    # --------------------------------------------------------

    trajectories = (
        patches.reshape(
            batch_size,
            NUM_PATCHES,
            NUM_FRAMES,
            hidden_dim
        )
    )

    sizes = torch.ones(
        batch_size,
        NUM_PATCHES,
        device=hidden.device,
        dtype=hidden.dtype
    )

    # --------------------------------------------------------
    # Progressive rounds
    # --------------------------------------------------------

    while (
        trajectories.shape[1]
        >
        TARGET_TOKENS
    ):

        current_count = (
            trajectories.shape[1]
        )

        next_target = max(
            TARGET_TOKENS,
            math.ceil(
                current_count
                /
                2
            )
        )

        trajectories, sizes = (
            tome_merge_round(
                trajectories,
                sizes,
                next_target
            )
        )

    final_k = (
        trajectories.shape[1]
    )

    if (
        final_k
        !=
        TARGET_TOKENS
    ):

        raise RuntimeError(
            f"Final trajectory count "
            f"{final_k}, "
            f"expected "
            f"{TARGET_TOKENS}"
        )

    compact_patches = (
        trajectories.reshape(
            batch_size,
            final_k
            *
            NUM_FRAMES,
            hidden_dim
        )
    )

    compact_hidden = (
        torch.cat(
            [
                cls_token,
                compact_patches
            ],
            dim=1
        )
    )

    return (
        compact_hidden,
        final_k
    )


# ============================================================
# Compact TimeSformer layer
# ============================================================

def compact_timesformer_layer(
    layer,
    hidden_states,
    num_spatial_tokens
):

    batch_size = (
        hidden_states.shape[0]
    )

    hidden_dim = (
        hidden_states.shape[-1]
    )

    expected_tokens = (
        1
        +
        num_spatial_tokens
        *
        NUM_FRAMES
    )

    if (
        hidden_states.shape[1]
        !=
        expected_tokens
    ):

        raise RuntimeError(
            "Unexpected compact hidden "
            f"shape: "
            f"{hidden_states.shape}"
        )

    init_cls_token = (
        hidden_states[
            :,
            :1,
            :
        ]
    )

    hidden_patches = (
        hidden_states[
            :,
            1:,
            :
        ]
    )

    # ========================================================
    # TEMPORAL ATTENTION
    # ========================================================

    temporal_embedding = (
        hidden_patches.reshape(
            batch_size,
            num_spatial_tokens,
            NUM_FRAMES,
            hidden_dim
        )
    )

    temporal_embedding = (
        temporal_embedding.reshape(
            batch_size
            *
            num_spatial_tokens,
            NUM_FRAMES,
            hidden_dim
        )
    )

    temporal_norm = (
        layer
        .temporal_layernorm(
            temporal_embedding
        )
    )

    temporal_outputs = (
        layer
        .temporal_attention(
            temporal_norm,
            output_attentions=False
        )
    )

    temporal_output = (
        temporal_outputs[0]
    )

    temporal_output = (
        layer
        .temporal_dense(
            temporal_output
        )
    )

    temporal_embedding = (
        temporal_embedding
        +
        layer.drop_path(
            temporal_output
        )
    )

    temporal_embedding = (
        temporal_embedding.reshape(
            batch_size,
            num_spatial_tokens,
            NUM_FRAMES,
            hidden_dim
        )
    )

    temporal_embedding = (
        temporal_embedding.reshape(
            batch_size,
            num_spatial_tokens
            *
            NUM_FRAMES,
            hidden_dim
        )
    )

    # ========================================================
    # SPATIAL ATTENTION
    # ========================================================

    cls_token = (
        init_cls_token
        .repeat(
            1,
            NUM_FRAMES,
            1
        )
    )

    cls_token = (
        cls_token.reshape(
            batch_size
            *
            NUM_FRAMES,
            1,
            hidden_dim
        )
    )

    spatial_embedding = (
        temporal_embedding.reshape(
            batch_size,
            num_spatial_tokens,
            NUM_FRAMES,
            hidden_dim
        )
    )

    spatial_embedding = (
        spatial_embedding.permute(
            0,
            2,
            1,
            3
        )
    )

    spatial_embedding = (
        spatial_embedding.reshape(
            batch_size
            *
            NUM_FRAMES,
            num_spatial_tokens,
            hidden_dim
        )
    )

    spatial_embedding = torch.cat(
        [
            cls_token,
            spatial_embedding
        ],
        dim=1
    )

    spatial_outputs = (
        layer.attention(
            layer.layernorm_before(
                spatial_embedding
            ),
            output_attentions=False
        )
    )

    spatial_output = (
        spatial_outputs[0]
    )

    residual_spatial = (
        layer.drop_path(
            spatial_output
        )
    )

    # --------------------------------------------------------
    # CLS residual
    # --------------------------------------------------------

    cls_residual = (
        residual_spatial[
            :,
            0,
            :
        ]
    )

    cls_residual = (
        cls_residual.reshape(
            batch_size,
            NUM_FRAMES,
            hidden_dim
        )
    )

    cls_residual = (
        torch.mean(
            cls_residual,
            dim=1,
            keepdim=True
        )
    )

    # --------------------------------------------------------
    # Patch residual
    # --------------------------------------------------------

    residual_spatial = (
        residual_spatial[
            :,
            1:,
            :
        ]
    )

    residual_spatial = (
        residual_spatial.reshape(
            batch_size,
            NUM_FRAMES,
            num_spatial_tokens,
            hidden_dim
        )
    )

    residual_spatial = (
        residual_spatial.permute(
            0,
            2,
            1,
            3
        )
    )

    residual_spatial = (
        residual_spatial.reshape(
            batch_size,
            num_spatial_tokens
            *
            NUM_FRAMES,
            hidden_dim
        )
    )

    hidden_states = (
        torch.cat(
            [
                init_cls_token,
                temporal_embedding
            ],
            dim=1
        )
        +
        torch.cat(
            [
                cls_residual,
                residual_spatial
            ],
            dim=1
        )
    )

    # ========================================================
    # MLP
    # ========================================================

    layer_output = (
        layer.layernorm_after(
            hidden_states
        )
    )

    layer_output = (
        layer.intermediate(
            layer_output
        )
    )

    layer_output = (
        layer.output(
            layer_output
        )
    )

    layer_output = (
        hidden_states
        +
        layer.drop_path(
            layer_output
        )
    )

    return layer_output


# ============================================================
# Remaining TimeSformer blocks
# ============================================================

def cloud_forward(
    model,
    compact_hidden,
    partition,
    num_spatial_tokens
):

    hidden = (
        compact_hidden
    )

    for layer_idx in range(
        partition,
        model.config.num_hidden_layers
    ):

        layer = (
            model
            .timesformer
            .encoder
            .layer[
                layer_idx
            ]
        )

        hidden = (
            compact_timesformer_layer(
                layer,
                hidden,
                num_spatial_tokens
            )
        )

    hidden = (
        model
        .timesformer
        .layernorm(
            hidden
        )
    )

    cls_representation = (
        hidden[
            :,
            0,
            :
        ]
    )

    logits = (
        model.classifier(
            cls_representation
        )
    )

    return logits


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "AdaST E10"
    )

    print(
        "ToMe-Style Controlled Baseline "
        "-- UCF101 Split-1"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Check files BEFORE model loading
    # --------------------------------------------------------

    required = [
        (
            "Dataset",
            DATASET_DIR
        ),
        (
            "Class index",
            CLASS_FILE
        ),
        (
            "Test split",
            TEST_FILE
        ),
        (
            "Dense checkpoint",
            CHECKPOINT
        )
    ]

    for name, path in required:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"\n{name} not found:\n"
                f"{path}\n\n"
                f"Please correct the path "
                f"at the top of this script."
            )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:"
    )

    print(
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "\nConfiguration:"
    )

    print(
        "Partition P          :",
        PARTITION
    )

    print(
        "Initial trajectories :",
        NUM_PATCHES
    )

    print(
        "Target trajectories  :",
        TARGET_TOKENS
    )

    print(
        "Frames               :",
        NUM_FRAMES
    )

    print(
        "Precision            : FP32"
    )

    print(
        "Selection            : "
        "similarity-based "
        "trajectory merging"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = UCF101Dataset(
        root_dir=DATASET_DIR,
        split_file=TEST_FILE,
        class_index_file=CLASS_FILE,
        model_name=MODEL_NAME,
        num_frames=NUM_FRAMES
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(
        "\nTest videos:",
        len(dataset)
    )

    if (
        len(dataset)
        !=
        3783
    ):

        print(
            "WARNING: Expected "
            "3,783 videos for "
            "official UCF101 Split-1."
        )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nLoading TimeSformer..."
    )

    model = (
        TimesformerForVideoClassification
        .from_pretrained(
            MODEL_NAME,
            num_frames=NUM_FRAMES,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    )

    model = load_checkpoint(
        model,
        CHECKPOINT
    )

    model.to(
        device
    )

    model.eval()

    # ========================================================
    # Evaluation
    # ========================================================

    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    final_k = None

    print(
        "\nEvaluating..."
    )

    with torch.inference_mode():

        for batch_idx, batch in enumerate(
            loader
        ):

            pixel_values = (
                batch[
                    "pixel_values"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch[
                    "label"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            # ------------------------------------------------
            # First P blocks
            # ------------------------------------------------

            hidden = edge_forward(
                model,
                pixel_values,
                PARTITION
            )

            # ------------------------------------------------
            # ToMe-style trajectory merging
            # ------------------------------------------------

            compact_hidden, k = (
                tome_style_merge(
                    hidden
                )
            )

            final_k = k

            # ------------------------------------------------
            # Remaining compact blocks
            # ------------------------------------------------

            logits = cloud_forward(
                model,
                compact_hidden,
                PARTITION,
                k
            )

            c1, c5 = topk_correct(
                logits,
                labels,
                topk=(1, 5)
            )

            batch_size = (
                labels.size(0)
            )

            total_top1 += c1
            total_top5 += c5

            total_samples += (
                batch_size
            )

            if (
                batch_idx + 1
            ) % 100 == 0:

                running_top1 = (
                    100.0
                    *
                    total_top1
                    /
                    total_samples
                )

                running_top5 = (
                    100.0
                    *
                    total_top5
                    /
                    total_samples
                )

                print(
                    f"{batch_idx + 1}"
                    f"/{len(loader)}"
                    f" | Top-1: "
                    f"{running_top1:.2f}%"
                    f" | Top-5: "
                    f"{running_top5:.2f}%"
                )

    # ========================================================
    # Final accuracy
    # ========================================================

    top1 = (
        100.0
        *
        total_top1
        /
        total_samples
    )

    top5 = (
        100.0
        *
        total_top5
        /
        total_samples
    )

    # ========================================================
    # Token / payload statistics
    # ========================================================

    dense_patch_tokens = (
        NUM_PATCHES
        *
        NUM_FRAMES
    )

    compact_patch_tokens = (
        final_k
        *
        NUM_FRAMES
    )

    retention = (
        compact_patch_tokens
        /
        dense_patch_tokens
    )

    reduction = (
        1.0
        -
        retention
    )

    dense_sequence_tokens = (
        1
        +
        dense_patch_tokens
    )

    compact_sequence_tokens = (
        1
        +
        compact_patch_tokens
    )

    dense_bytes = (
        dense_sequence_tokens
        *
        DIM
        *
        4
    )

    compact_bytes = (
        compact_sequence_tokens
        *
        DIM
        *
        4
    )

    payload_reduction = (
        1.0
        -
        compact_bytes
        /
        dense_bytes
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "AdaST E10 RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"Samples evaluated        : "
        f"{total_samples}"
    )

    print(
        f"Top-1 accuracy           : "
        f"{top1:.4f}%"
    )

    print(
        f"Top-5 accuracy           : "
        f"{top5:.4f}%"
    )

    print(
        f"Final trajectories       : "
        f"{final_k}"
    )

    print(
        f"Dense patch tokens       : "
        f"{dense_patch_tokens}"
    )

    print(
        f"Compact patch tokens     : "
        f"{compact_patch_tokens}"
    )

    print(
        f"Patch-token retention    : "
        f"{100.0 * retention:.2f}%"
    )

    print(
        f"Patch-token reduction    : "
        f"{100.0 * reduction:.2f}%"
    )

    print(
        f"Dense sequence tokens    : "
        f"{dense_sequence_tokens}"
    )

    print(
        f"Compact sequence tokens  : "
        f"{compact_sequence_tokens}"
    )

    print(
        f"FP32 payload reduction   : "
        f"{100.0 * payload_reduction:.2f}%"
    )

    print(
        "=" * 90
    )

    print(
        "\nNOTE:"
    )

    print(
        "This is a controlled ToMe-style "
        "similarity-merging baseline adapted "
        "to the evaluated TimeSformer pipeline."
    )

    print(
        "It should not be described as an exact "
        "reproduction of the complete original "
        "ToMe training/evaluation pipeline."
    )


if __name__ == "__main__":

    main()

