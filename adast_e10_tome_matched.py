import os
import math
import torch

from torch.utils.data import DataLoader
from transformers import TimesformerForVideoClassification

from ucf101_dataset import UCF101Dataset


# ============================================================
# Configuration
# ============================================================

ROOT = r"C:\Users\abc\ST-EdgeViT"

DATASET_DIR = os.path.join(
    ROOT,
    "datasets",
    "UCF101"
)

SPLIT_DIR = os.path.join(
    ROOT,
    "splits",
    "ucfTrainTestlist"
)

CLASS_FILE = os.path.join(
    SPLIT_DIR,
    "classInd.txt"
)

TEST_FILE = os.path.join(
    SPLIT_DIR,
    "testlist01.txt"
)

CHECKPOINT = os.path.join(
    ROOT,
    "checkpoints",
    "exp1_timesformer",
    "best.pt"
)

MODEL_NAME = (
    "facebook/timesformer-base-finetuned-k400"
)

NUM_CLASSES = 101

NUM_FRAMES = 8
NUM_PATCHES = 196
DIM = 768

PARTITION = 6

# ------------------------------------------------------------
# Final communication-equivalent trajectory budget
# ------------------------------------------------------------

TARGET_TOKENS = 74

BATCH_SIZE = 1
NUM_WORKERS = 0

BASELINE_TOP1 = 92.55


# ============================================================
# Accuracy helper
# ============================================================

def topk_correct(
    logits,
    labels,
    topk=(1, 5)
):

    maxk = max(topk)

    _, pred = logits.topk(
        maxk,
        dim=1,
        largest=True,
        sorted=True
    )

    pred = pred.t()

    correct = pred.eq(
        labels.view(
            1,
            -1
        ).expand_as(pred)
    )

    values = []

    for k in topk:

        correct_k = (
            correct[:k]
            .reshape(-1)
            .float()
            .sum()
            .item()
        )

        values.append(
            correct_k
        )

    return values


# ============================================================
# Edge forward
# ============================================================

def edge_forward(
    model,
    pixel_values,
    partition
):

    hidden = (
        model.timesformer
        .embeddings(
            pixel_values
        )
    )

    for layer_idx in range(
        partition
    ):

        layer = (
            model.timesformer
            .encoder
            .layer[layer_idx]
        )

        outputs = layer(
            hidden,
            output_attentions=False
        )

        hidden = outputs[0]

    return hidden


# ============================================================
# One ToMe-style bipartite merging round
#
# Token trajectories:
#
# [B, N, T, D]
#
# A and B are alternating trajectory sets.
# Each source trajectory in A is matched to its most similar
# trajectory in B using cosine similarity of temporally pooled
# descriptors.
#
# Highest-similarity source tokens are merged into destinations.
# ============================================================

def tome_merge_round(
    trajectories,
    sizes,
    target_count
):

    batch_size = trajectories.shape[0]
    num_tokens = trajectories.shape[1]
    hidden_dim = trajectories.shape[-1]

    if target_count >= num_tokens:

        return (
            trajectories,
            sizes
        )

    # Number of source trajectories to merge
    num_merges = (
        num_tokens
        -
        target_count
    )

    # ========================================================
    # Alternating bipartite split
    # ========================================================

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

    # Handle odd token count
    if dst_indices.numel() == 0:

        return (
            trajectories,
            sizes
        )

    src = trajectories[
        :,
        src_indices,
        :,
        :
    ]

    dst = trajectories[
        :,
        dst_indices,
        :,
        :
    ]

    src_sizes = sizes[
        :,
        src_indices
    ]

    dst_sizes = sizes[
        :,
        dst_indices
    ]

    # Maximum possible merges this round
    num_merges = min(
        num_merges,
        src.shape[1]
    )

    if num_merges <= 0:

        return (
            trajectories,
            sizes
        )

    # ========================================================
    # Temporally pooled trajectory descriptors
    # ========================================================

    src_desc = src.mean(
        dim=2
    )

    dst_desc = dst.mean(
        dim=2
    )

    src_desc = (
        torch.nn.functional.normalize(
            src_desc,
            p=2,
            dim=-1
        )
    )

    dst_desc = (
        torch.nn.functional.normalize(
            dst_desc,
            p=2,
            dim=-1
        )
    )

    # ========================================================
    # Pairwise cosine similarity
    #
    # [B, N_src, N_dst]
    # ========================================================

    similarity = torch.matmul(
        src_desc,
        dst_desc.transpose(
            1,
            2
        )
    )

    best_similarity, best_dst = (
        similarity.max(
            dim=-1
        )
    )

    # ========================================================
    # Choose the most redundant source trajectories
    # ========================================================

    _, merge_src_local = torch.topk(
        best_similarity,
        k=num_merges,
        dim=1,
        largest=True,
        sorted=False
    )

    # ========================================================
    # BATCH_SIZE = 1 in this experiment.
    #
    # Implement robust per-batch merging.
    # ========================================================

    batch_outputs = []
    batch_sizes_out = []

    for b in range(
        batch_size
    ):

        merge_mask = torch.zeros(
            src.shape[1],
            dtype=torch.bool,
            device=trajectories.device
        )

        merge_mask[
            merge_src_local[b]
        ] = True

        unmerged_src = src[
            b,
            ~merge_mask,
            :,
            :
        ]

        unmerged_src_sizes = src_sizes[
            b,
            ~merge_mask
        ]

        # Clone destination trajectories because
        # they will receive merged sources.
        merged_dst = dst[
            b
        ].clone()

        merged_dst_sizes = dst_sizes[
            b
        ].clone()

        selected_src = src[
            b,
            merge_mask,
            :,
            :
        ]

        selected_src_sizes = src_sizes[
            b,
            merge_mask
        ]

        selected_dst_indices = best_dst[
            b,
            merge_mask
        ]

        # ====================================================
        # Weighted merging
        #
        # Preserve token mass when multiple sources map
        # to the same destination.
        # ====================================================

        for local_idx in range(
            selected_src.shape[0]
        ):

            destination = int(
                selected_dst_indices[
                    local_idx
                ].item()
            )

            src_weight = (
                selected_src_sizes[
                    local_idx
                ]
            )

            dst_weight = (
                merged_dst_sizes[
                    destination
                ]
            )

            total_weight = (
                src_weight
                +
                dst_weight
            )

            merged_dst[
                destination
            ] = (
                merged_dst[
                    destination
                ]
                *
                dst_weight
                +
                selected_src[
                    local_idx
                ]
                *
                src_weight
            ) / (
                total_weight
                +
                1e-8
            )

            merged_dst_sizes[
                destination
            ] = total_weight

        # ====================================================
        # New trajectory sequence
        #
        # Keep unmerged source tokens + all destination tokens
        # ====================================================

        output_tokens = torch.cat(
            [
                unmerged_src,
                merged_dst
            ],
            dim=0
        )

        output_sizes = torch.cat(
            [
                unmerged_src_sizes,
                merged_dst_sizes
            ],
            dim=0
        )

        batch_outputs.append(
            output_tokens
        )

        batch_sizes_out.append(
            output_sizes
        )

    trajectories = torch.stack(
        batch_outputs,
        dim=0
    )

    sizes = torch.stack(
        batch_sizes_out,
        dim=0
    )

    return (
        trajectories,
        sizes
    )


# ============================================================
# ToMe-style trajectory merging
#
# 196 -> 98 -> 74
#
# This avoids arbitrary token deletion and progressively
# merges highly similar trajectories.
# ============================================================

def tome_style_merge(
    hidden
):

    batch_size = hidden.shape[0]
    hidden_dim = hidden.shape[-1]

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

    if patches.shape[1] != expected:

        raise RuntimeError(
            f"Unexpected patch count "
            f"{patches.shape[1]}, "
            f"expected {expected}"
        )

    # --------------------------------------------------------
    # TimeSformer token ordering used in the existing
    # ST-EdgeViT experiments:
    #
    # [B,N,T,D]
    # --------------------------------------------------------

    trajectories = patches.reshape(
        batch_size,
        NUM_PATCHES,
        NUM_FRAMES,
        hidden_dim
    )

    sizes = torch.ones(
        batch_size,
        NUM_PATCHES,
        device=hidden.device,
        dtype=hidden.dtype
    )

    # ========================================================
    # Round 1
    #
    # 196 -> 98
    # ========================================================

    intermediate_target = max(
        TARGET_TOKENS,
        math.ceil(
            NUM_PATCHES
            /
            2
        )
    )

    trajectories, sizes = (
        tome_merge_round(
            trajectories,
            sizes,
            intermediate_target
        )
    )

    # ========================================================
    # Additional rounds until target reached
    # ========================================================

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

    if final_k != TARGET_TOKENS:

        raise RuntimeError(
            f"Final merged token count "
            f"{final_k}, expected "
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

    compact_hidden = torch.cat(
        [
            cls_token,
            compact_patches
        ],
        dim=1
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
        != expected_tokens
    ):

        raise RuntimeError(
            "Unexpected compact hidden "
            f"shape: {hidden_states.shape}"
        )

    # ========================================================
    # Temporal attention
    # ========================================================

    patch_tokens = hidden_states[
        :,
        1:,
        :
    ]

    temporal_embedding = (
        patch_tokens.reshape(
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

    temporal_outputs = (
        layer.temporal_attention(
            layer.temporal_layernorm(
                temporal_embedding
            ),
            output_attentions=False
        )
    )

    temporal_output = (
        temporal_outputs[0]
    )

    residual_temporal = (
        layer.drop_path(
            temporal_output
        )
    )

    residual_temporal = (
        residual_temporal.reshape(
            batch_size,
            num_spatial_tokens,
            NUM_FRAMES,
            hidden_dim
        )
    )

    residual_temporal = (
        residual_temporal.reshape(
            batch_size,
            num_spatial_tokens
            *
            NUM_FRAMES,
            hidden_dim
        )
    )

    residual_temporal = (
        layer.temporal_dense(
            residual_temporal
        )
    )

    temporal_embedding = (
        patch_tokens
        +
        residual_temporal
    )

    # ========================================================
    # Spatial attention
    # ========================================================

    init_cls_token = (
        hidden_states[
            :,
            0,
            :
        ]
        .unsqueeze(1)
    )

    cls_token = (
        init_cls_token.repeat(
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

    # ========================================================
    # CLS residual
    # ========================================================

    cls_residual = residual_spatial[
        :,
        0,
        :
    ]

    cls_residual = cls_residual.reshape(
        batch_size,
        NUM_FRAMES,
        hidden_dim
    )

    cls_residual = torch.mean(
        cls_residual,
        dim=1,
        keepdim=True
    )

    # ========================================================
    # Patch residual
    # ========================================================

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
# Cloud forward
# ============================================================

def cloud_forward(
    model,
    compact_hidden,
    partition,
    num_spatial_tokens
):

    hidden = compact_hidden

    for layer_idx in range(
        partition,
        model.config.num_hidden_layers
    ):

        layer = (
            model.timesformer
            .encoder
            .layer[layer_idx]
        )

        hidden = (
            compact_timesformer_layer(
                layer,
                hidden,
                num_spatial_tokens
            )
        )

    hidden = (
        model.timesformer
        .layernorm(
            hidden
        )
    )

    cls_representation = hidden[
        :,
        0,
        :
    ]

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

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU required."
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 104)

    print(
        "ST-EdgeViT Exp-7C"
    )

    print(
        "ToMe-Style Controlled Baseline "
        "-- UCF101 Split-1"
    )

    print("=" * 104)

    print("\nGPU:")

    print(
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "\nConfiguration:"
    )

    print(
        "Partition P      :",
        PARTITION
    )

    print(
        "Initial tokens   :",
        NUM_PATCHES
    )

    print(
        "Target tokens    :",
        TARGET_TOKENS
    )

    print(
        "Precision        : FP32"
    )

    print(
        "Selection        : "
        "similarity-based trajectory merging"
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
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
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

    with torch.no_grad():

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

            # ================================================
            # Edge execution
            # ================================================

            hidden = edge_forward(
                model,
                pixel_values,
                PARTITION
            )

            # ================================================
            # ToMe-style merging
            # ================================================

            compact_hidden, k = (
                tome_style_merge(
                    hidden
                )
            )

            final_k = k

            # ================================================
            # Cloud execution
            # ================================================

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
            ) % 250 == 0:

                running_top1 = (
                    100.0
                    *
                    total_top1
                    /
                    total_samples
                )

                print(
                    f"{batch_idx + 1}"
                    f"/{len(loader)}"
                    f" | Top-1: "
                    f"{running_top1:.2f}%"
                )

    # ========================================================
    # Accuracy
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

    delta_accuracy = (
        BASELINE_TOP1
        -
        top1
    )

    # ========================================================
    # Payload
    #
    # Merged representation:
    #
    # 1 CLS + K*T feature tokens.
    #
    # No retained-index signalling is required because
    # merged features are directly transmitted.
    # ========================================================

    baseline_tokens = (
        1
        +
        NUM_PATCHES
        *
        NUM_FRAMES
    )

    baseline_bytes = (
        baseline_tokens
        *
        DIM
        *
        4
    )

    retained_tokens = (
        1
        +
        final_k
        *
        NUM_FRAMES
    )

    payload_bytes = (
        retained_tokens
        *
        DIM
        *
        4
    )

    payload_kb = (
        payload_bytes
        /
        1024.0
    )

    reduction = (
        1.0
        -
        payload_bytes
        /
        baseline_bytes
    ) * 100.0

    # ========================================================
    # Final results
    # ========================================================

    print(
        "\n"
        +
        "=" * 104
    )

    print(
        "EXP-7C TOME-STYLE CONTROLLED "
        "BASELINE RESULTS"
    )

    print(
        "=" * 104
    )

    print(
        f"Test videos       : "
        f"{total_samples}"
    )

    print(
        f"Partition P       : "
        f"{PARTITION}"
    )

    print(
        f"Retained K        : "
        f"{final_k}"
    )

    print(
        f"Top-1             : "
        f"{top1:.2f} %"
    )

    print(
        f"Top-5             : "
        f"{top5:.2f} %"
    )

    print(
        f"Payload           : "
        f"{payload_kb:.2f} KB"
    )

    print(
        f"Reduction         : "
        f"{reduction:.2f} %"
    )

    print(
        f"dA vs baseline    : "
        f"{delta_accuracy:.2f} pp"
    )

    print(
        "=" * 104
    )

    print(
        "\nNOTE:"
    )

    print(
        "This is a controlled ToMe-style "
        "similarity-based merging baseline, "
        "not an exact reproduction of the "
        "official ToMe implementation."
    )

    print(
        "\nEXP-7C COMPLETE"
    )


if __name__ == "__main__":

    main()