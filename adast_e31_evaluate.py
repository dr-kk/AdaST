# E31: Per-video evaluation based on E30.
# E12/E14/E30 original source files remain unchanged.

from pathlib import Path
from adast_e30_policy_ablation import *

def evaluate_point_e31(
    model,
    loader,
    device,
    args,
    rho,
    temporal_s,
):

    model.eval()

    model.force_spatial_r = (
        None if rho is None else float(0.5 if rho is None else rho)
    )
    model.force_temporal_s = (
        None if temporal_s is None else int(8 if temporal_s is None else temporal_s)
    )

    prediction_rows = []
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

        x, y, paths = e12.unpack_batch(
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

        pred_cpu = logits.argmax(dim=1).detach().cpu().tolist()
        target_cpu = y.detach().cpu().tolist()

        if isinstance(paths, (str, Path)):
            paths = [str(paths)]
        else:
            paths = [str(p) for p in paths]

        if len(paths) != len(target_cpu):
            raise RuntimeError(
                "Video-path count does not match batch size"
            )

        for path, target, pred in zip(
            paths, target_cpu, pred_cpu
        ):
            prediction_rows.append({
                "video": path.replace("\\\\", "/").replace("\\", "/"),
                "target": int(target),
                "prediction": int(pred),
                "correct": int(target == pred),
            })

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
        float(0.5 if rho is None else rho)
        * float(8 if temporal_s is None else temporal_s)
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
            None if rho is None else float(rho),

        "S_requested":
            None if temporal_s is None else int(temporal_s),

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
            float(0.5 if rho is None else rho),

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

        "predictions": prediction_rows,

        "seconds":
            time.time()
            - start_time,
    }
