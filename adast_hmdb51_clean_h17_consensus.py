import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

VALID_TARGETS = (6, 7, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "AdaST HMDB51 Clean H17: derive multi-view consensus temporal "
            "pseudo-targets from the 5-view H16-style reliability CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to H17 view_targets.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--expected-views",
        type=int,
        default=5,
        help="Expected number of temporal views per video"
    )

    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.8,
        help="Threshold for high-confidence consensus"
    )

    return parser.parse_args()


def safe_int(x, name):
    try:
        return int(x)
    except Exception as exc:
        raise ValueError(f"Cannot parse {name}={x!r} as int") from exc


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def std_population(values):
    if not values:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def pct(n, d):
    if d == 0:
        return None
    return 100.0 * n / d


def sorted_count_dict(counter):
    return {
        str(k): int(counter.get(k, 0))
        for k in VALID_TARGETS
    }


# ============================================================
# Read raw H17 views
# ============================================================

def load_rows(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {
            "index",
            "path",
            "label",
            "original_target",
            "view",
            "seed",
            "target_s",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"CSV is missing required columns: {sorted(missing)}\n"
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            parsed = dict(row)

            parsed["index"] = safe_int(row["index"], "index")
            parsed["label"] = safe_int(row["label"], "label")
            parsed["original_target"] = safe_int(
                row["original_target"], "original_target"
            )
            parsed["view"] = safe_int(row["view"], "view")
            parsed["seed"] = safe_int(row["seed"], "seed")
            parsed["target_s"] = safe_int(row["target_s"], "target_s")

            if parsed["original_target"] not in VALID_TARGETS:
                raise RuntimeError(
                    f"Invalid original_target={parsed['original_target']} "
                    f"for index={parsed['index']}"
                )

            if parsed["target_s"] not in VALID_TARGETS:
                raise RuntimeError(
                    f"Invalid target_s={parsed['target_s']} "
                    f"for index={parsed['index']}, view={parsed['view']}"
                )

            rows.append(parsed)

    return rows


# ============================================================
# Consensus
# ============================================================

def derive_consensus(rows, expected_views):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["index"]].append(row)

    records = []

    for index in sorted(grouped):
        group = sorted(grouped[index], key=lambda x: x["view"])

        if len(group) != expected_views:
            raise RuntimeError(
                f"Index {index} has {len(group)} views; "
                f"expected exactly {expected_views}."
            )

        # Verify identity fields are stable across views.
        paths = {x["path"] for x in group}
        labels = {x["label"] for x in group}
        originals = {x["original_target"] for x in group}

        if len(paths) != 1:
            raise RuntimeError(
                f"Index {index} has inconsistent paths across views."
            )

        if len(labels) != 1:
            raise RuntimeError(
                f"Index {index} has inconsistent labels across views."
            )

        if len(originals) != 1:
            raise RuntimeError(
                f"Index {index} has inconsistent original targets."
            )

        path = group[0]["path"]
        label = group[0]["label"]
        original_target = group[0]["original_target"]

        targets = [x["target_s"] for x in group]
        counts = Counter(targets)

        max_count = max(counts.values())

        winners = sorted([
            s for s in VALID_TARGETS
            if counts.get(s, 0) == max_count
        ])

        # With 5 views and 3 classes a 2-2-1 tie is possible.
        ambiguous = len(winners) != 1

        if ambiguous:
            consensus_target = None
            confidence = max_count / expected_views
        else:
            consensus_target = winners[0]
            confidence = max_count / expected_views

        original_match = (
            consensus_target == original_target
            if consensus_target is not None
            else False
        )

        record = {
            "index": index,
            "path": path,
            "label": label,
            "original_target": original_target,

            "view_targets": targets,

            "count_s6": int(counts.get(6, 0)),
            "count_s7": int(counts.get(7, 0)),
            "count_s8": int(counts.get(8, 0)),

            "consensus_target": consensus_target,
            "consensus_confidence": confidence,

            "ambiguous": ambiguous,
            "winning_targets": winners,

            "original_matches_consensus": original_match,
        }

        records.append(record)

    return records


# ============================================================
# Summary
# ============================================================

def make_summary(records, expected_views, high_conf_threshold):
    n = len(records)

    original_counts = Counter(
        r["original_target"] for r in records
    )

    resolved = [
        r for r in records
        if not r["ambiguous"]
    ]

    ambiguous = [
        r for r in records
        if r["ambiguous"]
    ]

    consensus_counts = Counter(
        r["consensus_target"]
        for r in resolved
    )

    confidences_all = [
        r["consensus_confidence"]
        for r in records
    ]

    confidences_resolved = [
        r["consensus_confidence"]
        for r in resolved
    ]

    confidence_counts = Counter(
        f"{r['consensus_confidence']:.1f}"
        for r in records
    )

    high_conf = [
        r for r in resolved
        if r["consensus_confidence"] >= high_conf_threshold
    ]

    original_match = [
        r for r in resolved
        if r["original_matches_consensus"]
    ]

    # --------------------------------------------------------
    # By original target
    # --------------------------------------------------------

    by_original = {}

    for original in VALID_TARGETS:
        subset = [
            r for r in records
            if r["original_target"] == original
        ]

        subset_resolved = [
            r for r in subset
            if not r["ambiguous"]
        ]

        subset_ambiguous = [
            r for r in subset
            if r["ambiguous"]
        ]

        cc = Counter(
            r["consensus_target"]
            for r in subset_resolved
        )

        matches = sum(
            1
            for r in subset_resolved
            if r["original_matches_consensus"]
        )

        confs = [
            r["consensus_confidence"]
            for r in subset
        ]

        high = sum(
            1
            for r in subset_resolved
            if r["consensus_confidence"] >= high_conf_threshold
        )

        by_original[str(original)] = {
            "n": len(subset),

            "resolved_n": len(subset_resolved),
            "ambiguous_n": len(subset_ambiguous),

            "consensus_6": int(cc.get(6, 0)),
            "consensus_7": int(cc.get(7, 0)),
            "consensus_8": int(cc.get(8, 0)),

            "agreement_n": matches,

            # Agreement among resolved examples.
            "agreement_percent_resolved": pct(
                matches, len(subset_resolved)
            ),

            # Conservative agreement over all examples,
            # treating ambiguous cases as non-matches.
            "agreement_percent_all": pct(
                matches, len(subset)
            ),

            "confidence_mean": mean(confs),
            "confidence_std": std_population(confs),

            "high_confidence_n": high,
            "high_confidence_percent_all": pct(
                high, len(subset)
            ),
        }

    # --------------------------------------------------------
    # Transition matrix
    # original -> consensus
    # --------------------------------------------------------

    transition_matrix = {}

    for original in VALID_TARGETS:
        transition_matrix[str(original)] = {
            "6": 0,
            "7": 0,
            "8": 0,
            "ambiguous": 0,
        }

    for r in records:
        src = str(r["original_target"])

        if r["ambiguous"]:
            transition_matrix[src]["ambiguous"] += 1
        else:
            dst = str(r["consensus_target"])
            transition_matrix[src][dst] += 1

    summary = {
        "experiment":
            "Clean H17 five-view consensus temporal targets",

        "n_videos": n,
        "views_per_video": expected_views,
        "view_total": n * expected_views,

        "original_target_counts":
            sorted_count_dict(original_counts),

        "resolved_consensus_n": len(resolved),
        "resolved_consensus_percent": pct(
            len(resolved), n
        ),

        "ambiguous_n": len(ambiguous),
        "ambiguous_percent": pct(
            len(ambiguous), n
        ),

        "consensus_target_counts":
            sorted_count_dict(consensus_counts),

        "consensus_confidence_mean_all":
            mean(confidences_all),

        "consensus_confidence_std_all":
            std_population(confidences_all),

        "consensus_confidence_mean_resolved":
            mean(confidences_resolved),

        "consensus_confidence_std_resolved":
            std_population(confidences_resolved),

        "confidence_counts":
            dict(sorted(confidence_counts.items())),

        "high_confidence_threshold":
            high_conf_threshold,

        "high_confidence_n":
            len(high_conf),

        "high_confidence_percent_all":
            pct(len(high_conf), n),

        "high_confidence_percent_resolved":
            pct(len(high_conf), len(resolved)),

        "original_vs_consensus_agreement_n":
            len(original_match),

        "original_vs_consensus_agreement_percent_resolved":
            pct(len(original_match), len(resolved)),

        "original_vs_consensus_agreement_percent_all":
            pct(len(original_match), n),

        "by_original_target":
            by_original,

        "transition_matrix":
            transition_matrix,

        "official_test_evaluated": False,
        "validation_evaluated": False,
        "training_performed": False,

        "notes": (
            "H17 is a training-only consensus analysis derived from "
            "five independently sampled temporal views per selected video. "
            "The 300-video sample is stratified by the original H9-A "
            "pseudo-target and is therefore not population-representative "
            "without reweighting. A unique modal target is used as the "
            "consensus label. A 2-2-1 tie is marked ambiguous rather than "
            "being assigned an arbitrary target. Confidence is the modal "
            "support divided by the number of views. No spatial "
            "augmentation, validation evaluation, official-test evaluation, "
            "or model training is performed."
        ),
    }

    return summary


# ============================================================
# Save
# ============================================================

def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def save_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AdaST HMDB51 Clean H17 | Multi-View Consensus Analysis")
    print("=" * 72)
    print(f"Input CSV       : {input_path}")
    print(f"Output          : {output_dir}")
    print(f"Expected views  : {args.expected_views}")
    print(f"High confidence : >= {args.high_confidence:.2f}")
    print()

    rows = load_rows(input_path)

    print(f"Raw view rows   : {len(rows)}")

    records = derive_consensus(
        rows,
        expected_views=args.expected_views
    )

    print(f"Unique videos   : {len(records)}")

    expected_rows = len(records) * args.expected_views

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} total rows for "
            f"{len(records)} videos x {args.expected_views} views, "
            f"but found {len(rows)}."
        )

    summary = make_summary(
        records,
        expected_views=args.expected_views,
        high_conf_threshold=args.high_confidence
    )

    consensus_path = output_dir / "consensus_targets.jsonl"
    summary_path = output_dir / "h17_summary.json"

    save_jsonl(records, consensus_path)
    save_json(summary, summary_path)

    print()
    print("=" * 72)
    print("H17 RESULT")
    print("=" * 72)

    print(json.dumps(summary, indent=2))

    print()
    print("Saved:")
    print(f"  {consensus_path}")
    print(f"  {summary_path}")

    print()
    print("H17 COMPLETE")


if __name__ == "__main__":
    main()