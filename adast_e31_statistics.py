import csv
import json
from pathlib import Path
from math import comb

import numpy as np

ROOT = Path("runs/e31_statistical_validation")
N_BOOT = 10000
SEED = 42

names = {
    "A": "E30-A_spatial_adaptive",
    "B": "E30-B_temporal_adaptive",
    "C": "E30-C_fully_adaptive",
    "D": "E30-D_fully_fixed",
}

def load_predictions(name):
    path = ROOT / f"{name}_predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    data = {}
    for row in rows:
        key = row["video"].replace("\\", "/").lower()
        if key in data:
            raise ValueError(f"Duplicate video: {key}")
        data[key] = (
            int(row["target"]),
            int(row["prediction"]),
        )
    return data

data = {k: load_predictions(v) for k, v in names.items()}
keys = sorted(data["A"])

assert len(keys) == 3783, "Expected 3783 test videos"

for name, records in data.items():
    assert set(records) == set(keys), f"Video mismatch: {name}"

targets = np.array([data["A"][k][0] for k in keys])

correct = {}
for name, records in data.items():
    y = np.array([records[k][0] for k in keys])
    assert np.array_equal(targets, y), f"Target mismatch: {name}"
    pred = np.array([records[k][1] for k in keys])
    correct[name] = (pred == targets)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    # Exact two-sided binomial test using stable recurrence.
    m = min(b, c)
    from scipy.stats import binomtest
    return float(binomtest(m, n, 0.5).pvalue)

rng = np.random.default_rng(SEED)

comparisons = [
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
    ("A", "D"),
    ("A", "C"),
]

results = []

print("=" * 90)
print("E31 | PAIRED STATISTICAL VALIDATION")
print("=" * 90)

for left, right in comparisons:
    x = correct[left]
    y = correct[right]

    delta = 100 * (x.mean() - y.mean())

    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))

    p = mcnemar_exact(b, c)

    # Paired bootstrap, preserving video alignment.
    differences = x.astype(np.int8) - y.astype(np.int8)

    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        boot[i] = 100 * differences[idx].mean()

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    row = {
        "comparison": f"{left} vs {right}",
        "delta_accuracy_pp": float(delta),
        "discordant_left_only": b,
        "discordant_right_only": c,
        "mcnemar_exact_p": p,
        "bootstrap_95_ci_pp": [
            float(ci_low),
            float(ci_high),
        ],
    }
    results.append(row)

    print(
        f"{left} vs {right}: "
        f"Delta={delta:+.4f} pp | "
        f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}] | "
        f"McNemar p={p:.6g} | "
        f"discordant={b}/{c}"
    )

# Holm adjustment across the five planned comparisons.
order = sorted(
    range(len(results)),
    key=lambda i: results[i]["mcnemar_exact_p"]
)
running = 0.0
m = len(results)
for rank, idx in enumerate(order):
    adjusted = min(
        1.0,
        (m - rank) * results[idx]["mcnemar_exact_p"]
    )
    running = max(running, adjusted)
    results[idx]["holm_adjusted_p"] = running

summary = {
    "experiment": "E31_paired_statistical_validation",
    "dataset": "UCF101 Split-1",
    "samples": len(keys),
    "bootstrap_resamples": N_BOOT,
    "seed": SEED,
    "comparisons": results,
}

out = ROOT / "e31_statistical_results.json"
out.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nHolm-adjusted p-values:")
for r in results:
    print(
        f"{r['comparison']}: "
        f"{r['holm_adjusted_p']:.6g}"
    )

print("\n[SAVED]", out)
print("E31 STATISTICAL ANALYSIS COMPLETE")
