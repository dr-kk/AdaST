import json
import csv
from pathlib import Path

root = Path("runs")
out = root / "final_tip_evidence"
out.mkdir(parents=True, exist_ok=True)

e14 = json.loads(
    (root / "e14_budget_sweep/budget_sweep.json").read_text(
        encoding="utf-8-sig"
    )
)

e30 = json.loads(
    (root / "e30_policy_ablation/results.json").read_text(
        encoding="utf-8-sig"
    )
)

points = []

for r in e14["results"]:
    points.append({
        "name": f"E14_K{int(r['K_per_frame'])}_S{int(r['S_requested'])}",
        "group": "E14 fixed",
        "top1": float(r["top1"]),
        "tokens": float(r["physical_compact_patch_tokens"])
    })

for name, r in e30.items():
    points.append({
        "name": name,
        "group": "E30 policy",
        "top1": float(r["top1"]),
        "tokens": float(r["physical_compact_patch_tokens"])
    })

for p in points:
    dominators = [
        q["name"] for q in points
        if q["name"] != p["name"]
        and q["tokens"] <= p["tokens"]
        and q["top1"] >= p["top1"]
        and (
            q["tokens"] < p["tokens"]
            or q["top1"] > p["top1"]
        )
    ]
    p["pareto"] = len(dominators) == 0
    p["dominated_by"] = "; ".join(dominators)

csv_path = out / "tip_pareto_analysis.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=points[0].keys())
    writer.writeheader()
    writer.writerows(points)

print("=== COMPLETE PARETO ANALYSIS ===")
print("Assumption: E14 and E30 protocols are comparable.")
print()

for p in sorted(points, key=lambda x: x["tokens"]):
    status = "PARETO" if p["pareto"] else "DOMINATED"
    print(
        f"{p['name']:30s} "
        f"Top1={p['top1']:8.4f}% "
        f"Tokens={p['tokens']:8.2f} "
        f"{status}"
    )

print("\n=== E30-C DOMINANCE ===")

target = next(
    p for p in points
    if p["name"] == "E30-C_fully_adaptive"
)

print("Dominated by:", target["dominated_by"] or "None")

print("\nSaved:", csv_path)

print("\n=== E14 PRECISION CODE INSPECTION ===")

source = Path("adast_e14_budget_sweep.py")

if source.exists():
    lines = source.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for i, line in enumerate(lines, 1):
        if any(k in line.lower() for k in [
            "autocast", "amp", "float16", "float32"
        ]):
            print(f"{i}: {line.strip()}")
else:
    print("E14 script not found")
