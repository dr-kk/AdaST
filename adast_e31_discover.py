from pathlib import Path
import csv
import json

roots = [
    Path("runs/e12_rep_preserve"),
    Path("runs/e14_budget_sweep"),
    Path("runs/e30_policy_ablation"),
]

keywords = (
    "prediction", "predictions", "per_video",
    "per_sample", "sample_level", "logits"
)

print("=" * 75)
print("E31 | EXISTING PREDICTION ARTIFACT DISCOVERY")
print("=" * 75)

for root in roots:
    print(f"\n[{root}]")

    if not root.exists():
        print("Directory not found")
        continue

    matches = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".csv", ".json", ".npz", ".pt", ".pth")
        and any(k in p.name.lower() for k in keywords)
    ]

    if not matches:
        print("No prediction artifacts found")
    else:
        for p in matches[:30]:
            print(p, f"({p.stat().st_size:,} bytes)")

print("\nE30 summary check:")
p = Path("runs/e30_policy_ablation/results.json")

if p.exists():
    results = json.loads(p.read_text(encoding="utf-8"))
    for name, r in results.items():
        print(
            f"{name}: "
            f"Top1={r['top1']:.4f}% "
            f"Tokens={r['physical_compact_patch_tokens']:.2f}"
        )
else:
    print("E30 results.json missing")

print("\nDiscovery complete.")
