import json
from pathlib import Path

root = Path("runs")

files = {
    "E12": root / "e12_rep_preserve/summary.json",
    "E30": root / "e30_policy_ablation/results.json",
    "E31": root / "e31_statistical_validation/e31_statistical_results.json",
}

for name, path in files.items():
    print("\n" + "=" * 75)
    print(name, "|", path)
    print("=" * 75)

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    print(json.dumps(data, indent=2))

print("\nRESULT EXTRACTION COMPLETE")
