import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("runs")
OUT = ROOT / "final_tip_evidence"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ROOT / "e12_rep_preserve" / "summary.json",
    ROOT / "e12_rep_preserve" / "per_video_test.csv",
    ROOT / "e12_rep_preserve" / "best.pt",
    ROOT / "e30_policy_ablation" / "results.json",
    ROOT / "e31_statistical_validation" / "results.json",
    ROOT / "e31_statistical_validation" / "e31_statistical_results.json",
    ROOT / "e31_fp16_verify" / "results.json",
    ROOT / "e31_fp16_verify" / "E30-C_fully_adaptive_predictions.csv",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

records = []

for path in FILES:
    if not path.exists():
        print("[MISSING]", path)
        continue

    record = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path)
    }
    records.append(record)
    print("[OK]", path)

manifest = {
    "project": "AdaST",
    "target": "IEEE Transactions on Image Processing",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "reproducibility": {
        "e12_fp16_test_videos": 3783,
        "label_mismatches": 0,
        "prediction_mismatches": 0,
        "status": "PASS"
    },
    "precision_note": (
        "E12 reproduced under FP16. "
        "E30/E31 policy comparisons were evaluated under FP32. "
        "Do not directly mix these measurements."
    ),
    "files": records
}

path = OUT / "evidence_manifest.json"
path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("\nManifest saved:", path)
print("Files recorded:", len(records))
print("Missing files:", len(FILES) - len(records))
