import csv
from pathlib import Path

def load(path, label_col):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = Path(row["video"].replace("\\", "/"))
            key = (p.parent.name.lower(), p.name.lower())
            if key in result:
                raise ValueError(f"Duplicate video: {key}")
            result[key] = (
                int(row[label_col]),
                int(row["prediction"])
            )
    return result

e12 = load(
    "runs/e12_rep_preserve/per_video_test.csv",
    "label"
)

fp16 = load(
    "runs/e31_fp16_verify/E30-C_fully_adaptive_predictions.csv",
    "target"
)

assert set(e12) == set(fp16), "Video sets differ"

label_mismatch = sum(
    e12[k][0] != fp16[k][0] for k in e12
)

prediction_mismatch = sum(
    e12[k][1] != fp16[k][1] for k in e12
)

print("=" * 65)
print("AdaST | FINAL FP16 REPRODUCIBILITY CHECK")
print("=" * 65)
print("Videos:", len(e12))
print("Label mismatches:", label_mismatch)
print("Prediction mismatches:", prediction_mismatch)

if label_mismatch == 0 and prediction_mismatch == 0:
    print("[PASS] E12 reproduced at per-video prediction level")
else:
    print("[CHECK] Further investigation required")
