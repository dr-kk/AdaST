import csv
from pathlib import Path

e12_path = Path("runs/e12_rep_preserve/per_video_test.csv")
e31_path = Path(
    "runs/e31_statistical_validation/"
    "E30-C_fully_adaptive_predictions.csv"
)

def load(path, label_column):
    data = {}

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            p = Path(row["video"].replace("\\", "/"))

            # Class folder + filename
            key = (p.parent.name.lower(), p.name.lower())

            if key in data:
                raise ValueError(f"Duplicate video: {key}")

            data[key] = {
                "label": int(row[label_column]),
                "prediction": int(row["prediction"]),
            }

    return data

a = load(e12_path, "label")
b = load(e31_path, "target")

common = sorted(set(a) & set(b))

print("=" * 75)
print("AdaST | E12 vs E31-C REPRODUCIBILITY AUDIT")
print("=" * 75)

print("E12 videos:", len(a))
print("E31 videos:", len(b))
print("Common videos:", len(common))
print("Missing from E31:", len(set(a) - set(b)))
print("Missing from E12:", len(set(b) - set(a)))

assert len(common) == 3783
assert set(a) == set(b), "Video sets differ"

label_mismatch = []
prediction_mismatch = []

correct_a = 0
correct_b = 0

for key in common:
    x = a[key]
    y = b[key]

    if x["label"] != y["label"]:
        label_mismatch.append(key)

    if x["prediction"] != y["prediction"]:
        prediction_mismatch.append(
            (
                key,
                x["prediction"],
                y["prediction"],
                x["label"],
            )
        )

    correct_a += x["prediction"] == x["label"]
    correct_b += y["prediction"] == y["label"]

print("\nLabel mismatches:", len(label_mismatch))
print("Prediction mismatches:", len(prediction_mismatch))

print("\nE12 correct:", correct_a)
print("E31 correct:", correct_b)

print(f"E12 Top-1: {100 * correct_a / 3783:.6f}%")
print(f"E31 Top-1: {100 * correct_b / 3783:.6f}%")

print("\nDiffering predictions:")

for key, pred_a, pred_b, label in prediction_mismatch[:30]:
    print(
        f"{key[0]}/{key[1]} | "
        f"Label={label} | "
        f"E12={pred_a} | E31={pred_b}"
    )

print("\nAUDIT COMPLETE")
