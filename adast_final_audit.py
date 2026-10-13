import csv
from pathlib import Path

root = Path("runs")

e12 = root / "e12_rep_preserve/per_video_test.csv"
e31 = root / "e31_statistical_validation/E30-C_fully_adaptive_predictions.csv"

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

cols12, rows12 = read_csv(e12)
cols31, rows31 = read_csv(e31)

print("=" * 75)
print("AdaST | FINAL REPRODUCIBILITY AUDIT")
print("=" * 75)

print("\nE12 columns:", cols12)
print("E31 columns:", cols31)

print("\nE12 samples:", len(rows12))
print("E31 samples:", len(rows31))

print("\nE12 first row:")
print(rows12[0])

print("\nE31 first row:")
print(rows31[0])

print("\nE12 last row:")
print(rows12[-1])

print("\nE31 last row:")
print(rows31[-1])

print("\nAudit inspection complete.")
