import json
import torch
from pathlib import Path

root = Path("runs/e12_rep_preserve")

print("=" * 70)
print("E12 | CHECKPOINT CONFIGURATION AUDIT")
print("=" * 70)

summary = json.loads(
    (root / "summary.json").read_text(encoding="utf-8-sig")
)

print("\n=== SUMMARY.JSON ===")
print(json.dumps(summary, indent=2, default=str)[:12000])

checkpoint = torch.load(
    root / "best.pt",
    map_location="cpu",
    weights_only=False
)

print("\n=== CHECKPOINT TOP-LEVEL KEYS ===")

if isinstance(checkpoint, dict):
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{key}: dict ({len(value)} entries)")
            if key not in ("model_state_dict", "state_dict"):
                print(str(value)[:1500])
        else:
            print(f"{key}: {str(value)[:500]}")
else:
    print("Checkpoint type:", type(checkpoint))

print("\nAUDIT COMPLETE")
