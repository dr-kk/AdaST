import sys
import torch
import adast_e14_budget_sweep as e14

from adast_e33_direct_drop import direct_drop_forward_compact

print("=" * 75)
print("E33 | SMOKE TEST")
print("=" * 75)

print("\n=== AVAILABLE E14 MAIN SOURCE ===")
import inspect

src = inspect.getsource(e14.main)
lines = src.splitlines()

for i, line in enumerate(lines):
    if any(k in line for k in [
        "test_loader =",
        "DataLoader(",
        "backbone =",
        "from_pretrained(",
        "set_forced_budget(",
        "spatial_levels",
    ]):
        lo = max(0, i - 5)
        hi = min(len(lines), i + 12)
        print("\n".join(
            f"{j+1}: {lines[j]}"
            for j in range(lo, hi)
        ))
        print("---")

print("\n=== CONSOLIDATOR PATCH CHECK ===")
print("Replacement callable:", callable(direct_drop_forward_compact))

print("\n=== BUDGET INTERFACE ===")
print(inspect.signature(e14.AdaSTE14.set_forced_budget))

print("\nREADY FOR E33 INTEGRATION")
