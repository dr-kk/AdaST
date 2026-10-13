import inspect
import adast_e14_budget_sweep as e14

print("=== E14 EVALUATE_POINT ===")
print(inspect.getsource(e14.evaluate_point))

print("\n=== E14 MAIN MODEL SETUP ===")
src = inspect.getsource(e14.main)
lines = src.splitlines()

for i, line in enumerate(lines):
    if any(k in line for k in [
        "AdaSTE14(",
        "load_e12_checkpoint",
        "physical_mode",
        "compact_tail",
        "evaluate_point("
    ]):
        start = max(0, i - 5)
        end = min(len(lines), i + 9)

        print("\n---")
        for j in range(start, end):
            print(f"{j+1}: {lines[j]}")
