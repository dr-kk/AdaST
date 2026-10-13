import inspect
import adast_e12_rep_preserve as e12
import adast_e14_budget_sweep as e14
import adast_e6_v2_pretrained_compact as e6

print("=" * 75)
print("AdaST | FINAL MODEL AND PREPROCESSING AUDIT")
print("=" * 75)

print("\n=== E6 VIDEO DECODING ===")
for name in ["decode", "UCF", "read_split"]:
    obj = getattr(e6, name, None)
    if obj is not None:
        print(f"\n{name}:")
        print(inspect.getsource(obj))

print("\n=== E12 MAIN FUNCTION ===")
if hasattr(e12, "main"):
    source = inspect.getsource(e12.main)
    for i, line in enumerate(source.splitlines(), 1):
        if any(k in line.lower() for k in [
            "dataset", "loader", "checkpoint",
            "amp", "seed", "model", "processor",
            "num_frames", "insert_block", "physical"
        ]):
            print(f"{i:04d}: {line}")

print("\n=== E12 MODEL CLASS ===")
for name in dir(e12):
    obj = getattr(e12, name)
    if inspect.isclass(obj) and name.startswith("AdaST"):
        print(name, inspect.signature(obj))

print("\n=== E14 MODEL CLASS ===")
print(inspect.signature(e14.AdaSTE14))

print("\nAUDIT COMPLETE")
