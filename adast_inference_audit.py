import inspect
import adast_e12_rep_preserve as e12
import adast_e14_budget_sweep as e14

print("=" * 75)
print("E12 vs E31 | INFERENCE REPRODUCIBILITY INSPECTION")
print("=" * 75)

for module, names in [
    (e12, ["build_test_loader", "evaluate", "evaluate_test"]),
    (e14, ["build_test_loader", "load_e12_checkpoint"]),
]:
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            print(f"\n=== {module.__name__}.{name} ===")
            print(inspect.getsource(obj))

print("\n=== E12 dataset implementation ===")
for name in ["UCF", "decode"]:
    obj = getattr(e12, name, None)
    if callable(obj):
        print(f"\n{name}:")
        print(inspect.getsource(obj))

print("\n=== E14 dataset implementation ===")
for name in ["UCF", "decode"]:
    obj = getattr(e14, name, None)
    if callable(obj):
        print(f"\n{name}:")
        print(inspect.getsource(obj))

print("\n=== E12 relevant settings ===")
for name in [
    "NUM_FRAMES", "SEED", "BATCH_SIZE",
    "SPATIAL_LEVELS", "TEMPORAL_LEVELS"
]:
    if hasattr(e12, name):
        print(name, "=", getattr(e12, name))

print("\nINSPECTION COMPLETE")
