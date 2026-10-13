import inspect
import adast_e14_budget_sweep as e14

print("=" * 75)
print("E33 | REPRESENTATION PRESERVATION BASELINE AUDIT")
print("=" * 75)

print("\n=== AdaSTE14 CONSTRUCTOR ===")
print(inspect.getsource(e14.AdaSTE14.__init__))

print("\n=== AdaSTE14 FORWARD ===")
print(inspect.getsource(e14.AdaSTE14.forward))

print("\n=== COMPACTION-RELATED METHODS ===")
for name in dir(e14.AdaSTE14):
    if any(k in name.lower() for k in [
        "compact", "consolid", "merge", "token", "prun"
    ]):
        obj = getattr(e14.AdaSTE14, name)
        if callable(obj):
            print("\nMETHOD:", name)
            try:
                print(inspect.getsource(obj))
            except (OSError, TypeError):
                print("Source unavailable")

print("\n=== E14 EVALUATION FUNCTIONS ===")
for name in dir(e14):
    if any(k in name.lower() for k in [
        "evaluate", "load_e12", "build_model"
    ]):
        obj = getattr(e14, name)
        if inspect.isfunction(obj):
            print(name, inspect.signature(obj))

print("\nAUDIT COMPLETE")
