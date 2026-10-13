import inspect
import adast_e14_budget_sweep as e14

cls = e14.AdaSTE14

print("=" * 75)
print("E33 | EXACT CONSOLIDATION INSPECTION")
print("=" * 75)

print("\n=== _hook() ===")
print(inspect.getsource(cls._hook))

print("\n=== PARENT CLASS ===")
for parent in cls.__mro__:
    print(parent.__module__, parent.__name__)

print("\n=== RELATED MODULE FUNCTIONS ===")
for module in [
    e14,
    __import__(cls.__mro__[1].__module__)
]:
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if any(k in name.lower() for k in [
            "consolid", "merge", "compact", "select"
        ]):
            print("\nFUNCTION:", name)
            try:
                print(inspect.getsource(obj))
            except (OSError, TypeError):
                print("Source unavailable")

print("\nINSPECTION COMPLETE")
