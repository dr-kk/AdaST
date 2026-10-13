import inspect
import adast_e14_budget_sweep as e14

print("=" * 75)
print("E33 | CONSOLIDATOR IMPLEMENTATION")
print("=" * 75)

for parent in e14.AdaSTE14.__mro__:
    print("\nCLASS:", parent.__name__)

    if hasattr(parent, "__init__"):
        try:
            src = inspect.getsource(parent.__init__)
            for i, line in enumerate(src.splitlines(), 1):
                if any(k in line.lower() for k in [
                    "consolidator", "compact", "physical"
                ]):
                    print(f"{i}: {line}")
        except (OSError, TypeError):
            pass

print("\n=== CONSOLIDATOR CLASS ===")

try:
    from adast_e6_v2_pretrained_compact import AdaSTE5

    src = inspect.getsource(AdaSTE5)

    for i, line in enumerate(src.splitlines(), 1):
        if "consolidator" in line.lower():
            lo = max(0, i - 6)
            hi = min(len(src.splitlines()), i + 8)
            print("\n".join(
                f"{j+1}: {src.splitlines()[j]}"
                for j in range(lo, hi)
            ))
except Exception as exc:
    print(type(exc).__name__, str(exc))

print("\n=== FORWARD_COMPACT IMPLEMENTATION ===")

import adast_e6_v2_pretrained_compact as e6

for name, obj in inspect.getmembers(e6, inspect.isclass):
    if hasattr(obj, "forward_compact"):
        print("\nCLASS:", name)
        print(inspect.getsource(obj.forward_compact))

print("\nAUDIT COMPLETE")
