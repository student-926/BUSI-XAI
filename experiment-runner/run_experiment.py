
from pathlib import Path
import runpy
import sys

RUNNER_DIR = Path(__file__).resolve().parent

step_files = sorted(RUNNER_DIR.glob("step_*.py"))

if len(step_files) != 6:
    raise RuntimeError(
        f"Expected 6 step files, found {len(step_files)}: "
        f"{[path.name for path in step_files]}"
    )

print("=" * 60, flush=True)
print("BUSI-XAI EXPERIMENT RUNNER — TEST", flush=True)
print("=" * 60, flush=True)

for number, step_file in enumerate(step_files, start=1):
    print(
        f"\n[{number}/6] Running {step_file.name}",
        flush=True,
    )

    try:
        runpy.run_path(str(step_file), run_name="__main__")
    except Exception:
        print(
            f"\nFAILED: {step_file.name}",
            flush=True,
        )
        raise

    print(f"PASSED: {step_file.name}", flush=True)

print("\n" + "=" * 60, flush=True)
print("ALL SIX STEP FILES PASSED THE TEST", flush=True)
print("=" * 60, flush=True)