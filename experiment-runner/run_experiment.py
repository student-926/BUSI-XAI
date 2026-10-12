import sys
from pathlib import Path
import runpy
import contextlib
from pathlib import Path
import time
from datetime import datetime

RUNNER_DIR = Path(__file__).resolve().parent
LOG_FILE = RUNNER_DIR / "experiment_output.log"
step_files = sorted(RUNNER_DIR.glob("step_*.py"))

if len(step_files) != 6:
    raise RuntimeError(
        f"Expected 6 step files, found {len(step_files)}: "
        f"{[path.name for path in step_files]}"
    )

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

with open(LOG_FILE, "w", encoding="utf-8") as log:
    tee = Tee(sys.stdout, log)
    with contextlib.redirect_stdout(tee):
        print("=" * 60, flush=True)
        print("BUSI-XAI EXPERIMENT RUNNER", flush=True)
        print("=" * 60, flush=True)
        start_time = datetime.now().astimezone()
        start_counter = time.perf_counter()
        print(f"EXPERIMENT started at: {start_time:%Y-%m-%d %H:%M:%S %Z}")
        print("=" * 60)

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
        print("Finished ALL SIX STEP", flush=True)
        print("=" * 60, flush=True)
        end_counter = time.perf_counter()
        end_time = datetime.now().astimezone()
        elapsed = end_counter - start_counter

        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n" + "=" * 60)
        print(f"EXPERIMENT started:  {start_time:%Y-%m-%d %H:%M:%S %Z}")
        print(f"EXPERIMENT finished: {end_time:%Y-%m-%d %H:%M:%S %Z}")
        print(
            f"Consumed time:   {int(hours):02d}:{int(minutes):02d}:"
            f"{seconds:05.2f} (HH:MM:SS)"
        )
        print("=" * 60)

print(f"Output saved to: {LOG_FILE}")