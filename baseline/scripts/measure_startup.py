"""Phase 0 baseline: measure import/startup cost in FRESH subprocesses.

Each measurement spawns a new interpreter, because `import engine` executes
model loading at module scope (engine.py lines 7-19) and Python caches modules
within a process. Repeat runs therefore measure warm OS/filesystem cache, not
warm Python state.

Usage:  python baseline/scripts/measure_startup.py <repo_root> <out.json>
"""
import json
import subprocess
import sys
import time

SNIPPET = (
    "import time,sys;"
    "t=time.perf_counter();"
    "import {mod};"
    "sys.stderr.write('ELAPSED=%.4f\\n' % (time.perf_counter()-t))"
)


def time_import(repo_root, module, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", SNIPPET.format(mod=module)],
            cwd=repo_root, capture_output=True, text=True,
        )
        wall = time.perf_counter() - t0
        inner = None
        for line in proc.stderr.splitlines():
            if line.startswith("ELAPSED="):
                inner = float(line.split("=", 1)[1])
        times.append({
            "import_seconds": inner,
            "process_wall_seconds": round(wall, 4),
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else "",
        })
    return times


def main():
    repo_root, out_path = sys.argv[1], sys.argv[2]
    result = {
        "note": (
            "Each run is a fresh interpreter. Run 1 reflects the coldest state "
            "available at measurement time; later runs reflect a warm OS file cache. "
            "No true cold-boot (post-reboot) measurement was taken."
        ),
        "python_executable": sys.executable,
        "measurements": {},
    }
    for module, runs in [("engine", 3), ("app", 2)]:
        print(f"timing 'import {module}' x{runs} ...", flush=True)
        runs_out = time_import(repo_root, module, runs)
        vals = [r["import_seconds"] for r in runs_out if r["import_seconds"] is not None]
        result["measurements"][module] = {
            "runs": runs_out,
            "first_run_seconds": vals[0] if vals else None,
            "min_seconds": round(min(vals), 4) if vals else None,
            "max_seconds": round(max(vals), 4) if vals else None,
        }
        for r in runs_out:
            print(f"  import {module}: {r['import_seconds']}s (rc={r['returncode']})", flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
