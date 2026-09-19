"""Phase 0 baseline: capture environment, dependency and model-artifact facts.

Read-only. Records what IS, including the places where the declared
configuration and the actual environment disagree.

Usage:  python baseline/scripts/capture_environment.py baseline/environment.json
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGES = ["tensorflow", "keras", "scikit-learn", "chess", "flask", "flask-cors",
            "numpy", "scipy", "pandas", "stockfish", "gunicorn", "matplotlib",
            "seaborn", "tqdm", "streamlit", "ipython", "h5py", "protobuf", "pytest"]

MODEL_FILES = ["cnn_model.keras", "mlp_model.pkl", "rf_model.pkl",
               "scaler.pkl", "weight_model.pkl"]

SOURCE_FILES = ["app.py", "engine.py", "requirements.txt", "runtime.txt",
                ".python-version", "Procfile", "benchmark_stockfish.py",
                "generate_visualizations.py", "templates/index.html",
                "chess_model_FINAL.ipynb", "README.md"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def main():
    out_path = sys.argv[1]

    installed = {}
    for name in PACKAGES:
        try:
            installed[name] = md.version(name)
        except Exception:  # noqa: BLE001
            installed[name] = None

    # Declared vs actual
    declared = {}
    for fname in ("requirements.txt", "runtime.txt", ".python-version"):
        p = os.path.join(REPO_ROOT, fname)
        declared[fname] = open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else None

    models = {}
    for fname in MODEL_FILES:
        p = os.path.join(REPO_ROOT, "models", fname)
        if not os.path.isfile(p):
            models[fname] = {"present": False}
            continue
        entry = {"present": True, "size_bytes": os.path.getsize(p),
                 "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                 "sha256": sha256(p)}
        if fname.endswith(".keras"):
            with zipfile.ZipFile(p) as z:
                entry["archive_entries"] = z.namelist()
                entry["metadata_json"] = json.loads(z.read("metadata.json"))
        models[fname] = entry

    sources = {}
    for rel in SOURCE_FILES:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(p):
            sources[rel] = {"size_bytes": os.path.getsize(p), "sha256": sha256(p)[:16]}

    sf_path = (r"C:\Users\vsriv\Downloads\stockfish-windows-x86-64-avx2\stockfish"
               r"\stockfish-windows-x86-64-avx2.exe")
    sf = {"hardcoded_path": sf_path, "exists_on_this_machine": os.path.isfile(sf_path)}
    if sf["exists_on_this_machine"]:
        sf["size_bytes"] = os.path.getsize(sf_path)
        try:
            proc = subprocess.run([sf_path], input="quit\n", capture_output=True,
                                  text=True, timeout=15)
            sf["version_banner"] = proc.stdout.strip().splitlines()[0]
        except Exception as exc:  # noqa: BLE001
            sf["version_banner"] = f"<error: {exc}>"
    sf["on_PATH"] = bool(__import__("shutil").which("stockfish"))

    keras_in_model = models.get("cnn_model.keras", {}).get("metadata_json", {}).get("keras_version")

    payload = {
        "schema_version": 1,
        "generated_by": "baseline/scripts/capture_environment.py",
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit_sha": git("rev-parse", "HEAD"),
            "commit_date": git("log", "-1", "--format=%ai"),
            "commit_subject": git("log", "-1", "--format=%s"),
            "status_porcelain": git("status", "--porcelain").splitlines(),
            "tracked_file_count": len(git("ls-files").splitlines()),
        },
        "interpreter": {
            "version": sys.version.split()[0],
            "version_full": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "installed_packages": installed,
        "declared_dependency_files": declared,
        "models": models,
        "source_files": sources,
        "stockfish": sf,
        "reproducibility_conflicts": [
            {
                "id": "PY-VERSION-MISMATCH",
                "declared": f"runtime.txt={declared.get('runtime.txt')!r}, "
                            f".python-version={declared.get('.python-version')!r}",
                "actual": sys.version.split()[0],
                "impact": "The interpreter actually in use is not the one the repo declares.",
            },
            {
                "id": "TF-PIN-CANNOT-LOAD-MODEL",
                "declared": "requirements.txt pins tensorflow==2.15.0, which ships Keras 2.x",
                "actual": f"models/cnn_model.keras was written by Keras {keras_in_model}; "
                          f"installed keras=={installed.get('keras')}, "
                          f"tensorflow=={installed.get('tensorflow')}",
                "impact": "A clean install from requirements.txt is expected to be unable "
                          "to load the committed model. NOT empirically verified in Phase 0 "
                          "(installing TF 2.15 was out of scope); based on the Keras version "
                          "recorded inside the model archive.",
            },
            {
                "id": "SKLEARN-PICKLE-SKEW",
                "declared": "requirements.txt pins scikit-learn with no version",
                "actual": f"installed scikit-learn=={installed.get('scikit-learn')}; "
                          "pickles emit InconsistentVersionWarning (written by 1.7.1)",
                "impact": "Unpinned; sklearn warns that results may be invalid.",
            },
            {
                "id": "GUNICORN-MISSING",
                "declared": "Procfile runs 'gunicorn app:app'; requirements.txt lists gunicorn",
                "actual": f"gunicorn installed: {installed.get('gunicorn')}",
                "impact": "The documented production start command cannot run in this venv.",
            },
            {
                "id": "STOCKFISH-HARDCODED-PATH",
                "declared": "absolute Windows path literal in app.py:383 and "
                            "benchmark_stockfish.py:12",
                "actual": f"exists on this machine: {sf['exists_on_this_machine']}; "
                          f"on PATH: {sf['on_PATH']}",
                "impact": "Stockfish-dependent code is machine-specific.",
            },
            {
                "id": "CWD-RELATIVE-MODEL-PATHS",
                "declared": "engine.py:7 loads 'models/cnn_model.keras' (relative)",
                "actual": "import fails unless CWD is the repo root; observed during "
                          "Phase 0 when running a script from baseline/scripts/",
                "impact": "Import is CWD-dependent; affects pytest and CI invocation.",
            },
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)
    print("python", payload["interpreter"]["version"],
          "| tf", installed.get("tensorflow"), "| keras", installed.get("keras"),
          "| model written by keras", keras_in_model)


if __name__ == "__main__":
    main()
