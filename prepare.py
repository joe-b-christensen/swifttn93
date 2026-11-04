#!/usr/bin/env python3
"""SwiftTN93: one-time setup for evaluate.py.

    python3 prepare.py            # build the reference, prepare datasets, record expected outputs + timings

Steps:
  1. Build the reference: tn93 (github.com/veg/tn93) pinned to commit 076eb0f (v1.1.0), compiled with the
     same flags as upstream's CMakeLists (g++ -O3 -std=c++14 -funroll-loops -fopenmp), and check it against
     upstream's own regression tests.
  2. Prepare the dev and holdout alignments: the real sets in data/, tn93's own data/large.fasta, and the simulated
     sets from gen_data.py (fixed seeds).
  3. For each dataset run the reference single-threaded (OMP_NUM_THREADS=1): `tn93 -q <args> -o OUT IN`.
     Record sha256 + size of OUT (the distance file) and of stdout (the JSON summary), and the wall time.
     Single-threaded is the only deterministic reference: with more threads the row order of OUT depends on
     OpenMP scheduling (recorded as ref_jT_matches_j1).
  4. Also time the reference at all CPUs, and upstream's own fast path `-S` (subquadratic, SIMD bitplanes + LSH)
     at all CPUs, as timing baselines.
  5. Write bench/manifest.json (expected outputs, timings, environment, script hashes).
Re-running is safe: finished steps are skipped.
"""
import hashlib
import json
import lzma
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "ext"
BENCH = ROOT / "bench"
DATA = BENCH / "data"
TN93_COMMIT = "076eb0f"
TN93_VERSION = "v1.1.0"
REF = EXT / "tn93" / "tn93_ref"
SRC_FILES = ["TN93.cpp", "stringBuffer.cc", "stringBuffer.h", "tn93_shared.cc", "tn93_shared.h", "argparse.cpp",
             "argparse.hpp", "subquadratic.cpp", "subquadratic.h"]
CXXFLAGS = f'-O3 -std=c++14 -funroll-loops -fopenmp -DVERSION_NUMBER=\\"{TN93_VERSION}\\"'
SCRIPT_FILES = ["prepare.py", "evaluate.py", "gen_data.py"]


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy"])
        if r.returncode != 0:  # PEP 668 "externally managed" images
            sh([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "numpy"])


def build_reference():
    EXT.mkdir(exist_ok=True)
    if REF.exists():
        return
    d = EXT / "tn93"
    if not d.exists():
        sh(["git", "clone", "-q", "https://github.com/veg/tn93.git", str(d)])
    sh(["git", "checkout", "-q", TN93_COMMIT], cwd=d)
    srcs = " ".join(f"src/{f}" for f in SRC_FILES if not f.endswith((".h", ".hpp")))
    sh(f"g++ {CXXFLAGS} -o tn93_ref {srcs}", cwd=d)
    out = subprocess.run([str(REF), "-v"], capture_output=True, text=True)
    assert TN93_VERSION in (out.stdout + out.stderr), out
    # upstream's own regression tests (ground truth shipped in the repo)
    for name, args, inp in [("small_default", "", "test.fas"), ("small_resolve", "-a resolve", "test.fas"),
                            ("small_average", "-a average", "test.fas"), ("small_overlap", "-l 50", "test.fas"),
                            ("small_csvn", "-f csvn", "test.fas"), ("subset_default", "", "subset.fasta")]:
        r = subprocess.run([sys.executable, "tests/test_runner.py", "--binary", str(REF), "--input",
                            f"data/{inp}", "--args", args, "--name", name, "--ground-truth-dir",
                            "tests/ground_truth"], cwd=d, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        assert r.returncode == 0, f"upstream test {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    print(f"reference tn93 {TN93_VERSION} built and passes upstream tests")


def expand(args, row_paths):
    return [row_paths.get(a.strip("{}"), a) if a.startswith("{") else a for a in args]


def run_ref(args, inp, out_path, threads, extra=()):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    t0 = time.perf_counter()
    r = subprocess.run([str(REF), "-q", *extra, *args, "-o", str(out_path), str(inp)], capture_output=True,
                       env=env, check=True)
    return time.perf_counter() - t0, r.stdout


def prepare_sets(specs, kind, threads):
    import gen_data  # noqa: E402  (needs numpy)
    rows = []
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = BENCH / "tmp_out"
    for name, spec in specs.items():
        paths = {}
        if spec["kind"] == "file":
            path = DATA / (name + ".fa")
            if not path.exists():
                shutil.copy(ROOT / spec["src"], path)
        elif spec["kind"] == "real":
            path = DATA / (name + ".fa")
            if not path.exists():
                with lzma.open(ROOT / spec["src"], "rb") as fi, open(path, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        else:
            path = DATA / (name + ".fa")
            path2 = DATA / (name + "_2.fa") if "n2" in spec else None
            if not path.exists():
                t0 = time.perf_counter()
                shape = gen_data.write_dataset(spec, path, path2)
                print(f"generated {name} {shape} in {time.perf_counter()-t0:.1f}s", flush=True)
            if path2:
                paths["in2"] = str(path2.relative_to(ROOT))
        args = spec["args"]
        abs_args = expand(args, {k: str(ROOT / v) for k, v in paths.items()})
        t1, stdout = run_ref(abs_args, path, tmp, 1)
        row = dict(name=name, kind=kind, real=spec["kind"] in ("real", "file"), path=str(path.relative_to(ROOT)),
                   extra_paths=paths, args=args,
                   expected_sha256=sha256_file(tmp), expected_bytes=tmp.stat().st_size,
                   expected_stdout_sha256=sha256_bytes(stdout), expected_stdout_bytes=len(stdout),
                   ref_j1_s=round(t1, 4))
        if threads > 1:
            tT, stdoutT = run_ref(abs_args, path, tmp, threads)
            row.update(ref_jT_s=round(tT, 4), ref_jT_matches_j1=(sha256_file(tmp) == row["expected_sha256"]),
                       ref_jT_stdout_matches_j1=(sha256_bytes(stdoutT) == row["expected_stdout_sha256"]))
        try:  # upstream's own fast path, all CPUs; only a timing baseline
            tS, _ = run_ref(abs_args, path, tmp, threads, extra=("-S",))
            row.update(ref_S_jT_s=round(tS, 4))
        except subprocess.CalledProcessError:
            row.update(ref_S_jT_s=None)
        best = [x for x in (row.get("ref_jT_s"), row["ref_j1_s"], row.get("ref_S_jT_s")) if x]
        row["ref_best_s"] = min(best)
        print(json.dumps(row), flush=True)
        rows.append(row)
    tmp.unlink(missing_ok=True)
    return rows


def main():
    os.chdir(ROOT)
    ensure_numpy()
    sys.path.insert(0, str(ROOT))
    build_reference()
    threads = os.cpu_count() or 1
    manifest_path = BENCH / "manifest.json"
    if manifest_path.exists():
        print("bench/manifest.json exists; delete bench/ to regenerate")
    else:
        import gen_data
        dev = prepare_sets(gen_data.DEV_SPECS, "dev", threads)
        hold = prepare_sets(gen_data.HOLDOUT_SPECS, "holdout", threads)
        cpu, flags = "", ""
        try:
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
            flags = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("flags"))
        except Exception:
            pass
        manifest = dict(
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            reference=dict(tool="tn93", version=TN93_VERSION, commit=TN93_COMMIT,
                           cmd="OMP_NUM_THREADS=1 tn93 -q <args> -o OUT IN"),
            threads=threads, cpu=cpu,
            simd=[f for f in ["sse4_2", "popcnt", "avx2", "avx512f", "avx512_vpopcntdq", "bmi2"] if f in flags.split()],
            gcc=subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
            python=platform.python_version(),
            dev=dev, holdout=hold,
            scripts_sha256={f: sha256_file(ROOT / f) for f in SCRIPT_FILES},
        )
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"wrote {manifest_path}")
    print("prepare: done")


if __name__ == "__main__":
    main()
