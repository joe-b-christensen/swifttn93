#!/usr/bin/env python3
"""SwiftTN93: correctness check against tn93 v1.1.0 + benchmark.

    python3 evaluate.py                     # build, random-input check, benchmark-set check + timing
    python3 evaluate.py --holdout           # the same on the five holdout sets
    python3 evaluate.py --quick             # build + random-input check only
    python3 evaluate.py --threads 1         # override the thread count used for timing

The check has two parts, and both must pass:
  * Fuzz: fresh random tiny inputs every run (new seed each time, printed), random tn93 flags, random candidate
    thread counts (OMP_NUM_THREADS). The candidate's distance file (-o) AND its stdout (the JSON summary) must
    equal the reference's (`OMP_NUM_THREADS=1 tn93 -q <flags> -o OUT IN`) byte for byte. When the reference
    exits non-zero, the candidate must too.
  * Benchmark sets: the candidate's distance file and stdout must match the recorded sha256 of the reference's.
Timing: wall clock of the full command (parse + compute + write), best of 3 when a run takes < 5 s. The output
file is deleted before each timed repetition, outside the timer.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gen_data  # noqa: E402

MANIFEST = ROOT / "bench" / "manifest.json"
REF = ROOT / "ext" / "tn93" / "tn93_ref"
CAND = ROOT / "solver" / "swifttn93"
FAIL_DIR = ROOT / "failures"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd, timeout, threads):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", b"", b""


def read_or_none(p):
    return p.read_bytes() if p.exists() else None


def fuzz_gate(n_cases, seed, max_fail=5):
    rng = random.Random(seed)
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for k in range(n_cases):
            files, flags = gen_data.fuzz_case(rng)
            names = {key: f"{key}_{rng.randrange(16**8):08x}.fa" for key in files}
            for key, data in files.items():
                (td / names[key]).write_bytes(data)
            flags = [names[f.strip("{}")] if f.startswith("{") else f for f in flags]
            threads = rng.choice([1, 2, 3, 4, 7])
            for p in ("ref.out", "cand.out"):
                (td / p).unlink(missing_ok=True)
            rc_r, so_r, _ = run([str(REF), "-q", *flags, "-o", "ref.out", names["in"]], td, 20, 1)
            rc_c, so_c, err_c = run([str(CAND), "-q", *flags, "-o", "cand.out", names["in"]], td, 20, threads)
            out_r, out_c = read_or_none(td / "ref.out"), read_or_none(td / "cand.out")
            if rc_r != 0:
                ok = rc_c != 0 and rc_c != "timeout"
            else:
                ok = rc_c == 0 and out_c == out_r and so_c == so_r
            if not ok:
                fails += 1
                FAIL_DIR.mkdir(exist_ok=True)
                tag = f"fuzz_{seed}_{k}"
                for key, data in files.items():
                    (FAIL_DIR / f"{tag}_{key}.fa").write_bytes(data)
                (FAIL_DIR / (tag + ".json")).write_text(json.dumps(dict(
                    flags=flags, file_names=names, threads=threads, ref_rc=rc_r, cand_rc=str(rc_c),
                    ref_out_head=(out_r or b"")[:800].decode(errors="replace"),
                    cand_out_head=(out_c or b"")[:800].decode(errors="replace"),
                    ref_stdout_head=so_r[:800].decode(errors="replace"),
                    cand_stdout_head=so_c[:800].decode(errors="replace"),
                    cand_stderr_head=err_c[:600].decode(errors="replace")), indent=1))
                print(f"FUZZ FAIL case {k}: flags={flags} threads={threads} ref_rc={rc_r} cand_rc={rc_c} "
                      f"-> failures/{tag}.*")
                if fails >= max_fail:
                    break
    return fails


class Bench:
    """Runs one benchmark row with randomly named symlinked inputs in a fresh temp dir."""

    def __init__(self, row, td):
        self.row, self.td = row, Path(td)
        src = ROOT / row["path"]
        self.inp = f"x{random.randrange(16**8):08x}.fa"
        os.symlink(src, self.td / self.inp)
        extra = {}
        for key, rel in row.get("extra_paths", {}).items():
            nm = f"y{random.randrange(16**8):08x}.fa"
            os.symlink(ROOT / rel, self.td / nm)
            extra[key] = nm
        self.args = [extra[a.strip("{}")] if a.startswith("{") else a for a in row["args"]]
        self.timeout = max(60.0, 5.0 * row["ref_j1_s"])
        self.k = 0

    def once(self, binary, threads):
        """One timed run. Returns (seconds or None, ok)."""
        self.k += 1
        out = self.td / f"out_{self.k}_{random.randrange(16**6):06x}"
        out.unlink(missing_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        t0 = time.perf_counter()
        try:
            r = subprocess.run([str(binary), "-q", *self.args, "-o", out.name, self.inp], cwd=self.td,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.timeout, env=env)
            rc, so = r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            rc, so = "timeout", b""
        dt = time.perf_counter() - t0
        row = self.row
        ok = (rc == 0 and out.exists() and out.stat().st_size == row["expected_bytes"]
              and sha256_file(out) == row["expected_sha256"]
              and hashlib.sha256(so).hexdigest() == row["expected_stdout_sha256"])
        if not ok:
            size = out.stat().st_size if out.exists() else None
            print(f"BENCH FAIL {row['name']} ({binary.name}): rc={rc} bytes={size} want={row['expected_bytes']} "
                  f"stdout_ok={hashlib.sha256(so).hexdigest() == row['expected_stdout_sha256']}")
        out.unlink(missing_ok=True)
        return dt, ok


def bench(rows, threads):
    results, fails = [], 0
    with tempfile.TemporaryDirectory() as td:
        for row in rows:
            b = Bench(row, td)
            times, ok = [], True
            for _ in range(3):
                dt, ok = b.once(CAND, threads)
                if not ok:
                    break
                times.append(dt)
                if dt >= 5.0:
                    break
            if not ok:
                fails += 1
                results.append(dict(name=row["name"], ok=False))
                continue
            best = min(times)
            results.append(dict(name=row["name"], ok=True, real=row.get("real", False), seconds=round(best, 6),
                                speedup=round(row["ref_j1_s"] / best, 3),
                                speedup_vs_upstream_best=round(row["ref_best_s"] / best, 3)))
    return results, fails


def build(solver_dir, binary):
    binary.unlink(missing_ok=True)
    b = subprocess.run(["bash", str(solver_dir / "build.sh")], cwd=solver_dir.parent, capture_output=True, text=True,
                       timeout=600)
    if b.returncode != 0 or not binary.exists():
        print(b.stdout[-3000:], b.stderr[-3000:])
        return False
    return True


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fuzz", type=int, default=300)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    bad = [f for f, h in manifest["scripts_sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        print(f"SCRIPTS MODIFIED since prepare.py: {bad}. Restore them, or delete bench/ and rerun prepare.py.")
        sys.exit(2)

    t_start = time.time()
    if not build(ROOT / "solver", CAND):
        print("---\ngate:             BUILD_FAIL")
        sys.exit(1)

    threads = args.threads or manifest["threads"]
    seed = random.SystemRandom().randrange(10**9)
    fuzz_fail = fuzz_gate(args.fuzz, seed)
    mode = "quick" if args.quick else ("holdout" if args.holdout else "dev")
    summary = dict(mode=mode, fuzz_seed=seed, fuzz_cases=args.fuzz, fuzz_failures=fuzz_fail, threads=threads)
    rows = manifest["holdout" if args.holdout else "dev"]
    if not args.quick:
        results, bench_fail = bench(rows, threads) if fuzz_fail == 0 else ([], -1)
        summary.update(bench_failures=bench_fail, per_dataset=results)
        if fuzz_fail == 0 and bench_fail == 0:
            summary.update(geomean_s=round(geomean([r["seconds"] for r in results]), 6),
                           speedup_vs_ref=round(geomean([r["speedup"] for r in results]), 3),
                           speedup_vs_upstream_best=round(geomean([r["speedup_vs_upstream_best"] for r in results]), 3))
            real = [r for r in results if r["real"]]
            if real:
                summary.update(real_speedup_vs_ref=round(geomean([r["speedup"] for r in real]), 3),
                               real_speedup_vs_upstream_best=round(
                                   geomean([r["speedup_vs_upstream_best"] for r in real]), 3))
    gate = fuzz_fail == 0 and (args.quick or summary.get("bench_failures") == 0)
    summary.update(gate="PASS" if gate else "FAIL", eval_seconds=round(time.time() - t_start, 1))
    name = {"holdout": "last_eval_holdout.json"}.get(mode, "last_eval.json")
    (ROOT / name).write_text(json.dumps(summary, indent=1))

    print("---")
    print(f"gate:             {summary['gate']}")
    print(f"mode:             {mode}")
    print(f"fuzz:             {args.fuzz} cases, seed {seed}, failures {fuzz_fail}")
    if "geomean_s" in summary:
        print(f"geomean_s:        {summary['geomean_s']:.6f}")
        print(f"speedup_vs_ref:   {summary['speedup_vs_ref']:.3f}")
        print(f"speedup_vs_upstream_best: {summary['speedup_vs_upstream_best']:.3f}")
        if "real_speedup_vs_ref" in summary:
            print(f"real_speedup_vs_ref: {summary['real_speedup_vs_ref']:.3f}   "
                  f"real_speedup_vs_upstream_best: {summary['real_speedup_vs_upstream_best']:.3f}")
        print("per_dataset:      " + "  ".join(f"{r['name']}={r['seconds']:.3f}s({r['speedup']:.1f}x)"
                                               for r in summary["per_dataset"]))
    print(f"threads:          {threads}")
    print(f"eval_seconds:     {summary['eval_seconds']}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
