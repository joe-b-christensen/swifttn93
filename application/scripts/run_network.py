#!/usr/bin/env python3
"""Application: HIV-1 pol network at one TN93 threshold. Appends one JSON line per threshold.
usage: run_network.py RESULTS.jsonl T [T ...]
Pair lists are deleted after each threshold; only aggregate statistics are kept."""
import hashlib, json, os, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = f"{ROOT}/application/work"
TN93 = f"{A}/ref/tn93/tn93_ref"
SWIFT = f"{ROOT}/solver/swifttn93"
IN = f"{A}/data/hiv_pol_8766.fa"
OUT = f"{A}/out"
N_SEQ = 8766


def flags(t):
    return ["-q", "-0", "-t", t, "-a", "resolve", "-l", "500", "-g", "0.05", "-f", "csv"]


def run(binary, t, threads, tag, extra=()):
    o = f"{OUT}/{tag}.csv"
    s = f"{OUT}/{tag}.json"
    if os.path.exists(o):
        os.remove(o)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    with open(s, "wb") as fh:
        r = subprocess.run([binary, *extra, *flags(t), "-o", o, IN], stdout=fh, stderr=subprocess.PIPE, env=env)
    return time.perf_counter() - t0, r.returncode, o, s


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lines_sha(p):
    r = subprocess.run(f"LC_ALL=C sort -S 200M '{p}' | sha256sum", shell=True, capture_output=True, text=True)
    return r.stdout.split()[0]


def network(p):
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    links = selfs = 0
    names = set()
    maxd = 0.0
    with open(p) as f:
        head = next(f).strip()
        for line in f:
            a, b, d = line.rstrip("\n").split(",")
            if a == b:
                selfs += 1
                names.add(a)
                continue
            links += 1
            maxd = max(maxd, float(d))
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
    size = {}
    for x in list(par):
        root = find(x)
        size[root] = size.get(root, 0) + 1
    in_cluster = sum(size.values())
    return dict(header=head, links=links, self_rows=selfs, names_in_self_rows=len(names), max_link_distance=maxd,
                clusters=len(size), largest=max(size.values()) if size else 0, sequences_in_clusters=in_cluster,
                share_in_clusters=in_cluster / N_SEQ, clusters_size2=sum(1 for v in size.values() if v == 2))


res_path = sys.argv[1]
for t in sys.argv[2:]:
    r = {"threshold": float(t), "t_arg": t}
    tag = f"t{t}"
    sec, rc, o_ref, s_ref = run(TN93, t, 1, f"{tag}_ref")
    r["tn93_1t_s"], r["tn93_1t_rc"] = sec, rc
    r["ref_file_sha256"], r["ref_stdout_sha256"], r["file_bytes"] = sha(o_ref), sha(s_ref), os.path.getsize(o_ref)
    r["network"] = network(o_ref)
    ref_sorted = lines_sha(o_ref)
    sec, rc, o, s = run(TN93, t, 2, f"{tag}_ref2")
    r["tn93_2t_s"], r["tn93_2t_rc"] = sec, rc
    r["tn93_2t_file_identical"] = sha(o) == r["ref_file_sha256"]
    r["tn93_2t_stdout_identical"] = sha(s) == r["ref_stdout_sha256"]
    r["tn93_2t_same_lines"] = lines_sha(o) == ref_sorted
    sec, rc, o, s = run(TN93, t, 2, f"{tag}_S", extra=["-S"])
    r["tn93_S_2t_s"], r["tn93_S_2t_rc"] = sec, rc
    r["S_file_identical"] = sha(o) == r["ref_file_sha256"]
    r["S_stdout_identical"] = sha(s) == r["ref_stdout_sha256"]
    r["S_same_lines"] = lines_sha(o) == ref_sorted
    r["S_network"] = network(o)
    try:
        js = json.load(open(s))
        rs = json.load(open(s_ref))
        keys = ("Actual comparisons performed", "Mean distance", "Maximum distance")
        r["S_summary"] = {k: js.get(k) for k in keys}
        r["ref_summary"] = {k: rs.get(k) for k in keys}
    except Exception as e:
        r["summary_parse_error"] = str(e)
    for th in (1, 2):
        ts = []
        same = []
        for i in range(3):
            sec, rc, o, s = run(SWIFT, t, th, f"{tag}_swift{th}")
            ts.append(sec)
            same.append(rc == 0 and sha(o) == r["ref_file_sha256"] and sha(s) == r["ref_stdout_sha256"])
        r[f"swift_{th}t_runs_s"] = ts
        r[f"swift_{th}t_s"] = statistics.median(ts)
        r[f"swift_{th}t_identical"] = all(same)
    with open(res_path, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(t, {k: r[k] for k in ("tn93_1t_s", "tn93_2t_s", "tn93_S_2t_s", "swift_1t_s", "swift_2t_s",
                                "swift_1t_identical", "swift_2t_identical", "S_same_lines")}, flush=True)
    for name in os.listdir(OUT):
        os.remove(f"{OUT}/{name}")
