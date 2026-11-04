#!/usr/bin/env python3
"""SwiftTN93: dataset and fuzz-case generators.

Benchmark sets are simulated with fixed seeds:
  * HIV-1 pol-like alignments (~1,300 bp): a root, subtypes ~8% apart (gamma site rates, transition bias),
    transmission clusters ~3% from their subtype and members 0.1-1.2% from the cluster centre. Partial coverage
    (leading/trailing gaps), codon indels, and IUPAC ambiguity codes at realistic rates.
  * SARS-CoV-2-like genomes (~29.9 kb): lineages a few mutations apart, amplicon drop-outs as N runs.
Fuzz cases are tiny adversarial FASTA files plus random tn93 flags.
"""
import random

import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
TS_MAP = np.array([2, 3, 0, 1])  # transitions: A<->G, C<->T
IUPAC = b"RYSWKMBDHVN"

# name, spec. `args` are the tn93 flags used for the benchmark (besides -q and -o OUT); {in2} is the second file.
# kind "real": public sequences shipped in data/ (see data/README.txt); "file": a file from the
# tn93 repo; "hiv"/"sars2": simulated. Real sets carry the paper's headline.
HIVTRACE = ["-t", "0.015", "-a", "resolve", "-g", "0.05", "-l", "500"]  # HIV-TRACE's tn93 invocation
DEV_SPECS = {
    "real_hiv_pol": dict(kind="real", src="data/real_hiv_pol_dev.fa.xz", args=HIVTRACE),
    "real_sars2": dict(kind="real", src="data/real_sars2_dev.fa.xz", args=["-t", "0.0002", "-l", "20000", "-0"]),
    "sim_hiv_allpairs_csvn": dict(kind="hiv", n=2600, seed=12, args=["-t", "1", "-f", "csvn"]),
    "sim_hiv_ambig_average": dict(kind="hiv", n=3500, seed=13, ambig=12.0, args=["-a", "average", "-t", "0.05",
                                                                              "-l", "500"]),
    "sim_hiv_two_file": dict(kind="hiv", n=2500, n2=2500, seed=15, args=["-t", "0.02", "-a", "resolve", "-g",
                                                                         "0.05", "-s", "{in2}"]),
}
HOLDOUT_SPECS = {
    "h_real_hiv_pol": dict(kind="real", src="data/real_hiv_pol_holdout.fa.xz", args=HIVTRACE),
    "h_real_sars2_allpairs": dict(kind="real", src="data/real_sars2_holdout.fa.xz", args=["-t", "1", "-D", ";"]),
    "h_real_tn93_large": dict(kind="file", src="ext/tn93/data/large.fasta", args=[]),
    "h_sim_gapmm_hyphy": dict(kind="hiv", n=1500, seed=21, args=["-a", "gapmm", "-f", "hyphy", "-t", "0.03"]),
    "h_sim_copynumbers": dict(kind="hiv", n=3000, seed=24, copies=True, args=["-t", "0.015", "-w", "0.001", "-n"]),
}


def _mutate(rng, seq, d, rates, kappa_ts=0.67):
    """Substitute each site with probability d*rate; transitions with probability kappa_ts."""
    seq = seq.copy()
    p = np.clip(d * rates, 0, 0.75)
    hit = np.nonzero(rng.random(seq.shape[0]) < p)[0]
    if hit.size:
        ts = rng.random(hit.size) < kappa_ts
        cur = seq[hit]
        idx = np.searchsorted(BASES, cur)  # BASES is sorted A,C,G,T
        new_idx = np.where(ts, TS_MAP[idx], (idx + rng.integers(1, 4, hit.size)) % 4)
        seq[hit] = BASES[new_idx]
    return seq


def _random_seq(rng, L, comp):
    return BASES[rng.choice(4, size=L, p=comp)]


def _sim_hiv(spec, rng, n):
    L = spec.get("L", 1302)
    rates = rng.gamma(0.5, 2.0, L)
    root = _random_seq(rng, L, [0.38, 0.17, 0.24, 0.21])
    subtypes = [_mutate(rng, root, 0.045, rates) for _ in range(spec.get("subtypes", 5))]
    seqs = []
    while len(seqs) < n:
        sub = subtypes[rng.integers(len(subtypes))]
        size = 1 if rng.random() < 0.55 else int(rng.geometric(0.25)) + 1
        centre = _mutate(rng, sub, 0.03, rates)
        indel = None
        if rng.random() < 0.15:
            s = int(rng.integers(30, L - 40)) // 3 * 3
            indel = (s, s + 3 * int(rng.integers(1, 4)))
        for _ in range(min(size, n - len(seqs))):
            s = _mutate(rng, centre, rng.uniform(0.001, 0.012), rates)
            if indel:
                s[indel[0]:indel[1]] = ord("-")
            seqs.append(s)
    order = rng.permutation(n)
    seqs = [seqs[i] for i in order]
    lam = spec.get("ambig", 2.0)
    for s in seqs:
        if rng.random() < 0.2:  # partial coverage
            a, b = int(rng.integers(0, 300)), int(rng.integers(0, 300))
            s[:a] = ord("-")
            if b:
                s[L - b:] = ord("-")
        k = rng.poisson(lam)
        if rng.random() < 0.01:
            k += int(0.05 * L)
        if k:
            pos = rng.integers(0, L, k)
            s[pos] = np.frombuffer(IUPAC, dtype=np.uint8)[rng.integers(0, len(IUPAC), k)]
    return seqs


def _sim_sars2(spec, rng, n):
    L = spec.get("L", 29903)
    rates = np.ones(L)
    root = _random_seq(rng, L, [0.30, 0.18, 0.20, 0.32])
    lineages = [_mutate(rng, root, 15.0 / L, rates) for _ in range(spec.get("lineages", 25))]
    seqs = []
    for _ in range(n):
        s = _mutate(rng, lineages[rng.integers(len(lineages))], rng.poisson(5) / L, rates)
        s[:int(rng.integers(20, 80))] = ord("N")
        s[L - int(rng.integers(20, 120)):] = ord("N")
        for _ in range(int(rng.integers(0, 4))):
            a = int(rng.integers(0, L - 400))
            s[a:a + int(rng.integers(200, 350))] = ord("N")
        if rng.random() < 0.3:
            k = int(rng.integers(1, 6))
            s[rng.integers(0, L, k)] = np.frombuffer(IUPAC, dtype=np.uint8)[rng.integers(0, len(IUPAC) - 1, k)]
        seqs.append(s)
    return seqs


def simulate(spec, which=1):
    """which=1: the main file. which=2: the second file of a two-file spec; both files are drawn from one shared
    simulation (same subtypes and clusters) and split, so that between-file links exist."""
    rng = np.random.default_rng(spec["seed"] * 7 + 1)
    n1, n2 = spec["n"], spec.get("n2", 0)
    seqs = _sim_hiv(spec, rng, n1 + n2) if spec["kind"] == "hiv" else _sim_sars2(spec, rng, n1 + n2)
    seqs, n = (seqs[:n1], n1) if which == 1 else (seqs[n1:], n2)
    rng = np.random.default_rng(spec["seed"] * 7 + 1 + which)
    names = []
    for i in range(n):
        nm = f"{spec['kind'].upper()}_{spec['seed']}_{which}_{i:06d}"
        if spec.get("copies"):
            nm += f":{int(rng.geometric(0.6))}"
        names.append(nm)
    return names, seqs


def write_fasta(names, seqs, path, wrap=0):
    with open(path, "wb") as f:
        for nm, s in zip(names, seqs):
            f.write(b">" + nm.encode() + b"\n")
            b = s.tobytes()
            if wrap:
                for i in range(0, len(b), wrap):
                    f.write(b[i:i + wrap] + b"\n")
            else:
                f.write(b + b"\n")


def write_dataset(spec, path, path2=None):
    names, seqs = simulate(spec, 1)
    write_fasta(names, seqs, path, wrap=60 if spec["kind"] == "sars2" else 0)
    if path2 is not None:
        n2, s2 = simulate(spec, 2)
        write_fasta(n2, s2, path2)
    return (len(seqs), len(seqs[0]))


# ---------------------------------------------------------------- fuzz

ALPH_CORE = b"ACGT"
ALPH_EXTRA = b"acgtRYSWKMBDHVNnrykm-"
ALPH_WEIRD = b"uU?.X*~ "  # characters the reader may drop or reject: used sparingly


def _fuzz_fasta(rng, n, L, alph_extra_p, names_mode, equal=True):
    weird_p = alph_extra_p * 0.1 if rng.random() < 0.1 else 0.0
    out, names = [], []
    for i in range(n):
        if names_mode == "dup" and i and rng.random() < 0.4:
            nm = names[rng.randrange(len(names))]
        elif names_mode == "copies":
            nm = f"s{i}:{rng.randint(1, 4)}" if rng.random() < 0.8 else f"s{i}"
        elif names_mode == "odd" and rng.random() < 0.3:
            nm = rng.choice(["", " ", "a b", "x,y", "q;r", "s\tt", "long" * 20])
        else:
            nm = f"seq{i}"
        names.append(nm)
        Li = L if equal or rng.random() < 0.7 else max(0, L + rng.randint(-5, 5))
        seq = bytes(rng.choice(ALPH_WEIRD) if rng.random() < weird_p else
                    rng.choice(ALPH_EXTRA) if rng.random() < alph_extra_p else rng.choice(ALPH_CORE)
                    for _ in range(Li))
        if rng.random() < 0.15:  # leading / trailing gaps
            a = rng.randint(0, max(0, Li // 3))
            seq = b"-" * a + seq[a:]
        out.append((nm, seq))
    if rng.random() < 0.5 and n >= 2:  # near-duplicates so that thresholds bite
        base = out[0][1]
        for j in range(1, n):
            if rng.random() < 0.5 and len(out[j][1]) == len(base):
                s = bytearray(base)
                for _ in range(rng.randint(0, 3)):
                    if s:
                        s[rng.randrange(len(s))] = rng.choice(b"ACGTRN-")
                out[j] = (out[j][0], bytes(s))
    return out


def _render(records, rng):
    wrap = rng.choice([0, 0, 0, 7, 60])
    nl = b"\r\n" if rng.random() < 0.05 else b"\n"
    buf = bytearray()
    for nm, s in records:
        buf += b">" + nm.encode() + nl
        if rng.random() < 0.05:
            buf += nl
        if wrap and s:
            for i in range(0, len(s), wrap):
                buf += s[i:i + wrap] + nl
        else:
            buf += s + nl
    if rng.random() < 0.05:
        buf = buf.rstrip(b"\r\n")
    return bytes(buf)


def fuzz_case(rng: random.Random):
    """Return (files, flags): files maps a placeholder ("in", "in2") to bytes; flags use {in2} for the 2nd file."""
    r = rng.random()
    if r < 0.02:
        return {"in": b""}, []
    if r < 0.04:
        return {"in": rng.choice([b">", b">a\n", b">a\nACGT\n>b\n", b"ACGT\n", b">a\nAC\n>b\nACG\n"])}, ["-l", "1"]
    n = rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 12, 20])
    L = rng.choice([1, 2, 5, 17, 40, 64, 65, 100, 127, 128, 129, 200, 300, 150, 250, 320, 400])
    extra_p = rng.choice([0.0, 0.0, 0.02, 0.1, 0.4])
    names_mode = rng.choice(["plain", "plain", "dup", "odd", "copies"])
    equal = rng.random() > 0.03
    files = {"in": _render(_fuzz_fasta(rng, n, L, extra_p, names_mode, equal), rng)}
    flags = []
    flags += ["-t", rng.choice(["0", "0.001", "0.015", "0.05", "0.1", "0.3", "0.75", "1", "2"])]
    flags += ["-l", str(rng.choice([1, 1, 1, 2, max(1, L // 4), max(1, L // 2), L, L + 1, 100]))]
    if rng.random() < 0.6:
        flags += ["-a", rng.choice(["resolve", "average", "skip", "gapmm", "RY", "RYSWKM", "N"])]
        if rng.random() < 0.4:
            flags += ["-g", rng.choice(["0", "0.01", "0.05", "0.5", "1"])]
    if rng.random() < 0.15:
        flags += ["-w", rng.choice(["0", "0.001", "0.02"])]
    fmt = rng.choice(["csv", "csv", "csvn", "hyphy"])
    flags += ["-f", fmt]
    for f, p in [("-c", 0.1), ("-n", 0.2), ("-0", 0.2)]:
        if rng.random() < p:
            flags.append(f)
    if rng.random() < 0.15:
        flags += ["-D", rng.choice([";", "|", ":"])]
    if names_mode == "copies" or rng.random() < 0.05:
        flags += ["-d", rng.choice([":", ":", "_"])]
    if rng.random() < 0.2:
        n2 = rng.choice([1, 2, 3, 5, 8])
        L2 = L if rng.random() < 0.95 else L + 1
        files["in2"] = _render(_fuzz_fasta(rng, n2, L2, extra_p, rng.choice(["plain", "dup", "copies"])), rng)
        flags += ["-s", "{in2}"]
        if rng.random() < 0.3:
            flags.append("-m")
    return files, flags
