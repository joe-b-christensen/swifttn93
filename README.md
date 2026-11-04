# swifttn93

swifttn93 is a drop-in replacement for [tn93](https://github.com/veg/tn93) v1.1.0, the Tamura-Nei distance engine of
HIV-TRACE. It writes tn93's single-thread distance file and JSON summary, byte for byte, at any thread count, much
faster than tn93's standard mode.

On three real HIV-1 sets at the 1.5% threshold, at two threads on an AVX-512 CPU, it was 25- to 30-fold faster than
standard tn93 and 1.2- to 1.4-fold faster than tn93's filtered mode, `-S`, which reports the same links but summarises
only the pairs that pass its filter. At one thread swifttn93 took 1.4- to 1.6-fold longer than `-S` at two. On an AVX2
laptop it was 6-fold faster than standard tn93, and `-S` was 3.5- to 7-fold faster than swifttn93. A pipeline that needs
only the links is well served by `-S`; swifttn93 is for pipelines that need tn93's standard summary, every pair written,
or files that match an existing tn93 run. The standard-mode run that takes tn93 29 s at two threads on 6,687 sequences
takes swifttn93 1.2 s on an AVX-512 CPU. Details are in the preprint (see Citation).

## Build

Requirements: Linux on x86-64, g++ with C++17 and OpenMP, and static C, C++ and OpenMP libraries (`build-essential` on
Debian/Ubuntu). Tested with g++ 13.3.0 and glibc 2.39 on Ubuntu 24.04.

```
bash solver/build.sh        # produces solver/swifttn93
```

The build uses `-march=native`, so the binary is tuned for the build machine's CPU. The fastest code paths need AVX-512;
on other x86-64 CPUs portable paths give the same output, more slowly. The linker warning about `dlopen` concerns
libgomp's offloading code, which swifttn93 does not use.

## Usage

swifttn93 takes the same command line as tn93:

```
solver/swifttn93 -q -t 0.015 -a resolve -g 0.05 -l 500 -o out.csv in.fasta > summary.json
OMP_NUM_THREADS=2 solver/swifttn93 -q -0 -t 0.015 -a resolve -g 0.05 -l 500 -f csv -o out.csv in.fasta > summary.json
```

| Group | Options |
|---|---|
| Input / output | `FASTA`, `-s SECOND_FASTA`, `-o OUTPUT`, `-f csv\|csvn\|hyphy`, `-D DELIMITER`, `-n`, `-c`, `-q` |
| Distances | `-t THRESHOLD`, `-w MINIMUM`, `-l OVERLAP`, `-a AMBIGS`, `-g FRACTION` |
| Other | `-d COUNTS_IN_NAME`, `-m`, `-0` |

- The distance file and standard output are byte-identical to `OMP_NUM_THREADS=1 tn93 -q` with the same options, at every
  thread count. Where tn93 exits non-zero, swifttn93 does as well.
- `OMP_NUM_THREADS` sets the number of threads (default: all CPUs).
- `-b`, `-r` and `-u`, which are random by design, and `-H` exit with an error. `-S` is accepted and ignored: swifttn93
  always writes the standard-mode output.
- Alignments of 65,535 columns or more use tn93's own per-pair code and were not benchmarked.
- swifttn93 needs more memory than tn93: up to 97 MB against 23 MB on the benchmark sets.

## Check it yourself

```
python3 prepare.py              # builds tn93 v1.1.0 (commit 076eb0f), prepares the benchmark sets and records tn93's
                                # outputs (needs g++, git, numpy, network; about 3 min on a 12-thread laptop)
python3 evaluate.py             # builds swifttn93; 300 random inputs and five benchmark sets compared byte for byte,
                                # plus timing
python3 evaluate.py --holdout   # the same with the five holdout sets
```

`evaluate.py` prints `gate: PASS` and exits 0 when every output matched. It times swifttn93 on all CPUs against tn93 at one
thread, so its speedups differ from the two-thread figures above and are smaller without AVX-512.

`data/` holds the four real sets made for the preprint, with their GenBank accession lists; the fifth, tn93's own
`data/large.fasta`, comes from tn93's repository (see `data/README.txt`). The simulated sets are regenerated from fixed seeds
by `gen_data.py`. `application/` holds the network analysis scripts and their aggregate results (see
`application/README.md`).

## Licence

MIT (`LICENSE`), as a derivative of tn93. swifttn93 includes MIT-licensed code from tn93 v1.1.0 and a re-implementation of
the logarithm of Arm's optimized-routines (MIT OR Apache-2.0 WITH LLVM-exception, used here under MIT); see `LICENSE` and
`solver/LICENSE_NOTE.txt`.

## Citation

Christensen J. SwiftTN93 writes tn93's exact output 25- to 30-fold faster than tn93 on AVX-512 CPUs and leads its filtered
mode 1.2- to 1.4-fold on HIV data. Preprint, 2026. Please also cite tn93
(https://github.com/veg/tn93) and HIV-TRACE (Kosakovsky Pond SL et al., Mol Biol Evol 35:1812–1819, 2018). See
`CITATION.cff`.
