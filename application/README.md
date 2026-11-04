# swifttn93 network analysis

The HIV-1 pol network at five TN93 distance thresholds (preprint Section 2.5, Tables 3 and S6). The input is all 8,766
sequences of `data/real_hiv_pol_dev` and `data/real_hiv_pol_holdout`, sha256
70985c02b3ba52585c9ed30db147ca14984a9f5fd03babb3af235432062a0524 after concatenation. The flags are HIV-TRACE's:
`tn93 -q -0 -o OUT -t T -a resolve -l 500 -g 0.05 -f csv`.

To rerun, from the repository root (needs g++, git, xz, network; everything it makes goes in `application/work/`):

    bash application/scripts/build.sh
    python3 application/scripts/run_network.py application/work/network.jsonl 0.005 0.0075 0.01 0.015 0.02
    python3 application/scripts/ambiguity_counts.py application/work/ambiguity_counts.json

`run_network.py` deletes the pair lists after each threshold. Only aggregate statistics are kept. The published ones are
`results/network.jsonl`, one line per threshold, and `results/ambiguity_counts.json`.

Links are the pairs tn93 writes, excluding the self-pairs that `-0` adds. Clusters are connected components of size 2 or
more.

| Threshold | Links | Clusters | Largest cluster | Sequences in clusters | Share of 8,766 |
|----------:|------:|---------:|----------------:|----------------------:|---------------:|
| 0.5% | 3,739 | 647 | 31 | 1,920 | 21.9% |
| 0.75% | 4,609 | 751 | 47 | 2,267 | 25.9% |
| 1% | 5,876 | 839 | 61 | 2,639 | 30.1% |
| 1.5% | 8,405 | 956 | 114 | 3,288 | 37.5% |
| 2% | 13,505 | 967 | 366 | 3,926 | 44.8% |

Wall time in seconds on a laptop without AVX-512 (AMD Ryzen 5 5600H), with other jobs running. tn93 and `-S` ran once per
threshold; swifttn93 is the median of three.

| Threshold | tn93 1 thread | tn93 2 threads | `tn93 -S` 2 threads | swifttn93 1 thread | swifttn93 2 threads |
|----------:|--------------:|---------------:|--------------------:|-------------------:|--------------------:|
| 0.5% | 42.6 | 21.8 | 0.47 | 6.79 | 3.29 |
| 0.75% | 37.5 | 19.9 | 0.86 | 6.38 | 3.35 |
| 1% | 39.1 | 20.2 | 0.59 | 5.90 | 3.12 |
| 1.5% | 38.9 | 20.8 | 0.75 | 6.26 | 3.18 |
| 2% | 37.0 | 19.2 | 0.93 | 6.00 | 3.29 |
| All five | 195 | 102 | 3.6 | 31 | 16 |

At every threshold swifttn93, at one and two threads, wrote the file and summary of tn93 at one thread byte for byte. tn93
at two threads wrote the same lines in a different order. `tn93 -S` gave the same links, and so the same clusters, with a
summary of only the pairs it kept.

The sequences carry 6.3 non-ACGT, non-gap characters on average, median 2, mostly R and Y.
