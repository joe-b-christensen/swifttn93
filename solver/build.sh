#!/bin/bash
# Builds solver/swifttn93. evaluate.py runs it from the repository root.
# -ffp-contract=off: no FMA contraction, so every floating-point expression is evaluated exactly as in tn93.
# -static: about 1 ms less process start-up (no dynamic loading of libstdc++/libgomp/libm). The static libm's
# log is the same glibc 2.39 code (checked in a -static build: 103.6M inputs, 0 mismatches against
# the vector replica), and the start-up self-test still checks the replica against the linked libm.
# (ld warns that libgomp's offloading code uses dlopen; offloading is never used.)
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -march=native -ffp-contract=off -std=c++17 -fopenmp -static -DVERSION_NUMBER=\"v1.1.0\" -o swifttn93 \
  swifttn93.cpp refkernel.cpp argparse.cpp
