#!/bin/bash
# Builds the reference tn93 (pinned, prepare.py's flags) and SwiftTN93, and makes the analysis input from the two real
# HIV-1 pol sets. Everything it makes goes in application/work/.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../.." && pwd); A=$R/application/work
mkdir -p "$A/ref" "$A/data" "$A/out"
if [ ! -x "$A/ref/tn93/tn93_ref" ]; then
  [ -d "$A/ref/tn93" ] || git clone -q https://github.com/veg/tn93 "$A/ref/tn93"
  git -C "$A/ref/tn93" checkout -q 076eb0f
  cd "$A/ref/tn93"
  SRC=$(python3 "$S/ref_sources.py" "$R/prepare.py")
  g++ -O3 -std=c++14 -funroll-loops -fopenmp -DVERSION_NUMBER=\"v1.1.0\" -o tn93_ref $SRC
fi
"$A/ref/tn93/tn93_ref" -v || true
if [ ! -x "$R/solver/swifttn93" ]; then bash "$R/solver/build.sh" 2>/dev/null; fi
ls -la "$R/solver/swifttn93" "$A/ref/tn93/tn93_ref"
grep -m1 'model name' /proc/cpuinfo; grep -c avx512 /proc/cpuinfo || true
xz -dc "$R/data/real_hiv_pol_dev.fa.xz" > "$A/data/hiv_pol_8766.fa"
xz -dc "$R/data/real_hiv_pol_holdout.fa.xz" >> "$A/data/hiv_pol_8766.fa"
grep -c '>' "$A/data/hiv_pol_8766.fa"; sha256sum "$A/data/hiv_pol_8766.fa"
awk 'NR==2{print length($0)}' "$A/data/hiv_pol_8766.fa"; head -c 40 "$A/data/hiv_pol_8766.fa"; echo
sha256sum "$R/data/real_hiv_pol_dev.fa.xz" "$R/data/real_hiv_pol_holdout.fa.xz"
