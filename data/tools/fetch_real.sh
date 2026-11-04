#!/bin/bash
# Fetch the raw GenBank records and align them with Nextclade 3.24.0.
# Usage: bash fetch_real.sh [RAW_DIR]   (default ./realdata_raw). Then run make_real_data.py on RAW_DIR.
set -e
RAW=${1:-realdata_raw}; mkdir -p "$RAW" && cd "$RAW"
[ -x nextclade ] || { curl -sL -o nextclade https://github.com/nextstrain/nextclade/releases/download/3.24.0/nextclade-x86_64-unknown-linux-gnu; chmod +x nextclade; }
./nextclade --version
E=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
fetch () { # query retmax out
  curl -s "$E/esearch.fcgi?db=nuccore&usehistory=y&retmax=0&term=$1" > es_$3.xml
  WE=$(grep -o '<WebEnv>[^<]*' es_$3.xml | cut -c9-); QK=$(grep -o '<QueryKey>[^<]*' es_$3.xml | cut -c11-); CNT=$(grep -o '<Count>[^<]*' es_$3.xml | head -1 | cut -c8-)
  echo "$3 count=$CNT"
  : > $3.fasta
  for ((s=0; s<$2; s+=500)); do
    curl -s "$E/efetch.fcgi?db=nuccore&query_key=$QK&WebEnv=$WE&rettype=fasta&retmode=text&retstart=$s&retmax=500" >> $3.fasta
    sleep 0.4
  done
  grep -c '>' $3.fasta
}
HIVQ='%22Human+immunodeficiency+virus+1%22%5BOrganism%5D+AND+pol%5BTitle%5D+AND+1000%3A1600%5BSLEN%5D+AND+2018%3A2026%5BPDAT%5D'
SARSQ='%22Severe+acute+respiratory+syndrome+coronavirus+2%22%5BOrganism%5D+AND+%22complete+genome%22%5BTitle%5D+AND+29000%3A30000%5BSLEN%5D+AND+2024%3A2026%5BPDAT%5D'
[ -s hiv_raw.fasta ] || fetch "$HIVQ" 9000 hiv_raw
[ -s sars_raw.fasta ] || fetch "$SARSQ" 1000 sars_raw
date -u > fetched_at.txt
./nextclade dataset get --name community/neherlab/hiv-1/hxb2 --output-dir ds_hiv
./nextclade dataset get --name nextstrain/sars-cov-2/wuhan-hu-1/orfs --output-dir ds_sars
./nextclade run -j 4 -D ds_hiv --output-fasta hiv_aln.fasta --output-tsv hiv_nc.tsv hiv_raw.fasta
./nextclade run -j 4 -D ds_sars --output-fasta sars_aln.fasta --output-tsv sars_nc.tsv sars_raw.fasta
grep -c '>' hiv_aln.fasta sars_aln.fasta
echo DONE
