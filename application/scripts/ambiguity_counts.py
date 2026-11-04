#!/usr/bin/env python3
"""Count non-ACGT, non-gap characters per sequence in the analysis input. usage: ambiguity_counts.py OUT.json"""
import collections
import json
import os
import statistics
import sys

seq, per, c = [], [], collections.Counter()


def done():
    codes = [x for x in "".join(seq) if x not in "ACGT-"]
    per.append(len(codes))
    c.update(codes)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for line in open(f"{ROOT}/application/work/data/hiv_pol_8766.fa"):
    if line[0] == ">":
        if seq:
            done()
        seq = []
    else:
        seq.append(line.strip().upper())
done()
json.dump(dict(sequences=len(per), mean=sum(per) / len(per), median=statistics.median(per), zero=per.count(0),
               by_code=c.most_common()), open(sys.argv[1], "w"), indent=1)
print(len(per), round(sum(per) / len(per), 2), statistics.median(per), c.most_common(4))
