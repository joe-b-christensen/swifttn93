# Cut, filter and split the Nextclade alignments from fetch_real.sh into the four real sets in data/.
# Usage: python3 make_real_data.py RAW_DIR OUT_DIR
import csv, lzma, random, os, sys
RAW, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
def rd(p):
    n=None; s=[]
    for line in open(p):
        line=line.rstrip()
        if line.startswith('>'):
            if n: yield n,''.join(s)
            n=line[1:]; s=[]
        else: s.append(line)
    if n: yield n,''.join(s)
A,B=1795,2860
hiv=[]
for n,s in rd(f'{RAW}/hiv_aln.fasta'):
    w=s[A:B]
    if len(w)-w.count('-')-w.count('N')>=800: hiv.append((n.split()[0],w))
print('hiv kept',len(hiv))
rng=random.Random(2026); rng.shuffle(hiv)
dev,hold=hiv[:5000],hiv[5000:]
qc={r['seqName'].split()[0]:r['qc.overallStatus'] for r in csv.DictReader(open(f'{RAW}/sars_nc.tsv'),delimiter='\t')}
sars=[(n.split()[0],s) for n,s in rd(f'{RAW}/sars_aln.fasta') if qc.get(n.split()[0]) in ('good','mediocre')]
rng.shuffle(sars); sd,sh=sars[:500],sars[500:]
print('sars kept',len(sars))
def w(name, recs):
    with lzma.open(f'{OUT}/{name}.fa.xz','wt',preset=9) as f:
        for n,s in recs: f.write(f'>{n}\n{s}\n')
    with open(f'{OUT}/{name}.accessions.txt','w') as f: f.write('\n'.join(n for n,_ in recs)+'\n')
    print(name, len(recs), len(recs[0][1]), os.path.getsize(f'{OUT}/{name}.fa.xz'))
w('real_hiv_pol_dev',dev); w('real_hiv_pol_holdout',hold); w('real_sars2_dev',sd); w('real_sars2_holdout',sh)
