"""Independently recount DE overlap and rank correlation from exported gene rows."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / 'results/quantifier-comparison'
for line in (D / 'result-hashes.txt').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    if hashlib.sha256((D / Path(name).name).read_bytes()).hexdigest() != digest:
        raise ValueError('Downloaded result hash mismatch')
rows = list(csv.DictReader((D / 'gene-comparison.tsv').open(), delimiter='\t'))
if len({r['gene_id'] for r in rows}) != len(rows):
    raise ValueError('Duplicate genes')
k = {r['gene_id'] for r in rows if float(r['padj_kallisto']) < .05}
s = {r['gene_id'] for r in rows if float(r['padj_salmon']) < .05}
opposite = [r for r in rows if r['gene_id'] in k & s and
            float(r['log2FoldChange_kallisto']) * float(r['log2FoldChange_salmon']) < 0]

def ranks(xs):
    ordered = sorted(range(len(xs)), key=xs.__getitem__)
    out = [0.] * len(xs)
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[ordered[j]] == xs[ordered[i]]:
            j += 1
        for pos in ordered[i:j]:
            out[pos] = (i+j-1)/2
        i = j
    return out

pairs = [(float(r['log2FoldChange_kallisto']), float(r['log2FoldChange_salmon']))
         for r in rows if r['log2FoldChange_kallisto'] != 'NA' and r['log2FoldChange_salmon'] != 'NA']
x, y = [ranks([p[j] for p in pairs]) for j in (0, 1)]
mx, my = sum(x)/len(x), sum(y)/len(y)
rho = sum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
summary = dict(common_tested=len(rows), kallisto_DE=len(k), salmon_DE=len(s), both_DE=len(k&s),
               kallisto_only=len(k-s), salmon_only=len(s-k), opposite_DE=len(opposite),
               lfc_spearman=rho, opposite_direction_genes=opposite,
               scope='Six complete paired libraries; condition-only R DESeq2 on a shared tested gene universe',
               rule='BH adjusted P strictly below 0.05; no additional fold-change cutoff',
               verification='All downloaded comparison output SHA256 hashes matched; overlap and rank correlation independently recomputed from gene rows')
reported = next(csv.DictReader((D / 'summary.tsv').open(), delimiter='\t'))
for key in ('common_tested','kallisto_DE','salmon_DE','both_DE','opposite_DE'):
    if int(reported[key]) != summary[key]:
        raise ValueError('Summary count mismatch')
if abs(float(reported['lfc_spearman'])-rho)>1e-10:
    raise ValueError('Rank correlation mismatch')
(D / 'audited-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='opposite_direction_genes'}))
