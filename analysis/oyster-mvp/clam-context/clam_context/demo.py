"""Synthetic examples; no clam observations, biological annotations or real SPLASH statistics."""
import csv
import random
from pathlib import Path
from Bio.Data import CodonTable
from Bio.Seq import Seq
from .core import AA, DAYHOFF, build_index, analyze


def demo(output):
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Demo output must be new or empty')
    root.mkdir(parents=True,exist_ok=True)
    rng=random.Random(71)
    proteins=[''.join(rng.choices(sorted(AA), k=240)) for _ in range(3)]
    reduced=''.join(rng.choice([b for b in sorted(AA) if DAYHOFF[b]==DAYHOFF[a] and b!=a] or [a]) for a in proteins[1])
    codons={}
    for codon,aa in sorted(CodonTable.unambiguous_dna_by_id[1].forward_table.items()):codons.setdefault(aa,codon)
    dna=[''.join(codons[a] for a in p) for p in [proteins[0],reduced,proteins[2]]]
    # Third query has no corresponding protein. Fourth is too short for reliable sketch signal.
    dna.append('ATGGCTTGTGATGAATTTGGTCATATTAAA')
    (root/'proteins.fasta').write_text(f'>DEMO_EXACT Synthetic exact-match control\n{proteins[0]}\n>DEMO_DAYHOFF Synthetic reduced-alphabet control\n{proteins[1]}\n')
    anchors=[s[:27] for s in dna]
    def write(name,header,rows):
        with (root/name).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(header);w.writerows(rows)
    write('scores.tsv',['anchor','pval_opt_corrected','effect_size_bin'],[(a,10**(-7+i),.8-i*.1) for i,a in enumerate(anchors)])
    write('compactors.tsv',['anchor','compactor'],zip(anchors,dna))
    write('counts.tsv',['anchor','sample','condition','target','count'],
          [(a,f'{group}{rep}',group,target,(90 if (group=='A')==(target=='T1') else 10)+rep)
           for a in anchors for group in ['A','B'] for rep in range(1,4) for target in ['T1','T2']])
    build_index(root/'proteins.fasta',root/'proteins.sqlite',label='SYNTHETIC controls — not Swiss-Prot or clam data')
    analyze(root/'scores.tsv',root/'compactors.tsv',root/'proteins.sqlite',root/'results',counts=root/'counts.tsv',demo=True)
    (root/'README.txt').write_text('Synthetic fixture: DNA is reverse-translated from seeded random peptides. Score and count tables are fabricated for interface testing, NOT output of SPLASH. Search results are actually computed by sourmash. DEMO_EXACT should match standard protein; DEMO_DAYHOFF tests substitutions preserving Dayhoff classes. Unmatched and short controls are retained. No biological findings.\n')
    return str(root/'results'/'report.html')
