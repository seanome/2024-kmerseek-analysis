"""Gene-level lengthScaledTPM-style aggregation, using complete transcript mapping."""
import csv,gzip,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
mapping={}
with gzip.open(R/'sources/feature_table.txt.gz','rt') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  if row['# feature']=='mRNA' and row['GeneID']:mapping[row['product_accession']]=row['GeneID']
with (R/'sources/transcript_gene.tsv').open('w') as f:
 f.write('transcript\tgene\n');f.writelines(f'{k}\t{v}\n' for k,v in mapping.items())
samples=pd.read_csv(R/'samples.tsv',sep='\t')['sample'];tpms={};lengths={};totals={};status={}
for sample in samples:
 d=pd.read_csv(R/'baseline'/sample/'abundance.tsv',sep='\t');d['gene']=d['target_id'].map(mapping);unmapped=float(d.loc[d.gene.isna(),'est_counts'].sum());d=d.dropna(subset=['gene'])
 d['weighted_length']=d.tpm*d.eff_length
 group=d.groupby('gene');tpm=group.tpm.sum();length=(group.weighted_length.sum()/tpm).replace([np.inf,-np.inf],np.nan)
 tpms[sample]=tpm;lengths[sample]=length;totals[sample]=float(d.est_counts.sum());status[sample]=dict(mapped_gene_est_counts=totals[sample],unmapped_est_counts=unmapped)
tpm=pd.DataFrame(tpms).fillna(0);length=pd.DataFrame(lengths).mean(axis=1)
# Missing lengths occur only in all-zero genes; their output stays zero.
weighted=tpm.mul(length.fillna(1),axis=0)
counts=weighted.div(weighted.sum(axis=0),axis=1).mul(pd.Series(totals),axis=1)
counts.index=counts.index.astype(str);counts.to_csv(R/'baseline/gene_counts.tsv',sep='\t',index_label='gene_id')
(R/'baseline/aggregation.json').write_text(json.dumps(dict(method='gene TPM times across-sample mean TPM-weighted effective gene length; per-sample scaling to total mapped transcript estimated counts; rounded in fit_de',sample_counts=status),indent=2))
print('Gene count matrix',counts.shape)
