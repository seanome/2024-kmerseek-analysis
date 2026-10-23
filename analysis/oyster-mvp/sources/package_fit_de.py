"""Condition-aware NB model; gold-standard genes are not read by this script."""
import argparse,json,time,math
from pathlib import Path
import numpy as np,pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import false_discovery_control
p=argparse.ArgumentParser();p.add_argument('--counts',required=True);p.add_argument('--samples',required=True);p.add_argument('--out',required=True);a=p.parse_args()
start=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
counts=pd.read_csv(a.counts,sep='\t',index_col=0).round().astype('int64')
metadata=pd.read_csv(a.samples,sep='\t').set_index('sample')[['condition']]
counts=counts.loc[:,metadata.index]
if counts.index.has_duplicates:raise ValueError('duplicate features')
if (counts.values<0).any():raise ValueError('negative counts')
keep=(counts.ge(5).sum(axis=1)>=3)&(counts.sum(axis=1)>=30)
filtered=counts.loc[keep]
print('Features input',len(counts),'retained',len(filtered),flush=True)
dds=DeseqDataSet(counts=filtered.T,metadata=metadata,design='~condition',refit_cooks=True,n_cpus=4,quiet=False)
dds.deseq2()
stats=DeseqStats(dds,contrast=['condition','breed','wild'],alpha=.05,cooks_filter=True,independent_filter=False,n_cpus=4)
stats.summary();result=stats.results_df.copy()
# Correct over the full count-filtered family, including Cook-filtered/undefined rows as p=1.
result['padj_model']=result['padj'];ps=result['pvalue'].fillna(1).clip(0,1).to_numpy();result['padj']=false_discovery_control(ps)
result['significant']=(result['padj']<.05)&(result['log2FoldChange'].abs()>math.log2(1.5))
result['feature']=result.index;result['abs_lfc']=result['log2FoldChange'].abs();result=result.sort_values(['padj','abs_lfc','feature'],ascending=[True,False,True]).drop(columns=['feature','abs_lfc'])
result.to_csv(out/'de_results.tsv',sep='\t',index_label='feature')
factors=np.asarray(dds.obs['size_factors']).reshape(-1)
normalized=filtered.div(pd.Series(factors,index=metadata.index),axis=1);normalized.to_csv(out/'normalized_counts.tsv',sep='\t',index_label='feature')
meta=dict(input_features=len(counts),tested_features=len(filtered),significant=int(result['significant'].sum()),size_factors=dict(zip(metadata.index,map(float,factors))),seconds=time.monotonic()-start,contrast='breed/wild',pydeseq2='0.5.4',independent_filter=False,family_correction='BH across every count-filtered feature, undefined p assigned 1',cooks_filter=True)
(out/'fit.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
