import os,gzip,csv,json,hashlib,math
from pathlib import Path
root=Path(os.environ['SCRATCH'])/'oyster-mvp-20261023';source=root/'dekupl/full/out/wild_vs_breed_kmer_counts/merged-diff-counts.tsv.gz';out=Path(os.environ['SCRATCH'])/'oyster-dekupl-queries-20261024';out.mkdir(exist_ok=True)
with source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
count=0;selected=0;zero=0;fields=None
with gzip.open(source,'rt') as f,gzip.open(out/'all-significant-contigs.tsv.gz','wt') as dest,(out/'all-significant-contigs.fasta').open('w') as fasta:
 reader=csv.DictReader(f,delimiter='\t');fields=['query_id',*reader.fieldnames];writer=csv.DictWriter(dest,fieldnames=fields,delimiter='\t');writer.writeheader()
 for row in reader:
  count+=1;q=float(row['pvalue']);fc=float(row['log2FC']);dna=row['contig']
  if not math.isfinite(q) or not math.isfinite(fc) or set(dna)-set('ACGT'):raise ValueError('Invalid DE-kupl row')
  if q>=.05:continue
  if abs(fc)<math.log2(1.5)-1e-9:raise ValueError('Effect filter mismatch')
  key=f'dekupl:full:{count}:{hashlib.sha256(dna.encode()).hexdigest()[:16]}';row['query_id']=key;writer.writerow(row);fasta.write(f'>{key}\n{dna}\n');selected+=1;zero+=not q
meta=dict(source=str(source),source_sha256=digest,input_rows=count,selected_contigs=selected,zero_adjusted_p=zero,rule='Full-depth DE-kupl merged contigs with BH-adjusted pvalue strictly<0.05; no top-N cap; original effect/sample columns retained',adjustment_source='Upstream DESeq2_diff_method.R lines246-262 computes BH and assigns corrected values to output pvalue',query_set='dekupl:full')
(out/'selection.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
