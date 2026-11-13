"""Matched >=20aa diagnostic using Branchwater sketches and exact hash overlap."""
import csv,gzip,hashlib,json,os,re,time,subprocess,random
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
import sourmash
from sourmash.save_load import SaveSignaturesToLocation
from resource_usage import run_timed

SETTINGS=[('protein',7,{'is_protein':True}),('dayhoff',12,{'dayhoff':True}),('hp',30,{'hp':True})]
ROOT=Path(os.environ['GROUP_SCRATCH'])/'oyster-min20-pilot-20261025'
FULL=Path(os.environ['SCRATCH'])/'oyster-full-extensions-20261023'
DE=Path(os.environ['SCRATCH'])/'oyster-dekupl-queries-20261024'

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def parts(dna):
 for strand in (Seq(dna),Seq(dna).reverse_complement()):
  for frame in range(3):
   n=(len(strand)-frame)//3*3
   if n>0:
    for peptide in re.split('[*X]',str(strand[frame:frame+n].translate())):
     if peptide:yield peptide

def sample(rows,n=1000):
 if len(rows)<n:raise ValueError('Insufficient sample')
 return [rows[i*len(rows)//n] for i in range(n)]

def main():
 ROOT.mkdir() # Never overwrite a previous pilot.
 start=time.monotonic()
 p=json.loads((FULL/'summary-progress.json').read_text());b=p['completed_batches'][0]
 src=FULL/'splash-trimmed-all'/f"batch-{b['batch']:03d}"/'compactors.tsv'
 if sha(src)!=b['output_sha256']:raise ValueError('SPLASH source mismatch')
 with src.open() as f:sp=[(f"splash:full:{b['batch']}:{i}:{r['anchor']}",r['compactor']) for i,r in enumerate(csv.DictReader(f,delimiter='\t')) if len(r['compactor'])>len(r['anchor'])]
 with gzip.open(DE/'all-significant-contigs.tsv.gz','rt') as f:de=[(r['query_id'],r['contig']) for r in csv.DictReader(f,delimiter='\t') if float(r['pvalue'])<.05]
 selection=json.loads((DE/'selection.json').read_text())
 if len(de)!=selection['selected_contigs'] or sha(Path(selection['source']))!=selection['source_sha256']:raise ValueError('DE source mismatch')
 queries=[('splash',key,dna) for key,dna in sample(sp)]+[('dekupl',key,dna) for key,dna in sample(de)]
 prior=Path(os.environ['GROUP_SCRATCH'])/'oyster-stopfree-public-pilot-linear-20261025'
 meta=json.loads((prior/'validated.json').read_text())['reference_metadata'];refs=[]
 cut=Path(os.environ['GROUP_HOME'])/'opt/oyster-mvp/envs/shazam/bin/cutadapt'
 tmp=Path(os.environ['L_SCRATCH_JOB']);raws=[]
 urls=meta['fastq_ftp'].split(';');md5s=meta['fastq_md5'].split(';')
 if len(urls)!=len(md5s):raise ValueError('Source list mismatch')
 for i,(url,md5) in enumerate(zip(urls,md5s)):
  raw=tmp/f'pilot-{i}.fq.gz';raws.append(raw)
  run_timed(['curl','--fail','--location','--retry','3','--speed-limit','1024','--speed-time','60','-o',raw,'https://'+url],ROOT/f'download-{i}.log')
  with raw.open('rb') as f:
   if hashlib.file_digest(f,'md5').hexdigest()!=md5:raise ValueError('ENA MD5 mismatch')
  run_timed(['trim_galore','--path_to_cutadapt',cut,'--cores','1','--quality','20','--length','20','--gzip','--output_dir',tmp,raw],ROOT/f'trim-{i}.log')
  trimmed=tmp/f'pilot-{i}_trimmed.fq.gz'
  with gzip.open(trimmed,'rt') as f:refs += [('reference',f'read:{i}:{j}',str(r.seq)) for j,r in enumerate(SeqIO.parse(f,'fastq'))]
 records=queries+refs
 (ROOT/'sample.tsv').write_text('group\tid\tdna_sha256\n'+''.join(f'{g}\t{k}\t{hashlib.sha256(d.encode()).hexdigest()}\n' for g,k,d in records))
 result={};all_sketches={}
 for threshold in (1,20):
  arm=ROOT/f'min{threshold}';arm.mkdir();fa=arm/'stretches.fa';stats=defaultdict(lambda:dict(records=0,represented=0,stretches=0,amino_acids=0));expected={};seqs={}
  with fa.open('w') as f:
   for i,(g,key,dna) in enumerate(records):
    ps=[s for s in parts(dna) if len(s)>=threshold];st=stats[g];st['records']+=1;st['represented']+=bool(ps);st['stretches']+=len(ps);st['amino_acids']+=sum(map(len,ps))
    for j,s in enumerate(ps):
     name=f'{i}:{j}';expected[name]=i;seqs[name]=s;f.write(f'>{name}\n{s}\n')
  listing=arm/'input.csv'
  with listing.open('w') as f:
   w=csv.writer(f);w.writerow(['name','genome_filename','protein_filename']);w.writerow(['stretches','',str(fa)])
  cmd=['sourmash','scripts','manysketch',listing,'-o',arm/'segments.zip','--singleton','-c','4']
  for mol,k,_ in SETTINGS:cmd+=['-p',f'{mol},k={k},scaled=10']
  run_timed(cmd,arm/'manysketch.log')
  merged={mol:[sourmash.MinHash(n=0,ksize=k,scaled=10,**flags) for _ in records] for mol,k,flags in SETTINGS};seen=set()
  for sig in sourmash.load_file_as_signatures(str(arm/'segments.zip')):
   name=sig.name;mol=sig.minhash.moltype.lower()
   if name not in expected or mol not in merged or (name,mol) in seen:raise ValueError('Unexpected/duplicate segment sketch')
   seen.add((name,mol));merged[mol][expected[name]].merge(sig.minhash)
   # Exact independent peptide API comparison on each segment.
   k,flags=next((k,flags) for m,k,flags in SETTINGS if m==mol)
   check=sourmash.MinHash(n=0,ksize=k,scaled=10,**flags);check.add_protein(seqs[name])
   if set(check.hashes)!=set(sig.minhash.hashes):raise ValueError('Branchwater peptide hash mismatch')
  if seen!={(n,m) for n in expected for m,_,_ in SETTINGS}:raise ValueError('Missing segment signatures')
  measurements=[]
  for mol,k,flags in SETTINGS:
   for scale in (10,100):
    sketches=[m.downsample(scaled=scale) for m in merged[mol]];all_sketches[threshold,mol,scale]=sketches
    ref=sourmash.MinHash(n=0,ksize=k,scaled=scale,**flags)
    for mh in sketches[len(queries):]:ref.merge(mh)
    reference_hashes=set(ref.hashes)
    for group in ('splash','dekupl'):
     indices=[i for i,(g,_,_) in enumerate(queries) if g==group];path=arm/f'{group}-{mol}-{scale}.sig.gz'
     with SaveSignaturesToLocation(str(path)) as save:
      for i in indices:save.add(sourmash.SourmashSignature(sketches[i],name=records[i][1]))
     hits={records[i][1] for i in indices if reference_hashes.intersection(sketches[i].hashes)}
     measurements.append(dict(group=group,moltype=mol,scaled=scale,queries=len(indices),nonempty=sum(bool(len(sketches[i])) for i in indices),hashes=sum(len(sketches[i]) for i in indices),signature_bytes=path.stat().st_size,matches=len(hits),match_ids=sorted(hits)))
    path=arm/f'reference-{mol}-{scale}.sig.gz'
    with SaveSignaturesToLocation(str(path)) as save:save.add(sourmash.SourmashSignature(ref,name=meta['run_accession']))
    measurements.append(dict(group='reference',moltype=mol,scaled=scale,records=len(refs),hashes=len(ref),signature_bytes=path.stat().st_size))
  result[str(threshold)]=dict(stretches=dict(stats),measurements=measurements,segment_comparisons=len(seen),manysketch_sha256=sha(arm/'segments.zip'))
 for mol,_,_ in SETTINGS:
  for scale in (10,100):
   for before,after in zip(all_sketches[1,mol,scale],all_sketches[20,mol,scale]):
    if set(after.hashes)-set(before.hashes):raise ValueError('Threshold gained hashes')
   if mol=='hp' and any(set(a.hashes)!=set(b.hashes) for a,b in zip(all_sketches[1,mol,scale],all_sketches[20,mol,scale])):raise ValueError('HP30 changed under min20')
 out=dict(protocol='all-six-frames split stop/X; compare minimum1 vs20aa; DNA unchanged',scope='1000 systematic DE contigs,1000 systematic first-batch SPLASH extensions; entire small public run',reference=meta,source_hashes=dict(splash=sha(src),de_export=sha(DE/'all-significant-contigs.tsv.gz'),sample=sha(ROOT/'sample.tsv')),arms=result,wall_seconds=time.monotonic()-start,match_method='Exact nonzero shared sampled hashes; independent of search index; candidate matches only')
 (ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print('PILOT COMPLETE',flush=True)

if __name__=='__main__':main()
