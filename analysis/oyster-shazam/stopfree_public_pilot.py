"""Validate separately labelled stop-free paired-scale queries on one complete public run."""
import csv,gzip,hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import sourmash
from sourmash.save_load import SaveSignaturesToLocation
from resource_usage import run_timed
from archive_search_results import save_matches
SETTINGS=[('dna',31,{}),('protein',7,{'is_protein':True}),('dayhoff',12,{'dayhoff':True}),('hp',30,{'hp':True})]
ROOT=Path(os.environ['GROUP_SCRATCH'])/'oyster-stopfree-public-pilot-20261024'
ARCHIVE=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
FULL=Path(os.environ['SCRATCH'])/'oyster-full-extensions-20261023'
DE=Path(os.environ['SCRATCH'])/'oyster-dekupl-queries-20261024'
CUTADAPT=Path(os.environ['GROUP_HOME'])/'opt/oyster-mvp/envs/shazam/bin/cutadapt'
def sha(p,alg='sha256'):
 with p.open('rb') as f:return hashlib.file_digest(f,alg).hexdigest()
def run(args,log):
 run_timed(args,log)
def sketch(dna,mol,k,flags):
 mh=sourmash.MinHash(n=0,ksize=k,scaled=10,**flags)
 if mol=='dna':mh.add_sequence(dna,force=True)
 else:
  for seq in (Seq(dna),Seq(dna).reverse_complement()):
   for frame in range(3):
    n=(len(seq)-frame)//3*3
    if n<=0:continue
    peptide=str(seq[frame:frame+n].translate())
    for part in re.split('[*X]',peptide):
     if len(part)>=k:mh.add_protein(part)
 return mh

def preflight():
 import random
 rng=random.Random(20261024)
 for _ in range(20):
  dna=''.join(rng.choices('ACGT',k=300))
  for mol,k,flags in SETTINGS:
   filtered=sketch(dna,mol,k,flags)
   native=sourmash.MinHash(n=0,ksize=k,scaled=10,**flags);native.add_sequence(dna,force=True)
   if set(filtered.hashes)-set(native.hashes):raise ValueError('Stop-free translation gained hashes')
   if mol=='dna' and set(filtered.hashes)!=set(native.hashes):raise ValueError('DNA changed')
   if not set(filtered.downsample(scaled=100).hashes)<=set(filtered.hashes):raise ValueError('Scale nesting failed')
 print('80 independent native/subset and DNA identity comparisons passed',flush=True)

def main():
 preflight()
 subprocess.run([str(CUTADAPT),'--version'],check=True)
 ROOT.mkdir() # Never merge with historical inclusive output.
 audit=json.loads((ARCHIVE/'archive-progress.json').read_text())
 ref=min(audit['runs'],key=lambda r:sum(map(int,r['metadata']['fastq_bytes'].split(';'))))
 accession=ref['run'];row=ref['metadata']
 progress=json.loads((FULL/'summary-progress.json').read_text());b=progress['completed_batches'][0]
 source=FULL/'splash-trimmed-all'/f"batch-{b['batch']:03d}"/'compactors.tsv'
 if sha(source)!=b['output_sha256']:raise ValueError('SPLASH source changed')
 with source.open() as f:queries=[(f"splash:full:{b['batch']}:{i}:{r['anchor']}",r['compactor']) for i,r in enumerate(csv.DictReader(f,delimiter='\t')) if len(r['compactor'])>len(r['anchor'])]
 selection=json.loads((DE/'selection.json').read_text())
 if sha(Path(selection['source']))!=selection['source_sha256']:raise ValueError('DE source changed')
 with gzip.open(DE/'all-significant-contigs.tsv.gz','rt') as f:
  de=[(r['query_id'],r['contig']) for r in csv.DictReader(f,delimiter='\t') if float(r['pvalue'])<.05]
 if len(de)!=selection['selected_contigs']:raise ValueError('Incomplete DE export')
 queries+=de
 if len({q for q,_ in queries})!=len(queries):raise ValueError('Duplicate query IDs')
 with tempfile.TemporaryDirectory(dir=os.environ['L_SCRATCH_JOB']) as tmp:
  tmp=Path(tmp);trimmed=[]
  for i,(url,md5) in enumerate(zip(row['fastq_ftp'].split(';'),row['fastq_md5'].split(';'))):
   raw=tmp/f'reads-{i}.fastq.gz'
   run(['curl','--fail','--location','--retry','5','-o',raw,'https://'+url],ROOT/f'download-{i}.log')
   if sha(raw,'md5')!=md5:raise ValueError('Read MD5 mismatch')
   run(['trim_galore','--path_to_cutadapt',CUTADAPT,'--cores','1','--quality','20','--length','20','--gzip','--output_dir',tmp,raw],ROOT/f'trim-{i}.log')
   trimmed.append(tmp/f'reads-{i}_trimmed.fq.gz')
  settings=[]
  for mol,k,flags in SETTINGS:
   qpaths={s:ROOT/f'{mol}.query{s}.sig.zip' for s in (10,100)};empty={10:set(),100:set()}
   with SaveSignaturesToLocation(str(qpaths[10])) as a,SaveSignaturesToLocation(str(qpaths[100])) as bsave:
    for key,dna in queries:
     mh=sketch(dna,mol,k,flags)
     for scale,dest in [(10,a),(100,bsave)]:
      ds=mh.downsample(scaled=scale)
      if not len(ds):empty[scale].add(key)
      dest.add(sourmash.SourmashSignature(ds,name=key))
   reference=sourmash.MinHash(n=0,ksize=k,scaled=10,**flags)
   for path in trimmed:
    with gzip.open(path,'rt') as f:
     for record in SeqIO.parse(f,'fastq'):reference.merge(sketch(str(record.seq),mol,k,flags))
   pairs={};hashes={}
   for scale in (10,100):
    refpath=ROOT/f'{mol}.reference{scale}.sig.zip'
    with SaveSignaturesToLocation(str(refpath)) as save:save.add(sourmash.SourmashSignature(reference.downsample(scaled=scale),name=accession))
    paths=tmp/'paths.txt';paths.write_text(str(refpath)+'\n');index=tmp/f'{mol}-{scale}.rocksdb'
    run(['sourmash','scripts','index',paths,'-o',index,'-k',k,'-m','DNA' if mol=='dna' else mol,'--scaled',scale,'-c','4'],ROOT/f'{mol}-{scale}-index.log')
    csvpath=tmp/f'{mol}-{scale}.csv';log=ROOT/f'{mol}-{scale}-search.log'
    run(['sourmash','scripts','manysearch',qpaths[scale],index,'-o',csvpath,'-k',k,'-m','DNA' if mol=='dna' else mol,'-s',scale,'-t','0','-A','-N','--ignore-abundance','-c','4'],log)
    result=ROOT/f'{mol}-{scale}.matches.csv.gz'
    pairs[scale],count=save_matches(csvpath,result,accession,log,len(queries));hashes[scale]=sha(result)
   if pairs[100]-pairs[10]:raise ValueError('Scaled100 gained matches')
   loss=ROOT/f'{mol}.lost.tsv.gz'
   with gzip.open(loss,'wt') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['query_id','run','reason'])
    for q in sorted(pairs[10]-pairs[100]):w.writerow([q,accession,'empty query sketch' if q in empty[100] else 'no retained shared hash'])
   settings.append(dict(moltype=mol,queries=len(queries),empty10=len(empty[10]),empty100=len(empty[100]),matches10=len(pairs[10]),matches100=len(pairs[100]),lost=len(pairs[10]-pairs[100]),result_hashes=hashes,loss_sha256=sha(loss)))
 (ROOT/'validated.json').write_text(json.dumps(dict(reference=accession,reference_metadata=row,query_records=len(queries),dekupl_records=len(de),splash_batches=1,settings=settings,translation='Standard-code six frames; remove words crossing stop or X; no reference coding gate',scope='All DE-kupl queries plus first completed full-depth SPLASH batch, one complete newly trimmed public run; paired10/100 preflight only'),indent=2)+'\n')
if __name__=='__main__':main()
