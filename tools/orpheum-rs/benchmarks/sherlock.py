"""Bounded real-read benchmark; compare native six-frame and stop-free sketch contents."""
import csv,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
import sourmash
out=Path(os.environ['SCRATCH'])/'orpheum-rs-benchmark-20261024';out.mkdir(exist_ok=True)
local=Path(os.environ['L_SCRATCH_JOB']);binary=local/'target/release/orpheum-rs';root=Path(os.environ['SCRATCH'])/'oyster-mvp-20261023';records=1000000
source=root/'trimmed/full/DW1_1.fq.gz';reads=local/'DW1_R1_first1000000.fastq'
with gzip.open(source,'rt') as src,reads.open('w') as dest:
 for _ in range(records):
  lines=[src.readline() for _ in range(4)]
  if any(not x for x in lines):raise ValueError('Input shorter than frozen benchmark prefix')
  dest.writelines(lines)
measurements=[]
def timed(name,args):
 start=time.monotonic()
 with (out/(name+'.out')).open('w') as stdout,(out/(name+'.log')).open('w') as stderr:subprocess.run(list(map(str,args)),stdout=stdout,stderr=stderr,check=True)
 measurements.append(dict(name=name,seconds=time.monotonic()-start))
index=out/'swissprot.protein7.idx'
timed('reference_index',[binary,'index',root/'references/swissprot.gz','-o',index,'--memory-mib','128'])
for threads in [1,4]:timed(f'translate_{threads}threads',[binary,'translate','-i',index,reads,'--threads',threads,'--peptides',local/f'coding-{threads}.fa'])
if hashlib.sha256((local/'coding-1.fa').read_bytes()).digest()!=hashlib.sha256((local/'coding-4.fa').read_bytes()).digest():raise ValueError('Thread count changed results')
params=['-p','protein,k=7,scaled=10','-p','dayhoff,k=12,scaled=10','-p','hp,k=30,scaled=10']
timed('native_stop_inclusive_sketch',['sourmash','sketch','translate',reads,*params,'--merge','benchmark','-o',out/'stop-inclusive.sig.zip'])
start=time.monotonic()
with (out/'stopfree-sketch.log').open('w') as log:
 translate=subprocess.Popen([str(binary),'sixframe',str(reads),'--min-aa','7'],stdout=subprocess.PIPE,stderr=log)
 try:subprocess.run(['sourmash','sketch','protein','-',*params,'--merge','benchmark','-o',str(out/'stop-free.sig.zip')],stdin=translate.stdout,stdout=log,stderr=log,check=True)
 finally:translate.stdout.close()
 if translate.wait()!=0:raise ValueError('Stop-free translation failed')
measurements.append(dict(name='rust_stop_free_translation_and_sketch',seconds=time.monotonic()-start))
a={s.minhash.moltype.lower():s.minhash for s in sourmash.load_file_as_signatures(str(out/'stop-inclusive.sig.zip'))};b={s.minhash.moltype.lower():s.minhash for s in sourmash.load_file_as_signatures(str(out/'stop-free.sig.zip'))}
comparisons={}
for mol in ['protein','dayhoff','hp']:
 old=set(a[mol].hashes);new=set(b[mol].hashes)
 if new-old:raise ValueError('Stop-free sketch gained hashes; inspect translation/hash compatibility')
 comparisons[mol]=dict(stop_inclusive_hashes=len(old),stop_free_hashes=len(new),removed_hashes=len(old-new))
result=dict(records=records,source=str(source),sampling='Fixed first1000000trimmed read1 records; diagnostic benchmark only',input_sha256=hashlib.sha256(reads.read_bytes()).hexdigest(),measurements=measurements,sketch_comparison=comparisons,total_stop_inclusive_zip_bytes=(out/'stop-inclusive.sig.zip').stat().st_size,total_stop_free_zip_bytes=(out/'stop-free.sig.zip').stat().st_size,upstream_speed_comparison='Not measured: timing baseline is native sourmash, not Python Orpheum',frame_truth='No correct-frame labels; classification yield is not accuracy')
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
