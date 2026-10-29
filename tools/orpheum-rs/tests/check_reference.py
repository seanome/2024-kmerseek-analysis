"""Independent exact-set checks of clean-frame scoring, stop policies and CLI I/O."""
import csv,gzip,random,subprocess,sys,tempfile
from pathlib import Path
binary=Path(sys.argv[1]).resolve();rng=random.Random(41)
aa='ACDEFGHIKLMNPQRSTVWY';bases='TCAG';table=dict(zip((a+b+c for a in bases for b in bases for c in bases),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
inverse={a:next(c for c,x in table.items() if x==a) for a in aa}
proteins=[''.join(rng.choices(aa,k=120)) for _ in range(40)]
reads=[''.join(inverse[a] for a in p[:40]) for p in proteins[:8]]+[''.join(rng.choices('ACGT',k=120)) for _ in range(60)]
reads+=['ATGTAA'+''.join(inverse[a] for a in proteins[0][:35])]
classes={'protein':{a:a for a in aa},'dayhoff':{a:str(i) for i,g in enumerate(['C','AGPST','DENQ','HKR','ILMV','FWY']) for a in g},'hp':{a:str(i) for i,g in enumerate(['AFGILMPVWY','CDEHKNQRST']) for a in g}}
def frames(dna):
 rev=dna.translate(str.maketrans('ACGT','TGCA'))[::-1]
 return [''.join(table[s[j:j+3]] for j in range(i,len(s)-2,3)) for s in [dna,rev] for i in range(3)]
def words(p,k,m):return [tuple(m[a] for a in p[i:i+k]) for i in range(len(p)-k+1) if set(p[i:i+k])<=set(aa)]
def run(*args,ok=True):
 r=subprocess.run([str(binary),*map(str,args)],capture_output=True,text=True)
 if (r.returncode==0)!=ok:raise AssertionError(r.stderr)
 return r
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);ref=root/'ref.fa';ref.write_text(''.join(f'>p{i}\n{p}\n' for i,p in enumerate(proteins)))
 source=root/'reads.fa';source.write_text(''.join(f'>r{i}\n{r}\n' for i,r in enumerate(reads)))
 checks=0
 for alphabet,k in [('protein',7),('dayhoff',12),('hp',30)]:
  m=classes[alphabet];target={w for p in proteins for w in words(p,k,m)};index=root/(alphabet+'.idx')
  run('index',ref,'-o',index,'--alphabet',alphabet,'-k',k,'--memory-mib',1)
  original=index.read_bytes();run('index',ref,'-o',index,ok=False);assert index.read_bytes()==original
  for policy in ['words','frame']:
   scores=root/f'{alphabet}-{policy}.tsv';peptides=root/f'{alphabet}-{policy}.fa'
   run('translate','-i',index,source,'--stop-policy',policy,'--scores',scores,'--peptides',peptides,'--threads',1)
   rows=list(csv.DictReader(scores.open(),delimiter='\t'));assert len(rows)==len(reads)*6
   for row in rows:
    i=int(row['read_id'][1:]);f=int(row['frame']);frame=(f-1) if f>0 else 2-f;p=frames(reads[i])[frame]
    if policy=='frame' and '*' in p:assert row['status']=='frame_has_stop';continue
    ws=words(p,k,m);unique=set(ws)
    if not unique:assert row['status']=='no_valid_words';continue
    if len(unique)*2<=len(ws):assert row['status']=='low_complexity_peptide';continue
    hits=len(unique&target);expected=hits/len(unique)
    assert int(row['unique_words'])==len(unique) and int(row['reference_hits'])==hits,(alphabet,policy,row)
    assert abs(float(row['membership_fraction'])-expected)<1e-8
    assert row['status']==('reference_supported' if hits>0 and expected>=(0.8 if alphabet=='hp' else 0.5) else 'unclassified'),(alphabet,policy,row)
    checks+=1
   threaded=root/f'{alphabet}-{policy}-threaded.fa'
   run('translate','-i',index,source,'--stop-policy',policy,'--peptides',threaded,'--threads',4)
   assert threaded.read_bytes()==peptides.read_bytes()
  broken=root/f'{alphabet}-bad.idx';blob=bytearray(original);blob[-1]^=1;broken.write_bytes(blob);run('translate','-i',broken,source,ok=False)
 fastq='@a\nATGGCTGATGAA\n+\nIIIIIIIIIIII\n';zipped=root/'reads.gz';zipped.write_bytes(gzip.compress(fastq.encode())+gzip.compress(fastq.replace('@a','@b').encode()))
 output=run('sixframe',zipped,'--min-aa',2).stdout;assert '>a|' in output and '>b|' in output and '*' not in output
 malformed=root/'bad.fastq';malformed.write_text('@bad\nACGT\n+\nIII\n');run('sixframe',malformed,ok=False)
 print(f'{checks} independent exact-set frame scores agree across protein/Dayhoff/HP and both stop policies; gzip, corruption, no-overwrite and thread determinism passed')
