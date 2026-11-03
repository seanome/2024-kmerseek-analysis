"""Reproduce a scaled100-only match using identical query sketches and explicit thresholds."""
import os,pathlib,csv,json,subprocess
import sourmash
from sourmash.save_load import SaveSignaturesToLocation
r=pathlib.Path(os.environ['SCRATCH'])/'oyster-shazam-20261023'
o=r/'threshold-diagnosis';o.mkdir(exist_ok=True)
name='splash:all:0:13858:AAAATTTTATAAGATAATTACATCGAGAATG'
query=o/'query.sig.zip'
with SaveSignaturesToLocation(str(query)) as save:
 for sig in sourmash.load_file_as_signatures(str(r/'search-public-bivalve/batch-000/hp.sig.zip')):
  if sig.name==name:save.add(sig);break
result={}
for scale in [10,100]:
 for mode,flags in [('default',[]),('zero',['-t','0'])]:
  dest=o/f'{scale}-{mode}.csv'
  subprocess.run(['sourmash','scripts','manysearch',str(query),str(r/'public-bivalve-indexes/hp.scaled5.rocksdb'),'-o',str(dest),'-k','30','-m','hp','-s',str(scale),'-c','4','-A','-N','--ignore-abundance',*flags],check=True)
  with dest.open() as f:rows=list(csv.DictReader(f))
  result[f'{scale}-{mode}']=[x for x in rows if x['match_name']=='SRR942484']
print('THRESHOLD_REPRODUCTION',json.dumps(result),flush=True)
(o/'result.json').write_text(json.dumps(result,indent=2)+'\n')
