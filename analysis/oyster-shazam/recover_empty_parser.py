import os,json,csv
from pathlib import Path
root=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
code=Path.home()/'oyster-shazam-code';plan=json.loads((code/'archive-worker-plan.json').read_text());jobs=[];moved=[];skipped=[]
for line in (root/'failed-status.tsv').read_text().splitlines():
 parts=line.split('|')
 if len(parts)<2 or parts[1]!='FAILED' or not parts[0].startswith('49241661_'):continue
 identifier=parts[0].split('_')[1]
 if not identifier.isdigit():continue
 log=root/f'worker-49241661-{identifier}.log'
 if not log.exists() or "TypeError: 'NoneType' object is not iterable" not in log.read_text(errors='replace'):
  skipped.append(identifier);continue
 for accession in plan['workers'][identifier]['runs']:
  out=root/'runs'/accession[:6]/accession
  if out.exists() and not (out/'validated.json').exists():
   dest=root/'interrupted-empty-parser'/identifier/accession;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():raise ValueError('Backup already exists; inspect before reuse')
   out.rename(dest);moved.append(dict(worker=identifier,run=accession,backup=str(dest)))
 jobs.append(int(identifier))
result=dict(workers=sorted(jobs),preserved=moved,other_failures=skipped,rule='Retry only completed FAILED worker tasks whose log shows the diagnosed CSV fieldnames None parser error; never touch running tasks or validated run outputs')
(root/'parser-recovery-plan.json').write_text(json.dumps(result,indent=2)+'\n');print('Retry workers',len(jobs),'preserved partial runs',len(moved),'other failures',len(skipped))
