#!/bin/bash
#SBATCH --job-name=oyster-full-extension-selection
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - <<'PY'
import os,pathlib,csv,json,hashlib,math,subprocess
r=pathlib.Path(os.environ['SCRATCH'])/'oyster-mvp-20261023'
s=r/'splash/full-memory-corrected'
if not (s/'complete').exists():raise ValueError('Full-depth SPLASH incomplete')
source=s/'oyster.after_correction.scores.tsv';anchors=[]
with source.open() as f:
 for row in csv.DictReader(f,delimiter='\t'):
  q=float(row['pval_opt_corrected'])
  if not math.isfinite(q) or not 0<=q<=1:raise ValueError('Invalid adjusted P')
  if q<.05:anchors.append(row['anchor'])
if len(anchors)!=len(set(anchors)):raise ValueError('Duplicate selected anchors')
with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
out=pathlib.Path(os.environ['SCRATCH'])/'oyster-full-extensions-20261023';out.mkdir(exist_ok=True)
if (out/'submission.json').exists():raise ValueError('Already submitted; inspect before retry')
reads=[]
for sample in ['DW1','DW2','DW3','ZW1','ZW2','ZW3']:
 meta=json.loads((r/f'trimmed/full/{sample}.validated.json').read_text())
 if meta['depth']!='full' or meta['prefix']!='full file':raise ValueError('Wrong input depth')
 for mate in [1,2]:
  p=r/f'trimmed/full/{sample}_val_{mate}.fq.gz'
  if not p.exists():raise ValueError('Missing full-depth mate')
  reads.append(str(p))
(out/'reads.txt').write_text('\n'.join(reads)+'\n')
with (out/'selected-anchors.tsv').open('w') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['anchor']);w.writerows([a] for a in sorted(anchors))
meta=dict(selected=len(anchors),threshold=.05,comparison='<',source=str(source),source_sha256=sha,discovery='Complete trimmed read1 from each of six libraries',extension_context='Both complete trimmed mates, twelve FASTQs',batches=math.ceil(len(anchors)/10000))
(out/'selection.json').write_text(json.dumps(meta,indent=2)+'\n')
if anchors:
 cmd=['sbatch','--parsable',f'--array=0-{meta["batches"]-1}%4',f'--output={out}/extension-%A-%a.log',str(pathlib.Path.home()/'oyster-shazam-code/extend-full-depth.sbatch')]
 job=subprocess.check_output(cmd,text=True).strip();meta['job']=job
(out/'submission.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
PY
