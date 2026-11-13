"""Measure a fixed read prefix for a diagnostic figure, without changing full-depth analysis."""
import csv,gzip,os,subprocess,json,hashlib
from pathlib import Path
root=Path(os.environ['SCRATCH'])/'oyster-mvp-20261023/trimmed/full'
out=Path(os.environ['SCRATCH'])/'oyster-stop-frames-20261024';out.mkdir(exist_ok=True)
rows=[];inputs=[]
for sample in ['DW1','DW2','DW3','ZW1','ZW2','ZW3']:
    for mate in [1,2]:
        source=root/f'{sample}_{mate}.fq.gz'
        # Bounded deterministic sampling is only for this descriptive diagnostic.
        path=Path(os.environ['L_SCRATCH_JOB'])/f'{sample}_{mate}.fastq'
        count=0
        with gzip.open(source,'rt') as f,path.open('w') as dest:
            for _ in range(100000):
                lines=[f.readline() for _ in range(4)]
                if not lines[0]:break
                if any(not line for line in lines):raise ValueError('Truncated input')
                dest.writelines(lines);count+=1
        result=out/f'{sample}_{mate}.tsv'
        with path.open() as src,result.open('w') as dest:subprocess.run([str(Path(os.environ['L_SCRATCH_JOB'])/'stop-frame-audit')],stdin=src,stdout=dest,check=True)
        with result.open() as f:
            for row in csv.DictReader(f,delimiter='\t'):rows.append(dict(sample=sample,mate=mate,**{k:int(v) for k,v in row.items()}))
        if sum(r['reads'] for r in rows if r['sample']==sample and r['mate']==mate)!=count:raise ValueError('Count mismatch')
        inputs.append(dict(sample=sample,mate=mate,source=str(source),reads=count,prefix_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        path.unlink()
(out/'summary.json').write_text(json.dumps(dict(rule='First100000trimmed reads of each mate of each full library; diagnostic only',genetic_code='Standard nuclear code: TAA TAG TGA; ambiguous codons not called stops',inputs=inputs,rows=rows),indent=2)+'\n')
print('Measured',sum(x['reads'] for x in inputs),'reads')
