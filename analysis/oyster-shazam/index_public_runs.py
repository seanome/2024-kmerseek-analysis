"""Retrieve complete public ENA runs, verify MD5, then sketch four alphabets."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    manifest=Path(sys.argv[1]);root=Path(sys.argv[2]);code=Path(__file__).parent
    with manifest.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    for row in rows:
        run=row['run_accession'];out=root/'public-bivalve-sketches'
        done=out/f'{run}.validated.json'
        if done.exists():
            raise RuntimeError('Existing run: verify checkpoint before scheduling again')
        urls=row['fastq_ftp'].split(';');checks=row['fastq_md5'].split(';')
        if not urls or len(urls)!=len(checks):raise ValueError('Missing source checksums')
        with tempfile.TemporaryDirectory(prefix=run+'-',dir=os.environ['L_SCRATCH_JOB']) as tmp:
            paths=[]
            expected=sum(int(x) for x in row['fastq_bytes'].split(';'))
            if __import__('shutil').disk_usage(tmp).free < expected*2:
                raise RuntimeError('Insufficient node-local space')
            for i,(url,md5) in enumerate(zip(urls,checks)):
                if not url.startswith('ftp.sra.ebi.ac.uk/'):raise ValueError('Unexpected ENA host')
                dest=Path(tmp)/f'reads-{i}.fastq.gz'
                subprocess.run(['curl','--fail','--location','--retry','5','--retry-delay','10',
                                '--speed-limit','1024','--speed-time','60','--continue-at','-',
                                '--output',str(dest),'https://'+url],check=True)
                with dest.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
                if actual!=md5:raise ValueError('ENA MD5 mismatch')
                paths.append(dest)
            subprocess.run([sys.executable,str(code/'index_reads.py'),'--run',run,
                            '--reads',*map(str,paths),'--output',str(out),
                            '--input-scope','Complete public ENA RNA-seq FASTQs; all listed files, no trimming or subsampling'],check=True)
            (out/f'{run}.metadata.json').write_text(json.dumps(row,indent=2)+'\n')
            print('Complete public run sketched:',run,flush=True)


if __name__=='__main__':main()
