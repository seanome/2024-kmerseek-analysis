"""Make per-run DNA and six-frame translated sketches with the upstream sourmash CLI."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import sourmash

SETTINGS={'dna':31,'protein':7,'dayhoff':12,'hp':30}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--reads',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-scope',required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    checkpoint=a.output/f'{a.run}.validated.json'
    source=[dict(path=str(r.resolve()),sha256=hashlib.file_digest(r.open('rb'),'sha256').hexdigest()) for r in a.reads]
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if old['sources']!=source or old['input_scope']!=a.input_scope:
            raise ValueError('Completed sketch has different inputs')
        print('Already validated',a.run);return
    commands=[];results=[]
    for moltype,k in SETTINGS.items():
        dest=a.output/moltype;dest.mkdir(exist_ok=True)
        base=dest/f'{a.run}.scaled5.sig.zip'
        cmd=[sys.executable,'-m','sourmash','sketch','dna' if moltype=='dna' else 'translate',
             *map(str,a.reads),'--merge',a.run,'-p',f'{moltype},k={k},scaled=5','-o',str(base)]
        start=time.monotonic()
        subprocess.run(cmd,check=True);commands.append(cmd)
        for scale in [10,100]:
            target=dest/f'{a.run}.scaled{scale}.sig.zip'
            cmd=[sys.executable,'-m','sourmash','sig','downsample','--scaled',str(scale),str(base),'-o',str(target)]
            subprocess.run(cmd,check=True);commands.append(cmd)
        for scale in [5,10,100]:
            target=dest/f'{a.run}.scaled{scale}.sig.zip'
            sigs=list(sourmash.load_file_as_signatures(str(target)))
            assert len(sigs)==1
            mh=sigs[0].minhash
            assert mh.moltype.lower()==moltype and mh.ksize==k and mh.scaled==scale
            results.append(dict(moltype=moltype,ksize=k,scaled=scale,hashes=len(mh),bytes=target.stat().st_size,
                                signature_md5=sigs[0].md5sum(),path=str(target)))
        print(a.run,moltype,'seconds',time.monotonic()-start,flush=True)
    checkpoint.write_text(json.dumps(dict(run=a.run,input_scope=a.input_scope,sources=source,
       settings=results,commands=commands,translation='sourmash sketch translate; all six frames, no CDS prediction'),indent=2)+'\n')


if __name__=='__main__':
    main()
