"""Real sourmash/Branchwater integration test; run inside a Slurm job."""
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import sourmash
from sourmash.save_load import SaveSignaturesToLocation
from search_extended import SETTINGS


def main():
    with tempfile.TemporaryDirectory(dir=os.environ['L_SCRATCH_JOB']) as d:
        root=Path(d); source=root/'splash-trimmed-all/batch-000';source.mkdir(parents=True)
        indexes=root/'pilot-indexes';indexes.mkdir()
        dna='ATGGCTGATGAATTTGGTCATATTAAACTGATGAATCCTCAACGTTCTACTGTTTGGTAT'*8
        with (source/'compactors.tsv').open('w') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['anchor','id','compactor'])
            w.writerow([dna[:31],'positive',dna]);w.writerow(['N'*31,'ambiguous','N'*100])
        (source/'validated.json').write_text(json.dumps(dict(output_sha256=hashlib.file_digest((source/'compactors.tsv').open('rb'),'sha256').hexdigest())))
        for mol,k,flags in SETTINGS:
            sigfile=indexes/f'{mol}.refs.sig.zip'
            with SaveSignaturesToLocation(str(sigfile)) as out:
                for name in ['DW1','DW2','DW3','ZW1','ZW2','ZW3']:
                    rng=random.Random(name)
                    tail=''.join(rng.choice('ACGT') for _ in range(300))
                    mh=sourmash.MinHash(n=0,ksize=k,scaled=5,**flags);mh.add_sequence(dna+tail,force=True)
                    assert len(mh)>0
                    out.add(sourmash.SourmashSignature(mh,name=name))
            listing=indexes/f'{mol}.paths';listing.write_text(str(sigfile)+'\n')
            subprocess.run(['sourmash','scripts','index',str(listing),'-o',str(indexes/f'{mol}.scaled5.rocksdb'),'-m','DNA' if mol=='dna' else mol,'-k',str(k),'--scaled','5','-c','4'],check=True)
            (indexes/f'{mol}.validated.json').write_text(json.dumps(dict(reference_names=['DW1','DW2','DW3','ZW1','ZW2','ZW3'])))
        subprocess.run([sys.executable,str(Path(__file__).with_name('search_extended.py')),'--root',str(root),'--batch','0'],check=True)
        result=json.loads((root/'search-trimmed-all/batch-000/validated.json').read_text())
        assert result['extended_records']==2 and len(result['searches'])==12
        for s in result['searches']:
            assert s['empty_sketches']>=1
            if s['scaled']==5:
                assert s['queries_with_overlap']==1 and s['match_rows']==6,s
        print('Real four-alphabet index/search test passed; ambiguous sequence retained as empty')


if __name__=='__main__':main()
