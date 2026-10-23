"""Optional native integration check: SPLASH_BIN=/path/to/binaries pytest -q."""
import csv
import os
import random
from pathlib import Path
import pytest
from clam_context.cli import run_splash
from clam_context.core import read_tsv, translate_segments, build_index, analyze

@pytest.mark.skipif(not os.environ.get('SPLASH_BIN'),reason='Set SPLASH_BIN to test external SPLASH/Compactors')
def test_real_splash_compactors_handoff(tmp_path):
    rng=random.Random(19)
    common=''.join(rng.choices('ACGT',k=130))
    tails=[''.join(rng.choices('ACGT',k=250)) for _ in range(2)]
    with (tmp_path/'samples.tsv').open('w') as sheet:
        writer=csv.writer(sheet,delimiter='\t');writer.writerow(['sample','read1','condition'])
        for group in range(2):
            for rep in range(3):
                name=f'g{group}r{rep}';path=tmp_path/(name+'.fastq')
                with path.open('w') as f:
                    for i in range(1500):
                        variant=group if rng.random()<.97 else 1-group
                        seq=(common+tails[variant])[rng.randrange(0,60):][:250]
                        f.write(f'@{name}_{i}\n{seq}\n+\n'+ 'I'*len(seq)+'\n')
                writer.writerow([name,str(path),f'group{group}'])
    outputs=run_splash(tmp_path/'samples.tsv',os.environ['SPLASH_BIN'],tmp_path/'native',top=10)
    assert read_tsv(outputs['scores'])
    assert read_tsv(outputs['counts'])
    rows=read_tsv(outputs['compactors']);assert rows
    # A derived reference is a positive software control, not independent biological validation.
    peptides=[p['peptide'] for row in rows for p in translate_segments(row['compactor'],min_aa=40)]
    assert peptides
    fasta=tmp_path/'control.faa';fasta.write_text(''.join(f'>positive_{i}\n{p}\n' for i,p in enumerate(peptides[:10])))
    build_index(fasta,tmp_path/'db.sqlite')
    r=analyze(outputs['scores'],outputs['compactors'],tmp_path/'db.sqlite',tmp_path/'report',counts=outputs['counts'],top=10)
    assert len(r['events'])==10
    assert any(e['hits'] for e in r['events'])
    assert all(e['counts'] for e in r['events'])
