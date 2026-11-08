"""Count distinct transcriptomic SRA sample and BioSample accessions.

Reuses the frozen 2026-11-06 manifest and assay annotations from sra_census.py.
Downloads only accession/sample columns; does not access controlled sequence data.
"""
import concurrent.futures as cf
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sra_census as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'sample_evidence'
OUT.mkdir(exist_ok=True)
base.CACHE = ROOT/'sra_sample_cache'
base.CACHE.mkdir(exist_ok=True)
base.COLS = ['acc','sample_acc','biosample']
SOURCES = ['TRANSCRIPTOMIC','TRANSCRIPTOMIC SINGLE CELL','METATRANSCRIPTOMIC']

def run():
    cached=OUT/'transcriptomic_sample_metadata.parquet'
    if cached.exists():
        return count(pq.read_table(cached))
    manifest = json.loads((ROOT/'sra_evidence/source_manifest.json').read_text())
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        paths = list(pool.map(lambda pair: base.process(*pair), enumerate(manifest)))
    selected=[]
    for path in paths:
        ids = pq.read_table(path)
        original = pq.read_table(ROOT/'sra_cache'/Path(path).name,
                                 columns=['acc','assay_type','consent','librarysource'])
        assert pc.all(pc.equal(ids['acc'],original['acc'])).as_py()
        t=original.append_column('sample_acc',ids['sample_acc']).append_column('biosample',ids['biosample'])
        keep=pc.or_(pc.fill_null(pc.is_in(t['assay_type'],value_set=pa.array(['RNA-Seq','EST'])),False),
                    pc.fill_null(pc.is_in(t['librarysource'],value_set=pa.array(SOURCES)),False))
        selected.append(t.filter(keep))
    t=pa.concat_tables(selected)
    pq.write_table(t,OUT/'transcriptomic_sample_metadata.parquet')
    return count(t)

def count(t):
    results={}
    cohorts={'strict_RNA_seq':pc.equal(t['assay_type'],'RNA-Seq'),
             'broad_transcriptomic':pc.or_(pc.equal(t['assay_type'],'RNA-Seq'),pc.is_in(t['librarysource'],value_set=pa.array(SOURCES))),
             'SRA_EST_assay':pc.equal(t['assay_type'],'EST')}
    for cohort,mask in cohorts.items():
        ct=t.filter(pc.fill_null(mask,False))
        results[cohort]={}
        for access in ['public','controlled','all']:
            at=ct if access=='all' else ct.filter(pc.equal(ct['consent'],'public') if access=='public' else pc.not_equal(ct['consent'],'public'))
            r={'runs':len(at)}
            for key in ['sample_acc','biosample']:
                a=at[key]
                valid=pc.and_(pc.is_valid(a),pc.not_equal(a,''))
                r[key+'_distinct']=pc.count_distinct(pc.filter(a,pc.fill_null(valid,False))).as_py()
                r[key+'_missing_runs']=len(at)-(pc.sum(pc.cast(pc.fill_null(valid,False),pa.int64())).as_py() or 0)
            results[cohort][access]=r
            print(cohort,access,json.dumps(r),flush=True)
        results[cohort]['public_controlled_overlap']={key:results[cohort]['public'][key+'_distinct']+results[cohort]['controlled'][key+'_distinct']-results[cohort]['all'][key+'_distinct'] for key in ['sample_acc','biosample']}
    (OUT/'sra_sample_counts.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': run()
