"""Build a self-contained report from checked compact outputs, preserving missing stages."""
import csv
import datetime
import json
import re
import shutil
from pathlib import Path
from public_results_html import render as render_public_results

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report'
SITE=Path('/Users/olga/code/oyster-shazam-report-site/dist')


def main():
    OUT.mkdir(exist_ok=True)
    raw=json.loads((ROOT/'results/raw-prefix-qc.json').read_text())
    trimmed=ROOT/'results/trimmed-qc.json'
    qc=json.loads(trimmed.read_text()) if trimmed.exists() else []
    metadata=list(csv.DictReader((ROOT/'provenance/ena-runs-2026-10-23.tsv').open(),delimiter='\t'))
    samples=[]
    for row in raw:
        s=row['sample'];meta=next(m for m in metadata if m['sample_alias']==s)
        result=next((x for x in qc if x['sample']==s),None)
        sample=dict(sample=s,run=meta['run_accession'],condition='wild' if s.startswith('DW') else 'breed',
                    raw_pairs=row['pairs'],raw_bases=sum(row['bases']),raw_lengths=row['lengths'],
                    trimmed_pairs=result['after']['pairs'] if result else None,
                    trimmed_bases=sum(result['after']['bases']) if result else None,
                    trimmed_lengths=result['after']['lengths'] if result else None,
                    reads_at_least_62bp=result['after']['reads_at_least_62bp'] if result else None)
        adapters=[]
        for mate in [1,2]:
            text=(ROOT/f'results/trimming-reports/{s}_trimmed_{mate}.fastq.gz_trimming_report.txt').read_text()
            match=re.search(r'Reads with adapters:\s*([\d,]+)',text)
            adapters.append(int(match.group(1).replace(',','')))
        sample['reads_with_adapter_matches']=adapters
        samples.append(sample)
    inventory=json.loads((ROOT/'provenance/bivalve-rnaseq-ena-2026-10-23.summary.json').read_text())
    comparison=json.loads((ROOT/'results/splash-comparison.json').read_text())
    sketches=[json.loads(f.read_text()) for f in sorted((ROOT/'results/index-pilot').glob('*.validated.json'))]
    data=dict(splash=comparison,index_pilot=sketches,snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=samples,inventory=inventory,
              preprocessing=dict(pipeline='nf-core/rnaseq 3.27.0',job='49186529',status='completed',
                                 validated_pairs=bool(qc),elapsed_seconds=254,peak_rss_kib=11691088,
                                 trim_galore='2.3.0',fastqc='0.12.1',multiqc='1.35'),
              stages=[dict(name='Raw reads',status='Complete',detail='Six first-million paired prefixes. All 12 uncompressed hashes match the supplied prior pilot.'),
                      dict(name='Trimming',status='Complete',detail='nf-core/rnaseq with alignment and pseudoalignment disabled. All quality and adapter summary checks passed.'),
                      dict(name='SPLASH',status='Full-depth discovery complete',detail='Full-depth read1 SPLASH job 49214375 completed; all 5,538,699 adjusted P < 0.05 anchors are selected for extension in 554 batches, job 49221554. Separately, extension of all 213,658 significant prefix anchors completed: 165,679 anchors extended into 354,316 records; 47,979 without extensions. The sequence explorer remains the historical 300-anchor pilot.'),
                      dict(name='DE-kupl',status='Full-depth running',detail='Full-depth job 49203559 is running in the separate oyster-mvp workflow. No full-depth DE-kupl result is claimed yet.'),
                      dict(name='Sourmash',status='Public pilot indexed',detail='Eight public untrimmed runs have validated indexes. The older six-library prefix HP index timed out and its searches were cancelled; corrected public search array 49222143 is submitted.'),
                      dict(name='Bivalve archive',status='Corrected validation needs access',detail='One trimmed public run passed independent four-alphabet paired-scale and self-search checks. An inconsistency in the older untrimmed public pilot comparison stopped allocation 49215860 and cancelled its dependent worker array. Explicit threshold 0 fixed a reproduced filtering issue; corrected validation session disconnected and SSH access is unavailable. 181 runs lack ENA FASTQ URLs.')],
              findings=[])
    with (ROOT/'results/splash-trimmed/compactors.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            data['findings'].append(dict(id=f"splash:trimmed:{r['anchor']}:{r['id']}",pipeline='splash',anchor=r['anchor'],dna=r['compactor'],support=int(r['support']),evidence_summary='Compactors DNA extension. Gene identity and protein annotation are pending.'))
    if (ROOT/'results/findings.json').exists():
        data['findings']=json.loads((ROOT/'results/findings.json').read_text())
    (OUT/'report-data.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    template=(ROOT/'report-template.html').read_text()
    public_path=ROOT/'results/public-match-summary.json'
    public=json.loads(public_path.read_text()) if public_path.exists() else None
    template=template.replace('__PUBLIC_RESULTS__',render_public_results(public) if public else '<p>Public search aggregation pending.</p>')
    if public:
        shutil.copy2(public_path,OUT/public_path.name)
        loss=ROOT/'results'/public.get('loss_table','matches-lost-at-scaled100.tsv.gz')
        if loss.exists():shutil.copy2(loss,OUT/loss.name)
    (OUT/'index.html').write_text(template.replace('__REPORT_DATA__',json.dumps(data).replace('</','<\\/')))
    for file in ['raw-prefix-qc.json','prior-prefix-comparison.json','execution_trace.txt','trimmed-qc.json','splash-comparison.json','job-status.tsv']:
        src=ROOT/'results'/file
        if src.exists():shutil.copy2(src,OUT/file)
    inventory_path=ROOT/'provenance/bivalve-rnaseq-ena-2026-10-23.tsv.gz'
    shutil.copy2(inventory_path,OUT/inventory_path.name)
    if SITE.parent.exists():
        SITE.mkdir(exist_ok=True)
        for f in OUT.iterdir():
            if f.is_file():shutil.copy2(f,SITE/f.name)
    print('Built report from',len(samples),'libraries and',len(data['findings']),'validated finding records')


if __name__=='__main__':main()
