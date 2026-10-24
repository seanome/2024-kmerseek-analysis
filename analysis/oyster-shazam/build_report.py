"""Build a self-contained report from checked compact outputs, preserving missing stages."""
import csv
import datetime
import json
import os
import re
import shutil
from pathlib import Path
from public_results_html import render as render_public_results

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report'
SITE=Path(os.environ.get('OYSTER_SITE_DIR','/Users/olga/code/oyster-shazam-report-site/dist'))


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
              stages=[dict(name='Raw reads',status='Complete',detail='Six complete original experimental libraries:153,671,701 raw pairs;153,628,178 pairs retained after trimming. Historical prefix hashes remain available separately.'),
                      dict(name='Trimming',status='Complete',detail='nf-core/rnaseq with alignment and pseudoalignment disabled. All quality and adapter summary checks passed.'),
                      dict(name='SPLASH',status='Full-depth discovery complete',detail='Full-depth read1 SPLASH job 49214375 completed; all 5,538,699 adjusted P < 0.05 anchors are selected for extension in 554 batches, job 49221554. Separately, extension of all 213,658 significant prefix anchors completed: 165,679 anchors extended into 354,316 records; 47,979 without extensions. The sequence explorer remains the historical 300-anchor pilot.'),
                      dict(name='DE-kupl',status='Full-depth completed',detail='Full-depth job 49203559 completed with exit0 in 3:05:46. Biological interpretation and a reference-free comparison are pending.'),
                      dict(name='Sourmash',status='Public pilot indexed',detail='Eight public untrimmed runs have validated indexes. The older six-library prefix HP index timed out and its searches were cancelled; all22 corrected public search batches completed and passed the independent pair-set/hash audit49241571.'),
                      dict(name='Bivalve archive',status='561 trimmed public runs completed; partial archive',detail='One trimmed public run passed independent four-alphabet paired-scale and self-search checks. The explicit threshold0 correction passed the complete public-pilot audit. Restored-access validation49241803 passed all four encodings and both scales with exact loss-set and self-search checks. A hash audit found561 completed trimmed runs from103studies. The empty-result parser is repaired; failed tasks are recovered separately. 181 runs lack ENA FASTQ URLs.')],
              findings=[])
    with (ROOT/'results/splash-trimmed/compactors.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            data['findings'].append(dict(id=f"splash:trimmed:{r['anchor']}:{r['id']}",pipeline='splash',anchor=r['anchor'],dna=r['compactor'],support=int(r['support']),evidence_summary='Compactors DNA extension. Gene identity and protein annotation are pending.'))
    if (ROOT/'results/findings.json').exists():
        data['findings']=json.loads((ROOT/'results/findings.json').read_text())
    data['full_depth']=json.loads((ROOT/'full-depth-status.json').read_text())
    full_progress=ROOT/'results/full-depth-extension-progress.json'
    data['full_extension_progress']=json.loads(full_progress.read_text()) if full_progress.exists() else None
    (OUT/'report-data.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    template=(ROOT/'report-template.html').read_text()
    full_path=ROOT/'results/full-depth-extension-progress.json'
    if full_path.exists():
        full=json.loads(full_path.read_text())
        full_html=f"<div class=\"card\"><h3>Full-depth extension results</h3><p>{full['selected_total']:,} anchors selected at adjusted P &lt; 0.05. {len(full['completed_batches']):,} of {full['total_batches']:,} extension batches have completion records.</p><p>Those batches cover {full['completed_selected_anchors']:,} selected anchors: {full['anchors_extended']:,} have extended sequence, producing {full['extension_records']:,} DNA records. Remaining batches are pending or running, not zero results. Records are not distinct genes.</p><p>Discovery used complete read 1 from each experimental library; extensions use both complete trimmed mates. All 554 full-depth search batches are now submitted against eight public runs. Searches for independently audited completed extension batches are eligible; later batches wait for their corresponding extension.</p><p>{full['verification']}</p><a href=\"full-depth-extension-progress.json\" download>Full-depth extension progress · JSON</a></div>"
        template=template.replace('__FULL_DEPTH_RESULTS__',full_html)
        shutil.copy2(full_path,OUT/full_path.name)
    else:template=template.replace('__FULL_DEPTH_RESULTS__','<p>Full-depth extension aggregation pending.</p>')
    public_path=ROOT/'results/public-match-summary.json'
    public=json.loads(public_path.read_text()) if public_path.exists() else None
    template=template.replace('__PUBLIC_RESULTS__',render_public_results(public) if public else '<p>Public search aggregation pending.</p>')
    full_public_path=ROOT/'results/full-public-match-summary.json'
    full_public=json.loads(full_public_path.read_text()) if full_public_path.exists() else None
    template=template.replace('__FULL_PUBLIC_RESULTS__',render_public_results(full_public,full_depth=True) if full_public else '<p>Full-depth public-match auditing is in progress.</p>')
    archive_path=ROOT/'results/public-archive-progress.json'
    if archive_path.exists():
        archive=json.loads(archive_path.read_text())
        table=''.join(f"<tr><td>{mol.upper()}</td><td>{v['matches_10']:,}</td><td>{v['matches_100']:,}</td><td>{v['lost_pairs']:,}</td></tr>" for mol,v in archive['encodings'].items())
        archive_html=f"<div class=\"card\"><h3>Trimmed public archive: {archive['validated_runs']:,} completed runs</h3><p>{archive['validated_runs']:,} of 17,623 inventory runs, from {archive['validated_studies']:,} studies, have completed four-alphabet sketching and searches at scaled 10 and 100 against all 354,316 prefix-derived extension records. These are separate from the full-depth queries above.</p><p>{archive['verification']}. Counts below sum completion records; a second complete pair-set audit of this growing archive is pending.</p><table><thead><tr><th>Encoding</th><th>Query/run pairs at 10</th><th>At 100</th><th>Lost at 100</th></tr></thead><tbody>{table}</tbody></table><p>These candidate occurrences do not establish genes or functions. Partial and failed runs are excluded. The zero-result parser was repaired and tested on a real run; other download/application failures remain under investigation.</p><a href=\"public-archive-progress.json\" download>Indexed runs, study/species metadata and completion counts · JSON</a></div>"
        template=template.replace('__ARCHIVE_PROGRESS__',archive_html)
        shutil.copy2(archive_path,OUT/archive_path.name)
    else:template=template.replace('__ARCHIVE_PROGRESS__','')

    if full_public:
        shutil.copy2(full_public_path,OUT/full_public_path.name)
        for entry in full_public.get('loss_tables',{}).values():shutil.copy2(ROOT/'results'/entry['file'],OUT/entry['file'])

    if public:
        shutil.copy2(public_path,OUT/public_path.name)
        loss=ROOT/'results'/public.get('loss_table','matches-lost-at-scaled100.tsv.gz')
        if public.get('loss_tables'):
            for entry in public['loss_tables'].values():shutil.copy2(ROOT/'results'/entry['file'],OUT/entry['file'])
        elif loss.exists():shutil.copy2(loss,OUT/loss.name)
    (OUT/'index.html').write_text(template.replace('__REPORT_DATA__',json.dumps(data).replace('</','<\\/')))
    for file in ['raw-prefix-qc.json','prior-prefix-comparison.json','execution_trace.txt','trimmed-qc.json','splash-comparison.json','job-status.tsv']:
        src=ROOT/'results'/file
        if src.exists():shutil.copy2(src,OUT/file)
    for asset in ['public_bivalve_raw_vs_sourmash_sizes.png','public-sketch-size-comparison.json','public-sketch-size-comparison.tsv','metro/workflow-overview.svg','metro/workflow-overview.mmd','oyster_trimmed_reads_stop_frames.png','results/stop-frame-summary.json','results/stop-frame-plot-summary.json']:
        if (ROOT/asset).exists():shutil.copy2(ROOT/asset,OUT/Path(asset).name)
    inventory_path=ROOT/'provenance/bivalve-rnaseq-ena-2026-10-23.tsv.gz'
    shutil.copy2(inventory_path,OUT/inventory_path.name)
    if SITE.parent.exists():
        SITE.mkdir(exist_ok=True)
        for f in OUT.iterdir():
            if f.is_file():shutil.copy2(f,SITE/f.name)
    print('Built report from',len(samples),'libraries and',len(data['findings']),'validated finding records')


if __name__=='__main__':main()
