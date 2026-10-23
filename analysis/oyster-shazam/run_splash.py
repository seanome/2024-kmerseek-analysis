"""Run upstream SPLASH and Compactors with validated per-stage checkpoints."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def command(args, cwd, name):
    start=time.monotonic()
    with (cwd/f'{name}.log').open('w') as log:
        result=subprocess.run(['/usr/bin/time','-v',*map(str,args)],cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    info=dict(command=list(map(str,args)),exit_status=result.returncode,wall_seconds=time.monotonic()-start)
    (cwd/f'{name}.command.json').write_text(json.dumps(info,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'{name} failed with status {result.returncode}; inspect {name}.log')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['trimmed','untrimmed'],default='trimmed')
    args=p.parse_args();root=args.root
    out=root/f'splash-{args.arm}';out.mkdir(exist_ok=True)
    software=root/'software'
    samples=['DW1','DW2','DW3','ZW1','ZW2','ZW3']
    if args.arm=='trimmed':
        assert len(json.loads((root/'trimmed-qc.json').read_text()))==6
    lines=[];reads=[]
    for sample in samples:
        merged=out/f'{sample}.fastq.gz'
        if not merged.exists():
            partial=Path(str(merged)+'.partial')
            with gzip.open(partial,'wb',compresslevel=1) as dest:
                for mate in (1,2):
                    source=(root/'shared-trimmed'/f'{sample}_{mate}.fastq.gz' if args.arm=='trimmed'
                            else root/'raw'/sample/f'{sample}_{mate}.fastq.gz')
                    with gzip.open(source,'rb') as f:
                        shutil.copyfileobj(f,dest,1024*1024)
            partial.replace(merged)
        lines.append(f'{sample} {merged}\n');reads.append(str(merged)+'\n')
    (out/'input.txt').write_text(''.join(lines));(out/'reads.txt').write_text(''.join(reads))
    if not (out/'splash.validated.json').exists():
        # SPLASH does not resume a partly completed stage; preserve it for diagnosis.
        if (out/'splash.command.json').exists():
            raise RuntimeError('Previous SPLASH attempt exists without validation; diagnose before retry')
        command([software/'splash','--bin_path',software,'--outname_prefix','splash',
                 '--anchor_len','31','--target_len','31','--gap_len','0','--fdr_threshold','0.05',
                 '--without_compactors','--n_threads_stage_1','1','--n_threads_stage_1_internal','8',
                 '--n_threads_stage_2','8','--n_bins','8','--kmc_max_mem_GB','12',
                 '--tmp_dir',Path(os.environ['L_SCRATCH_JOB'])/f'splash-{args.arm}',
                 '--dump_sample_anchor_target_count_txt','--satc_merge_dump_format','satc',out/'input.txt'],out,'splash')
        significant=table(out/'splash.after_correction.scores.tsv')
        with (out/'splash.after_correction.all_anchors.tsv').open() as f:
            header=next(f);tested=sum(1 for _ in f)
        assert tested>=len(significant)
        for row in significant:
            q=float(row['pval_opt_corrected'])
            assert math.isfinite(q) and 0<=q<=0.05
        selected=sorted(significant,key=lambda r:(float(r['pval_opt_corrected']),r['anchor']))[:300]
        assert len({r['anchor'] for r in selected})==len(selected)
        with (out/'selected_anchors.tsv').open('w') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['anchor']);w.writerows([r['anchor']] for r in selected)
        counts=defaultdict(int)
        for row in significant:
            counts[row['pval_opt_corrected']]+=1
        stats=dict(tested_anchors=tested,significant_anchors=len(significant),selected_anchors=len(selected),
                   selected_anchors_in_q_ties=sum(counts[r['pval_opt_corrected']]>1 for r in selected),
                   correction='Benjamini-Yekutieli',threshold=0.05)
        (out/'splash.validated.json').write_text(json.dumps(stats,indent=2)+'\n')
    selected=table(out/'selected_anchors.tsv');anchors={r['anchor'] for r in selected}
    if not (out/'compactors.validated.json').exists():
        if (out/'compactors.command.json').exists():
            raise RuntimeError('Previous Compactors attempt exists without validation; diagnose before retry')
        if anchors:
            command([software/'compactors','--num_threads','8',out/'reads.txt',out/'selected_anchors.tsv',out/'compactors.tsv'],out,'compactors')
            rows=table(out/'compactors.tsv')
        else:
            (out/'compactors.tsv').write_text('anchor\tcompactor\n');rows=[]
        extended=set();unextended=set();lengths=[]
        for row in rows:
            a=row['anchor'];seq=row['compactor']
            assert a in anchors and not set(seq.upper())-set('ACGTN')
            assert a in seq
            lengths.append(len(seq))
            (extended if len(seq)>len(a) else unextended).add(a)
        stats=dict(output_rows=len(rows),anchors_extended=len(extended),
                   anchors_only_unextended=len(unextended-extended),anchors_without_output=len(anchors-{r['anchor'] for r in rows}),
                   dna_lengths=lengths)
        (out/'compactors.validated.json').write_text(json.dumps(stats,indent=2)+'\n')
    # Keep selected target counts with the original biological sample name.
    totals=defaultdict(int);records=[]
    dumps=sorted((out/'splash_dumps').glob('*.satc.dump'))
    if anchors and not dumps:
        raise ValueError('Missing SPLASH target-count dumps')
    for dump in dumps:
        with dump.open() as f:
            for line in f:
                sample,anchor,target,count=line.split()
                if anchor in anchors:
                    assert sample in samples
                    count=int(count);assert count>=0
                    records.append((anchor,sample,target,count));totals[anchor,sample]+=count
    with (out/'selected-target-counts.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['anchor','sample','condition','target','count','sample_anchor_total','target_proportion'])
        for anchor,sample,target,count in records:
            total=totals[anchor,sample]
            w.writerow([anchor,sample,'wild' if sample.startswith('DW') else 'breed',target,count,total,count/total])
    print('SPLASH and Compactors validated:',args.arm)


if __name__=='__main__':
    main()
