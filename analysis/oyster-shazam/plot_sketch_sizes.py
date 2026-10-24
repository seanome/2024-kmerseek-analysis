"""Measured compressed FASTQ versus sourmash ZIP sizes for complete public pilots."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'notebooks'))
import pubfig as pf
from matplotlib.patches import Patch
from PIL import Image

ROOT=Path(__file__).resolve().parent
source=ROOT/'results/public-pilot'
rows=[]
for path in sorted(source.glob('*.validated.json')):
    meta=path.with_name(path.name.replace('.validated.json','.metadata.json'))
    if not meta.exists():continue
    data=json.loads(path.read_text());metadata=json.loads(meta.read_text())
    settings={(s['moltype'],s['scaled']):s['bytes'] for s in data['settings']}
    if len(settings)!=12:raise ValueError('Incomplete sketch settings')
    rows.append(dict(run=data['run'],raw_bytes=sum(map(int,metadata['fastq_bytes'].split(';'))),settings=settings))
if not rows:raise ValueError('No complete public pilots')
raw=sum(r['raw_bytes'] for r in rows)
colours=['#999999','#0072B2','#D55E00','#009E73']
labels=['Raw FASTQ.gz','Scaled 5','Scaled 10','Scaled 100']
encodings=[('dna',31,'DNA'),('protein',7,'Protein'),('dayhoff',12,'Dayhoff'),('hp',30,'HP')]
values={mol:[raw]+[sum(r['settings'][mol,scale] for r in rows) for scale in (5,10,100)] for mol,_,_ in encodings}
max_mb=max(v for numbers in values.values() for v in numbers)/1e6
pf.use_style();fig,axs=pf.figure(pf.TWO_COLUMN_MM,115,nrows=2,ncols=2,sharex=True)
fig.legend([Patch(facecolor=c) for c in colours],labels,ncol=4,loc='outside upper center',frameon=False,title=f'{len(rows)} complete public runs · before trimming · compressed file sizes')
for ax,(mol,k,name) in zip(axs.flat,encodings):
    mb=[v/1e6 for v in values[mol]]
    ax.barh(range(4),mb,color=colours)
    ax.set_yticks(range(4),labels)
    ax.invert_yaxis();ax.set_xlim(0,max_mb*1.24)
    ax.set_xlabel('Stored file size (MB; 1 MB = 1,000,000 bytes)')
    ax.set_title(f'{name} k{k}')
    for y,v in enumerate(mb):ax.text(v+max_mb*.015,y,f'{v:,.1f}',va='center')
# The comparison is the sum across the same completed full runs in every panel.
# FASTQ files are gzip-compressed; sketches are actual .sig.zip archive byte sizes.
pf.save(fig,ROOT/'public_bivalve_raw_vs_sourmash_sizes')
image=ROOT/'public_bivalve_raw_vs_sourmash_sizes.png'
with Image.open(image) as img:img.convert('RGB').quantize(colors=256).save(image)
with (ROOT/'public-sketch-size-comparison.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['encoding','k','setting','bytes','MB','runs'])
    for mol,k,_ in encodings:
        for label,size in zip(labels,values[mol]):w.writerow([mol,k,label,size,size/1e6,len(rows)])
summary=dict(runs=[r['run'] for r in rows],raw_bytes=raw,encodings=values,
             total_sketch_bytes={str(scale):sum(sum(r['settings'][mol,scale] for r in rows) for mol,_,_ in encodings) for scale in (5,10,100)},
             scope='Actual complete public-run pilot before trimming; file sizes exclude RocksDB indexes, metadata and search results.')
(ROOT/'public-sketch-size-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
