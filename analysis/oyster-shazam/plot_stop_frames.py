"""Show measured stop-frame ambiguity without assuming the correct coding frame."""
from pathlib import Path
import sys,json,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'notebooks'))
import pubfig as pf
from matplotlib.patches import Patch
from PIL import Image
root=Path(__file__).resolve().parent;data=json.loads((root/'results/stop-frame-summary.json').read_text())
counts=collections.Counter();lengths=collections.defaultdict(collections.Counter)
for row in data['rows']:
    n=row['reads']-row['ambiguous_reads'];counts[row['frames_with_stop']]+=n
    length=row['read_length'];group='20–49' if length<50 else '50–99' if length<100 else '100–149' if length<150 else '150+'
    lengths[group][row['frames_with_stop']]+=n
n=sum(counts.values());unique=counts[5]/n*100
pf.use_style();fig,axes=pf.figure(pf.TWO_COLUMN_MM,100,ncols=2,layout=None)
blue='#0072B2';grey='#999999'
fig.suptitle(f'One stop-free frame occurs in {unique:.1f}% of unambiguous reads',y=.99)
fig.legend(handles=[Patch(color=blue,label='Exactly one stop-free frame'),Patch(color=grey,label='Zero or multiple stop-free frames')],loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
ax=axes[0];xs=list(range(7));ys=[counts[x]/n*100 for x in xs]
ax.bar(xs,ys,color=[blue if x==5 else grey for x in xs]);ax.set(xticks=xs,xlabel='Frames containing a stop codon (out of six)',ylabel='Reads (%)',ylim=(0,42))
for x,y in zip(xs,ys):ax.text(x,y+1,f'{y:.1f}%',ha='center',va='bottom')
ax.set_title(f'{n:,} unambiguous reads')
ax=axes[1];groups=[x for x in ['20–49','50–99','100–149','150+'] if sum(lengths[x].values())]
ys=[100*lengths[x][5]/sum(lengths[x].values()) for x in groups]
ax.bar(range(len(groups)),ys,color=blue);ax.set(xticks=range(len(groups)),xticklabels=groups,xlabel='Trimmed read length (bases)',ylabel='Reads with one stop-free frame (%)',ylim=(0,100))
for i,(g,y) in enumerate(zip(groups,ys)):ax.text(i,y+2,f'{y:.1f}%\nn={sum(lengths[g].values()):,}',ha='center',va='bottom')
for a,label in zip(axes,'ab'):pf.panel_label(a,label)
fig.subplots_adjust(top=.73,bottom=.25,wspace=.4)
fig.text(.5,.025,'First 100,000 reads per mate in six complete oyster libraries. Standard genetic code. 625 reads with ambiguous bases excluded.\nA stop-free frame is not proof of the coding frame; no reference-based frame assignment is used.',ha='center',va='bottom')
# Clear-figures: seven categories retained; blue means exactly one stop-free frame in both panels; legend precedes marks.
pf.save(fig,root/'oyster_trimmed_reads_stop_frames')
svg=root/'oyster_trimmed_reads_stop_frames.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
p=root/'oyster_trimmed_reads_stop_frames.png'
with Image.open(p) as im:im.convert('RGB').quantize(colors=256).save(p)
summary=dict(total_reads=1200000,excluded_ambiguous=625,plotted_reads=n,frames_with_stop_counts=dict(counts),exactly_one_stop_free_percent=unique)
(root/'results/stop-frame-plot-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
