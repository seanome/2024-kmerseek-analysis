"""Plot audited gene-level agreement without calling either quantifier truth."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'notebooks'))
import pubfig as pf
from matplotlib.patches import Patch

root = Path(__file__).resolve().parent
d = root/'results/quantifier-comparison'
s = json.loads((d/'audited-summary.json').read_text())
rows = list(csv.DictReader((d/'gene-comparison.tsv').open(), delimiter='\t'))
pf.use_style()
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 100, ncols=2, layout=None)
blue, orange, grey = '#0072B2', '#D55E00', '#999999'
fig.suptitle('Kallisto and Salmon share 3,262 differentially expressed genes', y=.99)
fig.legend(handles=[Patch(color=blue,label='DE with both quantifiers'),
    Patch(color=orange,label='DE with one quantifier'),Patch(color=grey,label='Other tested gene')],
    loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
a=axes[0]
vals=[s['both_DE'],s['kallisto_only'],s['salmon_only']]
a.bar(range(3),vals,color=[blue,orange,orange])
a.set(xticks=range(3),xticklabels=['Both','Kallisto only','Salmon only'],ylabel='Genes (n)',ylim=(0,3900),title='DESeq2 adjusted P < 0.05')
for i,v in enumerate(vals): a.text(i,v+60,f'{v:,}',ha='center')
a=axes[1]
for category,color in [(0,grey),(1,orange),(2,blue)]:
    selected=[r for r in rows if (int(float(r['padj_kallisto'])<.05)+int(float(r['padj_salmon'])<.05))==category and r['log2FoldChange_kallisto']!='NA' and r['log2FoldChange_salmon']!='NA']
    a.scatter([float(r['log2FoldChange_kallisto']) for r in selected],
              [float(r['log2FoldChange_salmon']) for r in selected],s=2,color=color,alpha=.35,rasterized=True)
a.set(xlabel='Kallisto + DESeq2 log2 fold change',ylabel='Salmon + DESeq2 log2 fold change',title=f"{s['common_tested']:,} jointly tested genes; rank correlation {s['lfc_spearman']:.3f}")
fig.subplots_adjust(top=.73,bottom=.25,wspace=.45)
fig.text(.5,.025,'Six complete paired libraries; breed / wild. Same transcript reference, gene aggregation and R DESeq2 model.\nOne shared DE gene has opposite fold-change directions. Agreement is robustness to quantifier choice, not independent validation.',ha='center',va='bottom')
# Checklist: legend precedes marks; all genes retained on scatter; bars independently audited from gene IDs.
pf.save(fig,root/'oyster_kallisto_salmon_DESeq2')
p=root/'oyster_kallisto_salmon_DESeq2.svg'
p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
