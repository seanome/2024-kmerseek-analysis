from pathlib import Path
from xml.sax.saxutils import escape as e
out=Path(__file__).with_name('nextflow-migration-plan.svg')
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1040" viewBox="0 0 1500 1040">','<rect width="1500" height="1040" fill="#fff"/>','<style>text{font-family:Arial,sans-serif;fill:#202b30}.h{font-size:27px;font-weight:bold}.t{font-size:17px}.small{font-size:15px}.station{fill:white;stroke-width:4}</style>']
def text(x,y,t,cl='t'):s.append(f'<text x="{x}" y="{y}" class="{cl}">{e(t)}</text>')
def line(d,c,w=5):s.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>')
def station(x,y,title,detail,c):
 s.append(f'<circle cx="{x}" cy="{y}" r="9" class="station" stroke="{c}"/>');text(x-15,y-28,title);text(x-15,y+40,detail,'small')
blue='#0072B2';green='#00856A';orange='#B85A00';purple='#79519A'
text(35,43,'Nextflow migration plan — preserve completed work; replace only missing stages','h')
for x,c,t in [(35,blue,'Public references'),(365,green,'Oyster queries'),(685,purple,'Validation and reuse'),(1080,orange,'Rollout gates')]:
 line(f'M{x},80 h30',c);text(x+42,86,t)
text(35,124,'PLAN ONLY: Nextflow is not implemented. Access renewal is required for Sherlock tests.','t')
s.append('<rect x="25" y="145" width="1450" height="110" rx="12" fill="#F0F5F5"/>')
text(45,174,'Fixed translation rule','h');text(45,204,'ALL SIX FRAMES  →  split ONLY at stop codons (*)  →  preserve internal X  →  keep stretches ≥30 amino acids')
text(45,234,'Keep original DNA and query IDs. No proteome-match filter. Test X handling in the pinned sketcher before adoption.','small')
text(35,292,'1  Audit before reuse','h');line('M75,345 H1340',purple)
station(75,345,'Existing outputs','Files + manifests + job logs',purple)
station(440,345,'Content validation','Checksums · settings · IDs · ownership',purple)
station(910,345,'Classify','Compatible / historical / missing',purple)
text(35,430,'2  Missing reference stages only — separate Slurm tasks','h');line('M75,490 H1340',blue)
for x,t,d in [(75,'Download','FASTQ + verified source MD5'),(370,'Trim','Trimmed FASTQ + trim reports'),(700,'Translate','≥30aa FASTA + origin/frame IDs'),(1080,'manysketch','Sketches + settings manifest')]:station(x,490,t,d,blue)
text(45,558,'Compatible validated reference sketches bypass download/trim/sketch. Historical translated sketches do not substitute for the corrected protocol.','small')
text(35,611,'3  Oyster query stages — retain every original query in membership records','h');line('M75,670 H1340',green)
for x,t,d in [(75,'Full-depth inputs','SPLASH extensions + all DE-kupl contigs'),(545,'Translate + manysketch','Same stop-only ≥30aa rule'),(1080,'Query collection','Sketches + explicit empty-query IDs')]:station(x,670,t,d,green)
text(35,755,'4  Retrieval and audit','h');line('M75,810 H1340',purple)
for x,t,d in [(75,'Branchwater index','Compatible reference sketches'),(425,'Branchwater manysearch','Reference index + query sketches'),(900,'Independent audit','Match TSV/CSV + exact scope')]:station(x,810,t,d,purple)
text(35,900,'5  Prove recovery before transferring production ownership','h');line('M75,945 H1340',orange)
for x,t,d in [(75,'Required tests','Reuse · resume · interruption'),(530,'Required tests','Query change · corruption · protocol mismatch'),(1120,'Then roll out','Drain safely · trace · report')]:station(x,945,t,d,orange)
text(35,1022,'No blanket cancellations. No duplicate writers. No deletion of historical outputs. Public runs receive reference preparation only.','small')
s.append('</svg>');out.write_text('\n'.join(s))
