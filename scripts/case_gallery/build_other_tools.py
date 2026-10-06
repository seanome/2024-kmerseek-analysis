"""Other tools' hits for Ced9, P66 and BHF against GENCODE v49 canonical human proteins.
Input: other_tools/*.tsv (query, target header, qstart, qend, tstart, tend, bits, evalue;
1-based inclusive). Output: other_tools.json {query: {tool: [{gene, tl, qs, qe, ts, te, E}]}},
each tool's hits sorted by E-value."""
from pathlib import Path as _P
import os as _os
_os.chdir(_P(__file__).resolve().parents[2] / "tables" / "case_gallery")  # reads and writes the JSON tables there

import json, csv
from pathlib import Path
T=Path(__file__).resolve().parents[2] / "tables" / "case_gallery" / "other_tools"
TOOLS=[("phmmer","phmmer"),("jackhmmer","jackhmmer (3 rounds)"),("mmseqs_it1","MMseqs2"),("mmseqs_it3","MMseqs2 (3 rounds)"),("blastp","blastp")]
out={q:{} for q in ["Ced9","P66","BHF"]}
for f,name in TOOLS:
    for q in out: out[q][name]=[]
    for r in csv.reader(open(T/f"{f}.tsv"),delimiter="\t"):
        h=r[1].split("|")
        out[r[0]][name].append({"gene":h[6],"tl":int(h[7]),"qs":int(r[2]),"qe":int(r[3]),"ts":int(r[4]),"te":int(r[5]),"E":float(r[7])})
    for q in out: out[q][name].sort(key=lambda x:x["E"])
json.dump(out,open("other_tools.json","w"),separators=(",",":"))
for q in out: print(q,{t:len(v) for t,v in out[q].items()}, [ (t,v[0]["gene"],v[0]["E"]) for t,v in out[q].items() if v][:5])
