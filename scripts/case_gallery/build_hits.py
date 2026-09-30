"""Per query and alphabet/k of notebook 241: the top 20 human proteins by best region
E-value (by best mean IDF where the arm has no finite E-value), plus the known partner,
with every search region's coordinates and residues. Coordinates 0-based half-open."""
from pathlib import Path as _P
import os as _os
_os.chdir(_P(__file__).resolve().parents[2] / "tables" / "case_gallery")  # reads and writes the JSON tables there

import json, math, polars as pl
from pathlib import Path
D=Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
HUMAN=Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
TOP=20
seqs={}; name=None
for l in HUMAN.read_text().splitlines():
    if l.startswith(">"): name=l[1:].strip(); seqs[name]=[]
    else: seqs[name].append(l.strip())
seqs={k:"".join(v) for k,v in seqs.items()}
q={}; name=None
for l in (D/"queries.fa").read_text().splitlines():
    if l.startswith(">"): name=l[1:].strip(); q[name]=""
    else: q[name]+=l.strip()
partner={"Ced9":"BCL2","P66":"CD47","BHF":None}
fin=lambda x: x is not None and math.isfinite(x)
reg=pl.read_parquet(D/"regions.parquet",columns=["alphabet","ksize_arm","query_name","target_name","gene","region_start","region_end","target_start","target_end","region_evalue","region_mean_idf"])
reg=reg.with_columns(pl.when(pl.col("region_evalue").is_infinite()).then(None).otherwise(pl.col("region_evalue")).alias("E"))
out={}; found={}
for (a,k,qq),g in reg.group_by(["alphabet","ksize_arm","query_name"]):
    per=g.group_by("target_name","gene").agg(pl.col("E").min().alias("bestE"),pl.col("region_mean_idf").max().alias("bestIDF"))
    hasE=per["bestE"].is_not_null().any()
    per=per.sort(["bestE","bestIDF","gene"],descending=[False,True,False],nulls_last=True) if hasE else per.sort(["bestIDF","gene"],descending=[True,False])
    per=per.with_row_index("pos",1)
    keep=per.head(TOP)
    if partner[qq]:
        pr=per.filter(pl.col("gene")==partner[qq])
        if pr.height and pr["pos"][0]>TOP: keep=pl.concat([keep,pr])
    rows=[]
    for r in keep.iter_rows(named=True):
        t=seqs[r["target_name"]]
        rs=g.filter(pl.col("target_name")==r["target_name"]).sort("region_start")
        regs=[]
        for x in rs.iter_rows(named=True):
            a0,a1,b0,b1=int(x["region_start"]),int(x["region_end"]),int(x["target_start"]),int(x["target_end"])
            regs.append([a0,a1,b0,b1,x["E"] if fin(x["E"]) else None,round(x["region_mean_idf"],1),q[qq][a0:a1],t[b0:b1]])
        rows.append({"pos":r["pos"],"gene":r["gene"],"tl":len(t),"E":r["bestE"],"idf":round(r["bestIDF"],1),"rg":regs})
    out[f"{qq}|{a}|{k}"]={"by":"E-value" if hasE else "mean IDF","n":per.height,"rows":rows}
print(len(out)); s=json.dumps(out,separators=(",",":")); print(len(s))
open("hits241.json","w").write(s)
