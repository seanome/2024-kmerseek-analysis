"""BHF across notebook 241's ladder: per arm, the human protein with the best E-value
(or, where the arm has no E-value, the best mean IDF), and kmerseek pair against it."""
from pathlib import Path as _P
import os as _os
_os.chdir(_P(__file__).resolve().parents[2] / "tables" / "case_gallery")  # reads and writes the JSON tables there

import json, subprocess, polars as pl
from pathlib import Path
import sys; sys.path.insert(0, str(_P(__file__).resolve().parent))
from build_pairs import compact, run_pair, D241, HUMAN, OUT
rk=pl.read_csv(D241/"ranks.csv",schema_overrides={"rank":pl.Float64,"n_tied":pl.Float64,"partner_value":pl.Float64}).filter(pl.col("query")=="BHF")
plan=json.loads((D241/"plan.json").read_text())
hdr={}
for l in HUMAN.read_text().splitlines():
    if l.startswith(">"):
        f=l[1:].split("|"); hdr.setdefault(f[6],l[1:].strip())
pd=json.load(open("pairdata.json")); arms={}
for arm in plan:
    a,k=arm["alphabet"],arm["ksize"]
    r=rk.filter(pl.col("alphabet")==a,pl.col("ksize")==k)
    def one(m):
        x=r.filter(pl.col("metric")==m)
        return x.row(0,named=True) if x.height else {"best_value":None,"top_gene":None,"n_targets":None}
    ev,mi=one("E-value"),one("mean IDF")
    use,metric=(ev,"E-value") if ev["best_value"] is not None else (mi,"mean IDF")
    g=use["top_gene"]
    rec={"bits":arm["bits"],"n_targets":ev["n_targets"],"bestE":ev["best_value"],"geneE":ev["top_gene"],"bestIDF":mi["best_value"],"geneIDF":mi["top_gene"],"shown":g,"shownBy":metric}
    if g and g in hdr:
        pj=run_pair(D241/"queries.fa","BHF",HUMAN,hdr[g],k,a,OUT/f"BHFarm.{a}.k{k}.{g}.json")
        pd[f"BHFarm|{a}|{k}"]=compact(pj); rec["tlen"]=len(pj["target"]["sequence"])
    arms[f"{a}|{k}"]=rec
json.dump(arms,open("bhf_arms.json","w"),separators=(",",":"))
json.dump(pd,open("pairdata.json","w"),separators=(",",":"))
print(len(arms), sum(1 for v in arms.values() if v["bestE"] is not None), sum(1 for v in arms.values() if v["bestE"] is not None and v["bestE"]<1))
print(sorted([(v["bestE"],k,v["geneE"]) for k,v in arms.items() if v["bestE"] is not None])[:5])
print(Path("pairdata.json").stat().st_size)
