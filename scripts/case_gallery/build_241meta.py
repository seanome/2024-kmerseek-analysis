from pathlib import Path as _P
import os as _os
_os.chdir(_P(__file__).resolve().parents[2] / "tables" / "case_gallery")  # reads and writes the JSON tables there
import json, polars as pl
from pathlib import Path
D=Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")  # plan.json and ranks.csv are copied to tables/case_gallery/241/
plan=json.loads((D/"plan.json").read_text())
rk=pl.read_csv(D/"ranks.csv",schema_overrides={"rank":pl.Float64,"n_tied":pl.Float64,"partner_value":pl.Float64})
rg=pl.read_parquet(D/"regions.parquet").filter(pl.col("gene").is_in(["BCL2","CD47"]))
pd=json.load(open("pairdata.json"))
meta={}; off=0; tot=0
for q,partner in [("Ced9","BCL2"),("P66","CD47")]:
    for arm in plan:
        a,k=arm["alphabet"],arm["ksize"]
        key=f"{q}|{a}|{k}"
        s=rg.filter(pl.col("query_name")==q,pl.col("alphabet")==a,pl.col("ksize_arm")==k,pl.col("gene")==partner).sort("region_start")
        sr=[[int(x["region_start"]),int(x["region_end"]),int(x["target_start"]),int(x["target_end"]),x["region_evalue"]] for x in s.iter_rows(named=True)]
        rows=rk.filter(pl.col("query")==q,pl.col("alphabet")==a,pl.col("ksize")==k)
        ranks={}
        for m in ["E-value","mean IDF","bit score","shared k-mers"]:
            r=rows.filter(pl.col("metric")==m)
            if r.height:
                x=r.row(0,named=True)
                ranks[m]=[x["rank"],x["n_targets"],x["n_tied"]]
        p=pd[key]
        for g in sr:
            tot+=1
            if not any(h[0]-h[2]==g[0]-g[2] and h[0]<g[1] and h[1]>g[0] for h in p["rg"]): off+=1
        meta[key]={"bits":arm["bits"],"sr":sr,"ranks":ranks}
print("search regions not on a pair diagonal:",off,"of",tot)
json.dump(meta,open("meta241.json","w"),separators=(",",":"))
print(meta["Ced9|hp_lehninger2|17"]); print(meta["P66|hp_lehninger2|17"])
