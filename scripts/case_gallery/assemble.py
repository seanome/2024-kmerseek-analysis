import json, math
"""Rebuild reports/case_gallery/case_gallery.html in place: replace its data block with the JSON
tables in tables/case_gallery/ and the code in scripts/case_gallery/pair_view.js."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
T=str(ROOT/"tables"/"case_gallery")+"/"
PAGE=str(ROOT/"reports"/"case_gallery"/"case_gallery.html")
S=T
s=open(PAGE).read()
i=s.index("const PAIRS="); j=s.index("\nfilter();",i)
def clean(o):
    if isinstance(o,float) and not math.isfinite(o): return None
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    return o
import polars as _pl
BHFFIG1=_pl.read_csv(T+"bhf_fig1_ranks.csv").to_dicts()
# the hit lists are too big to embed: one file per query next to the page, fetched on demand
_h=json.load(open(T+"hits241.json"))
for _q in ["Ced9","P66","BHF"]:
    open(str(ROOT/"reports"/"case_gallery"/("hits241."+_q+".json")),"w").write(json.dumps(clean({k:v for k,v in _h.items() if k.startswith(_q+"|")}),separators=(",",":")))
L=lambda f: json.dumps(clean(json.load(open(S+f))),separators=(",",":"))
js=open(str(ROOT/"scripts"/"case_gallery"/"pair_view.js")).read().replace("' '.repeat(w)+'classes'.padStart(5).slice(-5)+' '","'classes'.padEnd(w+6)")
block="const PAIRS="+L("pairdata.json")+";\nconst META241="+L("meta241.json")+";\nconst BHFARMS="+L("bhf_arms.json")+";\nconst SUMM241="+L("hits241_summary.json")+";\nconst BHFFIG1="+json.dumps(BHFFIG1,separators=(",",":"))+";\nconst OTHER="+L("other_tools.json")+";\n"+js
s=s[:i]+block+s[j:]
css=""".refgrid{display:grid;grid-template-columns:232px minmax(0,1fr);gap:20px;margin-top:10px;position:relative;z-index:0}
.margin{border-right:1px solid var(--line);padding-right:14px}
@media (min-width:821px){.margin{position:sticky;top:calc(env(safe-area-inset-top,0px) + 8px);align-self:start;max-height:calc(100vh - 24px);overflow:auto}}
@media (max-width:820px){.refgrid{grid-template-columns:minmax(0,1fr)}.margin{border-right:none;border-bottom:1px solid var(--line);padding:0 0 12px}}
.mhead{font-weight:600;font-size:14px}
.mnote{font-size:12px;color:var(--soft);margin:4px 0 8px}
.mgrp{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);margin:12px 0 4px}
.malpha{margin:0 0 7px}
.mname{font-family:var(--mono);font-size:12px;color:var(--soft)}
.malpha.cur .mname{color:var(--ink);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:3px;margin-top:2px}
.chip{font-family:var(--mono);font-size:11px;min-width:26px;padding:1px 4px;border-radius:4px;border:1px solid var(--soft);background:var(--panel);color:var(--ink);line-height:1.5}
.chip.on{background:var(--accent);border-color:var(--accent);color:var(--panel)}
.chip.sel{outline:2.5px solid var(--ink);outline-offset:1px}
.chip.demo{display:inline-block;text-align:center;min-width:20px;cursor:default}
.hr{cursor:pointer}.hr:focus-visible rect{stroke:var(--accent)}
.main{min-width:0}
.hh{font-size:15px;margin:16px 0 6px}
.hm svg{min-width:640px}
.hc{cursor:pointer}.hc:focus-visible rect{stroke:var(--accent);stroke-width:3}
"""
if ".refgrid{" not in s: s=s.replace(".hh{font-size:15px",css.split(".hh{")[0]+".hh{font-size:15px",1)
if ".hh{" not in s: s=s.replace(".fig{isolation:isolate}",".fig{isolation:isolate}\n"+css)
if "--hm0:" not in s:
    s=s.replace("--feat:#c9a227;","--hm0:#dbe6f1; --hm1:#8fb1d3; --hm2:#3f74a8; --hm3:#1d3f66; --feat:#c9a227;",1)
    s=s.replace("--feat:#e0b83a;","--hm0:#22364a; --hm1:#3d6a96; --hm2:#7fb0dd; --hm3:#cfe3f5; --feat:#e0b83a;")
open(PAGE,"w").write(s)
i=s.index('<script>')+8; open("/dev/null","w").write(s[i:s.rindex('</script>')])
print(len(s))

EXTRA=""".mswitch{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0 10px;font-size:13px}
.mswitch button[aria-pressed="true"]{background:var(--accent);color:var(--panel);border-color:var(--accent)}
.mswitch .note{flex-basis:100%;margin:2px 0 0}
.tbl{overflow-x:auto;margin:6px 0}
.tbl table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top}
.tbl th{font-weight:600;color:var(--soft);font-size:12px}
.tbl td.num{text-align:right;font-family:var(--mono)}.tbl td.g{font-family:var(--mono);font-weight:600}.tbl td.w{font-family:var(--mono);font-size:11.5px}
.malpha{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 8px;margin:0 0 5px}
.mname{cursor:help;min-width:0}
.chips{display:inline-flex;flex-wrap:wrap;gap:3px;margin:0}
.chip{min-width:24px;padding:0 3px;font-size:10.5px}
.refgrid{grid-template-columns:var(--mw,330px) 10px minmax(0,1fr);gap:10px}
.resizer{cursor:col-resize;border-radius:4px;background:linear-gradient(var(--line),var(--line)) center/2px 100% no-repeat;touch-action:none}
.resizer:hover,.resizer:focus-visible{background:linear-gradient(var(--accent),var(--accent)) center/3px 100% no-repeat;outline:none}
@media (max-width:820px){.refgrid{grid-template-columns:minmax(0,1fr)!important}.resizer{display:none}}
.margin{border-right:none;padding-right:4px}
.wrap.wide{max-width:1500px}
#tip{position:fixed;z-index:10;max-width:360px;background:var(--ink);color:var(--panel);font:12px/1.4 var(--mono);padding:6px 8px;border-radius:6px;pointer-events:none}
details.alltools{margin-top:8px}details.alltools summary{cursor:pointer;font-size:13px;color:var(--accent)}
"""
if ".mswitch{" not in s: s=s.replace("</style>",EXTRA+"</style>",1)
if 'id="tip"' not in s: s=s.replace('<div id="tabpairs" hidden>','<div id="tip" hidden></div>\n<div id="tabpairs" hidden>',1)
open(PAGE,"w").write(s)
i=s.index('<script>')+8; open("/dev/null","w").write(s[i:s.rindex('</script>')])
