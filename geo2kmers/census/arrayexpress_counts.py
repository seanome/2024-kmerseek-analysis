"""ArrayExpress RNA array-assay studies outside E-GEOD (search window is 10,000 hits, so only the non-GEO set is listed): accessions, then Sample count for each non-GEO-import study."""
import urllib.request, urllib.parse, json, time, sys, collections
from concurrent.futures import ThreadPoolExecutor
S="https://www.ebi.ac.uk/biostudies/api/v1/"
def get(u):
    for i in range(12):
        try: return json.load(urllib.request.urlopen(u, timeout=120))
        except Exception as e: err=e; time.sleep(min(60,3*(i+1)))
    raise err
accs=[]; p=1
while True:
    q=urllib.parse.urlencode({"facet.technology":"Array assay","facet.assay_by_molecule":"RNA assay","query":"*:* AND NOT accession:E-GEOD*","pageSize":100,"page":p,"sortBy":"release_date","sortOrder":"ascending"})
    h=get(S+"arrayexpress/search?"+q)["hits"]
    if not h: break
    accs+= [(x["accession"],x.get("release_date","")) for x in h]; p+=1
    if p%50==0: print("page",p,len(accs),file=sys.stderr,flush=True)
accs=list(dict(accs).items())
pref=collections.Counter(a.split("-")[1] if a.count("-")>=2 else a for a,_ in accs)
types={}
def nsamp(a):
    try: d=get(S+"studies/"+a)
    except Exception: return a,None
    types[a]=[x.get("value") for x in d.get("section",{}).get("attributes",[]) or [] if x.get("name")=="Study type"]
    for sub in d.get("section",{}).get("subsections",[]) or []:
        subs = sub if isinstance(sub,list) else [sub]
        for s in subs:
            for at in s.get("attributes",[]) or []:
                if at.get("name")=="Sample count":
                    try: return a,int(at["value"])
                    except: pass
    return a,None
non=[(a,r) for a,r in accs if not a.startswith("E-GEOD")]
with ThreadPoolExecutor(8) as ex: res=list(ex.map(lambda t:nsamp(t[0]), non))
rel=dict(non); yr=collections.Counter()
for a,n in res:
    if n: yr[rel[a][:4]]+=n
json.dump(dict(per_study_types=types, studies=len(accs), prefix_counts=pref.most_common(), non_geod_studies=len(non),
  non_geod_with_count=sum(n is not None for _,n in res), non_geod_samples=sum(n or 0 for _,n in res),
  samples_by_year=dict(sorted(yr.items())), per_study=dict(res)), open(sys.argv[1],"w"), indent=1)
