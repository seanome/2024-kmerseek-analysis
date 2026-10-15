"""Build compact `kmerseek pair` data for the case gallery.

- 183 notebook-244 cases: the pair JSONs already on olgabot/244-all-case-figures (68d6946).
- Ced9 -> BCL2 and P66 -> CD47: kmerseek pair for every arm of notebook 241's ladder.
All new runs use the same binary as the 183 cases (kmerseek 0.3.1, 921baa7).
Writes pairdata.json: {case key: compact pair}.
"""
import json, subprocess
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "tables" / "case_gallery"
BIN = Path.home() / "code/kmerseek-244-pairs/target/release/kmerseek"
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
D241 = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
OUT = S / "newpairs"
OUT.mkdir(exist_ok=True)
NBLOCK = 8


def compact(pj, call=None):
    """call: (qs, qe, ts, te) 1-based inclusive, or None."""
    regs = pj["regions"]
    rg = [[g["query_start"], g["query_end"], g["target_start"], g["target_end"]] for g in regs]
    call_i = None
    if call:
        want = (call[0] - 1, call[1], call[2] - 1, call[3])
        exact = [i for i, g in enumerate(rg) if tuple(g) == want]
        if exact:
            call_i, how = exact[0], "exact"
        else:
            # same diagonal, containing the call
            cont = [i for i, g in enumerate(rg)
                    if g[0] - g[2] == want[0] - want[2] and g[0] <= want[0] and g[1] >= want[1]]
            call_i, how = (cont[0], "contains") if cont else (None, "none")
    order = sorted(range(len(rg)), key=lambda i: (-(rg[i][1] - rg[i][0]), rg[i][0], rg[i][2]))
    show = order[:NBLOCK]
    if call_i is not None and call_i not in show:
        show.append(call_i)
    q, t = pj["query"], pj["target"]
    blocks = []
    for i in show:
        a, b, c, d = rg[i]
        blocks.append({"i": i, "q": q["sequence"][a:b], "t": t["sequence"][c:d],
                       "qe": q["encoded"][a:b], "te": t["encoded"][c:d]})
    km = []
    for s in pj["shared_kmers"]:
        km += [s["query_pos"], s["target_pos"]]
    out = {"a": pj["moltype"], "k": pj["ksize"], "cls": pj["classes"],
           "qn": q["name"], "tn": t["name"], "ql": len(q["sequence"]), "tl": len(t["sequence"]),
           "km": km, "rg": rg, "bl": blocks}
    if call:
        out["call"] = list(call)
        out["callI"] = call_i
        out["callHow"] = how
    return out


def run_pair(qfa, qname, tfa, tname, k, a, dest):
    if not dest.exists():
        subprocess.run([BIN, "pair", "-q", qfa, "--query-name", qname, "-t", tfa,
                        "--target-name", tname, "-k", str(k), "-a", a, "-o", dest],
                       check=True, capture_output=True)
    return json.loads(dest.read_text())


def main():
    data = {}
    # 1. the 183 cases
    calls = pl.read_csv(ROOT / "tables" / "244_case_calls.csv", infer_schema_length=None).filter(
        pl.col("tool") == "kmerseek chosen arm")
    for f in sorted((ROOT / "figures" / "244_pairs").glob("*/*.pair.json")):
        cid = int(f.name[:3])
        c = calls.filter(pl.col("case_id") == cid).row(0, named=True)
        call = (int(c["query_start"]), int(c["query_end"]), int(c["target_start"]), int(c["target_end"]))
        data[str(cid)] = compact(json.loads(f.read_text()), call)
    hows = [v["callHow"] for v in data.values()]
    print("244 cases:", len(data), {h: hows.count(h) for h in set(hows)})

    # 2. Ced9 -> BCL2, P66 -> CD47 over notebook 241's ladder
    headers = {l[1:].strip().split("|")[6]: l[1:].strip() for l in HUMAN.read_text().splitlines()
               if l.startswith(">")}
    plan = json.loads((D241 / "plan.json").read_text())
    for q, partner in [("Ced9", "BCL2"), ("P66", "CD47")]:
        for arm in plan:
            a, k = arm["alphabet"], arm["ksize"]
            pj = run_pair(D241 / "queries.fa", q, HUMAN, headers[partner], k, a, OUT / f"{q}.{a}.k{k}.json")
            data[f"{q}|{a}|{k}"] = compact(pj)
    print("241 arms per query:", len(plan))

    (S / "pairdata.json").write_text(json.dumps(data, separators=(",", ":")))
    print("pairdata bytes:", (S / "pairdata.json").stat().st_size)


if __name__ == "__main__":
    main()
