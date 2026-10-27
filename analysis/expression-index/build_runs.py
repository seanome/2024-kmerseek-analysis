"""Build runs.tsv: every SRA run to index, with where each accession came from.

No accession, tissue or condition label is written here by hand. Botryllus runs come from
the BioProject in Domen et al. 2026's data availability statement; human runs from the
GEO series of MetaMEx Table 2 (Pillon et al. 2020) plus GSE97084 and GSE167186; mouse and
zebrafish runs from the SRA searches in SEARCHES, each run recorded with its query.

    python build_runs.py OUT_DIR

Writes OUT_DIR/runs.tsv (columns as the plan asks, plus metadata columns kept verbatim)
and OUT_DIR/series_status.tsv (one row per human series: microarray, no open RNA-seq run,
controlled, or kept).
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from pysradb.sraweb import SRAweb

HERE = Path(__file__).parent
SOURCES = HERE / "sources"
EXTRA_HUMAN = ["GSE97084", "GSE167186"]  # named in the plan, added if Table 2 lacks them
MAX_PER_THEME = 6
# (organism, theme, query). Each query is sent to SRA as written and recorded per run.
SEARCHES = [
    (org, theme, f'"{org_q}"[Organism] AND "rna seq"[Strategy] AND "transcriptomic"[Source] AND {terms}')
    for org, org_q in (("mouse", "Mus musculus"), ("zebrafish", "Danio rerio"))
    for theme, terms in (
        ("skeletal muscle after exercise", '"skeletal muscle" AND exercise'),
        ("skeletal muscle young versus old", '"skeletal muscle" AND (aging OR aged) AND young'),
        ("whole animal or heart young versus old", '(heart OR "whole body" OR "whole animal") AND (aging OR aged) AND young'),
    )
]
META_KEEP = ["tissue", "source_name", "cell type", "age", "age in days", "stage", "condition",
             "treatment", "time", "time point", "time of collection", "timepoint", "exercise",
             "training", "intervention", "group", "experimental group", "sex", "type"]

db = SRAweb()
CACHE = HERE / ".sra_cache"
CACHE.mkdir(exist_ok=True)


def cached(key, fn):
    p = CACHE / (re.sub(r"\W+", "_", key)[:150] + ".pkl")
    if p.exists():
        return pd.read_pickle(p)
    for attempt in range(8):
        time.sleep(0.5)  # NCBI allows 3 requests a second without an API key
        try:
            out = fn()
            break
        except NoSraExperiment:
            raise
        except Exception as e:  # NCBI rate limits and timeouts; retry, then give up loudly
            if attempt == 7:
                raise
            time.sleep(10 * (attempt + 1))
    pd.to_pickle(out, p)
    return out


class NoSraExperiment(Exception):
    pass


def gse_to_srp(gse):
    """pysradb's gse_to_srp, with its zero-result crash turned into NoSraExperiment.

    pysradb reads esearchresult["count"]; a successful search with no hits lacks "count"
    in that call's response, while a rate-limit reply lacks "esearchresult" itself.
    """
    try:
        return db.gse_to_srp([gse])
    except KeyError as e:
        if e.args == ("count",):
            raise NoSraExperiment(gse)
        raise


def geo_series_type(gse):
    """GEO's own record of the series: type, taxon and sample count."""
    import urllib.request
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    def get():
        with urllib.request.urlopen(f"{base}esearch.fcgi?db=gds&retmode=json&term={gse}[ACCN]+AND+gse[ETYP]", timeout=60) as r:
            ids = json.load(r)["esearchresult"]["idlist"]
        if not ids:
            return "not found in GEO"
        time.sleep(0.5)
        with urllib.request.urlopen(f"{base}esummary.fcgi?db=gds&retmode=json&id={ids[0]}", timeout=60) as r:
            d = json.load(r)["result"][ids[0]]
        return f"{d.get('gdstype', '')} | {d.get('taxon', '')} | n_samples={d.get('n_samples')}"
    return cached(f"geotype_{gse}", get)


def sra_record_count(term):
    """How many SRA records NCBI's search finds for term (pysradb crashes when it is 0)."""
    import urllib.request
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&retmode=json&term={term}"
    def get():
        with urllib.request.urlopen(url, timeout=60) as r:
            return int(json.load(r)["esearchresult"]["count"])
    return cached(f"count_{term}", get)


def metadata(acc):
    m = cached(f"meta_{acc}", lambda: db.sra_metadata(acc, detailed=True))
    return m if m is not None else pd.DataFrame()


def condition_text(row):
    """The sample's own metadata fields, verbatim, joined as key=value."""
    parts = [f"{k}={row[k]}" for k in META_KEEP if k in row and pd.notna(row[k]) and str(row[k]).strip()]
    return "; ".join(parts)


def access(row):
    vals = {str(row.get(c, "")).lower() for c in ("ncbi_access_type", "aws_access_type", "gcp_access_type")}
    return "controlled" if any("controlled" in v for v in vals) else "open"


def size_gb(row):
    sizes = [float(s) for s in str(row.get("public_size", "")).split(",") if s.strip().isdigit()]
    if not sizes and str(row.get("total_size", "")).isdigit():
        sizes = [float(row["total_size"])]
    return round(max(sizes) / 1e9, 4) if sizes else None


def rows_from(m, organism, source, query, extra):
    out = []
    for _, r in m.iterrows():
        out.append({
            "accession": r["run_accession"], "organism": organism,
            "tissue": r.get("tissue") if pd.notna(r.get("tissue", None)) else r.get("source_name", ""),
            "condition": condition_text(r), "source": source, "built_from": "reads",
            "size_gb": size_gb(r), "query": query, "access": access(r),
            "library_strategy": r.get("library_strategy"), "library_layout": r.get("library_layout"),
            "organism_name": r.get("organism_name"), "sample_title": r.get("sample_title"),
            "study_accession": r.get("study_accession"), "spots": r.get("run_total_spots"),
            "bases": r.get("run_total_bases"),
            **{c: r.get(c) for c in ("ena_fastq_http", "ena_fastq_http_1", "ena_fastq_http_2", "ncbi_url", "public_url")},
            **extra,
        })
    return out


def main(out_dir):
    out_dir = Path(out_dir)
    rows = []

    # a. Botryllus: BioProjects named in the paper's data availability statement.
    statement = (SOURCES / "domen2026_data_availability.txt").read_text()
    for bp in sorted(set(re.findall(r"\bPRJ[A-Z]{2}\d+\b", statement))):
        rows += rows_from(metadata(bp), "Botryllus schlosseri", "paper", bp, {"gse": "", "table2_rows": ""})

    # b. Human: MetaMEx Table 2 plus the two named series.
    t2 = pd.read_csv(SOURCES / "metamex_table2.tsv", sep="\t", comment="#")
    t2 = t2[t2["GEO"].str.match(r"^GSE\d+$")]
    series = {}
    for _, r in t2.iterrows():
        series.setdefault(r["GEO"], []).append(f"{r['table2_row']}:{r['Protocol']}")
    for g in EXTRA_HUMAN:
        series.setdefault(g, ["added: named in the plan"])
    status = []
    for gse, t2rows in series.items():
        try:
            srp = cached(f"gse2srp_{gse}", lambda: gse_to_srp(gse))
        except NoSraExperiment:
            status.append({"gse": gse, "table2_rows": " | ".join(t2rows), "status": "no SRA experiment (pysradb: the series' GEO samples link to none)", "runs_open_rnaseq": 0, "runs_controlled": 0, "runs_other": 0})
            continue
        srps = [] if srp is None or srp.empty else list(srp["study_accession"].dropna().unique())
        if not srps:
            status.append({"gse": gse, "table2_rows": " | ".join(t2rows), "status": "no SRA study (microarray or not in SRA)", "runs_open_rnaseq": 0, "runs_controlled": 0, "runs_other": 0})
            continue
        n_open = n_ctrl = n_other = 0
        for s in srps:
            m = metadata(s)
            got = rows_from(m, "Homo sapiens", "metamex" if not t2rows[0].startswith("added") else "plan", gse,
                            {"gse": gse, "table2_rows": " | ".join(t2rows)})
            for g in got:
                if g["organism_name"] != "Homo sapiens" or g["library_strategy"] != "RNA-Seq":
                    n_other += 1
                elif g["access"] == "controlled":
                    n_ctrl += 1
                    rows.append({**g, "keep": False, "drop_reason": "controlled access"})
                else:
                    n_open += 1
                    rows.append({**g, "keep": True, "drop_reason": ""})
        status.append({"gse": gse, "table2_rows": " | ".join(t2rows), "status": "SRA " + ",".join(srps),
                       "runs_open_rnaseq": n_open, "runs_controlled": n_ctrl, "runs_other": n_other})
    status = pd.DataFrame(status)
    status["geo_series_type"] = [geo_series_type(g) for g in status["gse"]]
    status.to_csv(out_dir / "series_status.tsv", sep="\t", index=False)

    # c. Mouse and zebrafish searches, at most MAX_PER_THEME runs each.
    from pysradb.search import SraSearch
    for org, theme, q in SEARCHES:
        def run_search():
            s = SraSearch(query=q, return_max=200, verbosity=2)
            s.search()
            return s.get_df()
        hits = cached(f"search_{q}", run_search)
        if hits is None or hits.empty:
            continue
        bulk = hits[~hits["experiment_library_source"].str.upper().str.contains("SINGLE CELL", na=False)]
        runs = list(bulk["run_1_accession"].dropna().unique())[:MAX_PER_THEME]
        for acc in runs:
            m = metadata(acc)
            got = rows_from(m[m["run_accession"] == acc], org.replace("mouse", "Mus musculus").replace("zebrafish", "Danio rerio"),
                            "search", q, {"gse": "", "table2_rows": "", "theme": theme})
            rows += got

    df = pd.DataFrame(rows)
    df["keep"] = df["keep"].fillna(True) if "keep" in df else True
    df.loc[df["access"] == "controlled", "keep"] = False
    df = df.drop_duplicates("accession")
    cols = ["accession", "organism", "tissue", "condition", "source", "built_from", "size_gb", "query"]
    rest = [c for c in df.columns if c not in cols]
    df[cols + rest].to_csv(out_dir / "runs_all.tsv", sep="\t", index=False)
    df[df["keep"].astype(bool)][cols + rest].sort_values("size_gb").to_csv(out_dir / "runs.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
