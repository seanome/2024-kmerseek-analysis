"""Count GEO expression samples, and SRA RNA-seq experiments, with NCBI E-utilities.

Three counts, each written to counts/<name>.json:

  platform   GEO samples (GSM) by platform technology and sample type, from esearch
             counts alone. "Array platform AND sample type RNA" is the array total
             used in the README (1,827,810 on 2026-11-08).
  series     unique GSMs in all series of one GEO series type, read from esummary,
             by release year of the series that first lists each GSM. A GSM in two
             series is counted once.
  sra        SRA experiments (SRX) by strategy and source. SRA's Entrez database
             counts experiments, not samples; one sample can have several.

Usage:
  python geo_counts.py platform
  python geo_counts.py series "expression profiling by array" counts/geo_array_series.json
  python geo_counts.py sra
"""

import collections
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
COUNTS = Path(__file__).resolve().parent / "counts"

ARRAY_PLATFORMS = [
    "in situ oligonucleotide",
    "spotted oligonucleotide",
    "spotted DNA/cDNA",
    "oligonucleotide beads",
    "mixed spotted oligonucleotide/cDNA",
]
OTHER_PLATFORMS = ["high-throughput sequencing", "RT-PCR", "antibody", "MPSS", "other"]
SAMPLE_TYPES = ["RNA", "genomic", "SRA", "protein", "mixed", "other"]


def post(path, **params):
    """POST to E-utilities, retrying on network errors; NCBI allows 3 requests/s without a key."""
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(5):
        try:
            body = urllib.request.urlopen(EUTILS + path, data=data, timeout=120).read()
            time.sleep(0.35)
            return body
        except OSError as err:
            last = err
            time.sleep(2 * (attempt + 1))
    raise last


def count(term, db="gds"):
    body = post("esearch.fcgi", db=db, term=term, retmax=0).decode()
    return int(re.search(r"<Count>(\d+)</Count>", body).group(1))


def platform_counts():
    arrays = " OR ".join(f'"{p}"[PTYP]' for p in ARRAY_PLATFORMS)
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "all_gsm": count("gsm[ETYP]"),
        "all_gse": count("gse[ETYP]"),
        "gsm_by_platform": {p: count(f'gsm[ETYP] AND "{p}"[PTYP]') for p in ARRAY_PLATFORMS + OTHER_PLATFORMS},
        "gsm_by_sample_type": {s: count(f'gsm[ETYP] AND "{s}"[STYP]') for s in SAMPLE_TYPES},
        "array_gsm": count(f"gsm[ETYP] AND ({arrays})"),
        "array_gsm_rna": count(f'gsm[ETYP] AND ({arrays}) AND "RNA"[STYP]'),
        "array_gsm_genomic": count(f'gsm[ETYP] AND ({arrays}) AND "genomic"[STYP]'),
    }
    return out


def series_counts(series_type, max_series=None):
    """Unique GSMs in every GSE of one series type, with the year each GSM was first released."""
    found = json.loads(
        post(
            "esearch.fcgi",
            db="gds",
            term=f'gse[ETYP] AND "{series_type}"[GTYP]',
            usehistory="y",
            retmax=0,
            retmode="json",
        )
    )["esearchresult"]
    n_series = int(found["count"]) if max_series is None else min(max_series, int(found["count"]))
    gsms, by_year, by_taxon = set(), collections.Counter(), collections.Counter()
    sum_n_samples = 0
    for start in range(0, n_series, 500):
        result = json.loads(
            post(
                "esummary.fcgi",
                db="gds",
                WebEnv=found["webenv"],
                query_key=found["querykey"],
                retstart=start,
                retmax=min(500, n_series - start),
                retmode="json",
            )
        )["result"]
        for uid in result.get("uids", []):
            series = result[uid]
            accessions = [s["accession"] for s in series.get("samples", [])]
            new = [a for a in accessions if a not in gsms]
            gsms.update(accessions)
            sum_n_samples += int(series.get("n_samples") or 0)
            by_year[series.get("pdat", "")[:4]] += len(new)
            by_taxon[series.get("taxon", "").split(";")[0]] += len(new)
        print(series_type, start, len(gsms), file=sys.stderr, flush=True)
    return {
        "date": time.strftime("%Y-%m-%d"),
        "series_type": series_type,
        "series": n_series,
        "unique_gsm": len(gsms),
        "sum_n_samples": sum_n_samples,
        "by_year": dict(sorted(by_year.items())),
        "top_taxa": by_taxon.most_common(8),
    }


def sra_counts():
    terms = {
        "all": "all[filter]",
        "rna_seq": '"rna seq"[Strategy]',
        "rna_seq_bulk": '"rna seq"[Strategy] AND "transcriptomic"[Source]',
        "rna_seq_single_cell": '"rna seq"[Strategy] AND "transcriptomic single cell"[Source]',
        "rna_seq_human": '"rna seq"[Strategy] AND "homo sapiens"[Organism]',
        "rna_seq_mouse": '"rna seq"[Strategy] AND "mus musculus"[Organism]',
        "rna_seq_public": '"rna seq"[Strategy] AND "public"[Access]',
        "rna_seq_in_geo": '"rna seq"[Strategy] AND "sra gds"[Filter]',
        "mirna_seq": '"mirna seq"[Strategy]',
    }
    return {"date": time.strftime("%Y-%m-%d"), "unit": "SRA experiments (SRX)"} | {
        name: count(term, db="sra") for name, term in terms.items()
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "platform":
        out, path = platform_counts(), COUNTS / "geo_platform.json"
    elif mode == "series":
        max_series = int(sys.argv[4]) if len(sys.argv) > 4 else None
        out, path = series_counts(sys.argv[2], max_series), Path(sys.argv[3])
    elif mode == "sra":
        out, path = sra_counts(), COUNTS / "sra_experiments.json"
    else:
        raise SystemExit(__doc__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:1500])
