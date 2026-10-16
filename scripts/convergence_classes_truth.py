#!/usr/bin/env python3
"""Build the truth sets for the adhesive, shell-matrix and silk pattern classes.

Class A, the ice-binding proteins, is scripts/convergence_afp_truth.py. This script is
the same idea for three more patterns that evolved more than once:

  B adhesive     tyrosine- and lysine-rich underwater glue, where tyrosine becomes DOPA
                 after translation. Mussels and sandcastle worms arrived at it separately.
  C shell        acidic, aspartate- and glutamate-rich, low-complexity matrix proteins of
                 shells, coral skeletons, sea-urchin tests and vertebrate bone.
  D silk         long repeats of short glycine- and alanine-rich units that pack into a
                 beta sheet. Spiders and the moths-plus-caddisflies clade are two origins.

Each class has a control lineage that does the same job with a different pattern, so a
hit there is a leak rather than a find: barnacles for B, bees and ants for D. Class C has
no control lineage; its control is the folded-domain column, the entries a sequence tool
finds by ancestry.

Every query goes to the UniProt REST API, and its URL, the date, the release and the row
count are written to data/convergence_<class>/truth_query_log.tsv. Nothing is typed from
memory.

Two rules in the task brief need data UniProt's search does not return, and each is
resolved here:

  * Class C splits an entry into "acidic or low complexity" and "folded domain" by whether
    its Pfam domains cover at least half its length. The coordinates come from the InterPro
    API, one call per entry, cached under raw/interpro/.
  * Class D keeps an entry with at least one REPEAT feature or a low-complexity fraction
    of at least 30%. SEG itself is not installed, so low complexity is measured the way
    SEG triggers: the share of residues in 12-residue windows whose Shannon entropy over
    the 20 amino acids is below 2.2 bits, SEG's default trigger K1(12). This is SEG's
    trigger, not its full segmentation, and the column says so.

    python scripts/convergence_classes_truth.py            reuse raw/ if present
    python scripts/convergence_classes_truth.py --refetch  download again
"""

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
STREAM = "https://rest.uniprot.org/uniprotkb/stream"
INTERPRO = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot"
RELEASE = "2026_03"
FIELDS = (
    "accession,id,protein_name,organism_name,organism_id,lineage,protein_families,"
    "cc_similarity,keyword,go_id,length,ft_repeat,ft_mod_res,xref_pfam,protein_existence,"
    "reviewed,sequence"
)
# SEG's defaults (Wootton & Federhen 1993): window 12, trigger complexity K1 = 2.2 bits.
SEG_WINDOW, SEG_TRIGGER_BITS, LOW_COMPLEXITY_MIN = 12, 2.2, 0.30
# Accessions per flat-file request; 476 in one URL is refused with HTTP 400.
FLAT_BATCH = 100

# GO:0110148 "biomineralization" is obsolete (QuickGO, read 2026-10-16), so the only
# usable GO term of the two the brief names is the tissue-development one.
CLASSES = {
    # Every class B search carries a taxonomy, and the DOPA one a count, a secreted
    # keyword and a not-antimicrobial clause. Without them "foot protein" matched the
    # split hand/foot malformation protein (a proteasome subunit), "cement protein"
    # matched phage decoration proteins, and one oxidised tyrosine matched hemoglobins,
    # chorion peroxidases and styelins: 15 of 25 entries were not adhesives.
    "adhesive": {
        "prefix": "B",
        "queries": {
            "name_foot": "reviewed:true AND taxonomy_name:Bivalvia AND "
            '(protein_name:"foot protein" OR protein_name:byssal OR '
            'protein_name:"adhesive plaque")',
            "name_cement": 'reviewed:true AND protein_name:"cement protein" AND '
            "(taxonomy_name:Sabellariidae OR taxonomy_name:Thecostraca)",
            "dopa": 'reviewed:true AND ft_mod_res:"3,4-dihydroxyphenylalanine" AND '
            "keyword:KW-0964 AND taxonomy_name:Metazoa NOT keyword:KW-0929",
        },
        "trembl": "reviewed:false AND existence:1 AND ((taxonomy_name:Bivalvia AND "
        '(protein_name:"foot protein" OR protein_name:byssal)) OR '
        '(protein_name:"cement protein" AND (taxonomy_name:Sabellariidae OR '
        "taxonomy_name:Thecostraca)))",
    },
    "shell": {
        "prefix": "C",
        "queries": {
            "keyword": "reviewed:true AND keyword:KW-0091",
            "go_biomineral_tissue": "reviewed:true AND go:0031214",
        },
        "trembl": "reviewed:false AND existence:1 AND (keyword:KW-0091 OR go:0031214)",
    },
    "silk": {
        "prefix": "D",
        "queries": {
            "name": "reviewed:true AND (protein_name:spidroin OR protein_name:fibroin "
            "OR protein_name:silk)",
        },
        "trembl": "reviewed:false AND existence:1 AND (protein_name:spidroin OR "
        "protein_name:fibroin OR protein_name:silk)",
        # D3, the control lineage: bee, ant and hornet silks are small coiled-coil
        # proteins, a different answer to the same job. None is reviewed, so this search
        # takes evidence at protein or transcript level.
        "extra": {
            "d3_control": "(existence:1 OR existence:2) AND taxonomy_name:Hymenoptera "
            "AND (protein_name:silk OR protein_name:fibroin)",
        },
    },
}

# A name that means the entry is not the pattern the class is about. Each is printed.
EXCLUDE_NAME = {
    # Byssal collagens share ancestry with every other collagen.
    "adhesive": re.compile(r"collagen", re.IGNORECASE),
    # Sericin is the glue coat, not the fibre; the light chain and fibrohexamerin P25 are
    # small folded proteins that travel with the fibre.
    # Seroin is a 23 kDa glycoprotein that travels with the fibre, not the fibre.
    "silk": re.compile(
        r"sericin|seroin|fibroin light chain|light chain fibroin|P25|fibrohexamerin",
        re.IGNORECASE,
    ),
}

LINEAGES = {
    "adhesive": [
        ("B1 mussels and other bivalves", lambda t, n: "Bivalvia" in t),
        (
            "B2 tube-building polychaetes",
            lambda t, n: "Polychaeta" in t or "Annelida" in t,
        ),
        (
            "B3 barnacles (control)",
            lambda t, n: "Thecostraca" in t or "Cirripedia" in t,
        ),
        ("B4 other", lambda t, n: True),
    ],
    "shell": [
        ("C1 molluscs", lambda t, n: "Mollusca" in t),
        ("C2 stony corals", lambda t, n: "Scleractinia" in t or "Anthozoa" in t),
        ("C3 sea urchins and other echinoderms", lambda t, n: "Echinodermata" in t),
        ("C4 vertebrates", lambda t, n: "Vertebrata" in t),
        ("C5 other", lambda t, n: True),
    ],
    "silk": [
        ("D1 spiders", lambda t, n: "Araneae" in t),
        (
            "D2 moths, butterflies and caddisflies",
            lambda t, n: "Lepidoptera" in t or "Trichoptera" in t,
        ),
        ("D3 bees, ants and hornets (control)", lambda t, n: "Hymenoptera" in t),
        ("D4 other", lambda t, n: True),
    ],
}
# The lineages whose entries carry the class's shared pattern. A cross-lineage hit counts
# only between these; the control lineage and "other" are reported, never counted.
PATTERN_LINEAGES = {
    "adhesive": ["B1 mussels and other bivalves", "B2 tube-building polychaetes"],
    "shell": [
        "C1 molluscs",
        "C2 stony corals",
        "C3 sea urchins and other echinoderms",
        "C4 vertebrates",
    ],
    "silk": ["D1 spiders", "D2 moths, butterflies and caddisflies"],
}
# An entry kept only by the DOPA search needs at least this many DOPA sites. UniProt's
# search cannot count features, so this is applied after the fetch: one oxidised tyrosine
# is ordinary chemistry, and it let in chorion peroxidases and a tyrosinase inhibitor.
MIN_DOPA_SITES = 3
MIN_CROSS_TARGETS = 20
MIN_QUERY_LINEAGES = 2


def get(url: str, timeout: int = 180, tries: int = 4) -> tuple[str, dict]:
    """One GET, retried: a single SSL handshake timeout should not lose 600 cached calls."""
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode(), dict(r.headers)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == tries:
                raise
            wait = 2**attempt
            print(
                f"  {type(e).__name__} on {url[:70]}...; retry {attempt} in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch(url: str, path: Path, refetch: bool) -> tuple[str, str]:
    hdr = path.with_suffix(path.suffix + ".release")
    if refetch or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        text, headers = get(url)
        release = headers.get("X-UniProt-Release", "")
        path.write_text(text)
        hdr.write_text(release)
    return path.read_text(), hdr.read_text().strip()


def lineage_of(cls: str, row: dict) -> str:
    tax = row["Taxonomic lineage"] or ""
    name = row["Protein names"] or ""
    for label, test in LINEAGES[cls]:
        if test(tax, name):
            return label
    return LINEAGES[cls][-1][0]


def n_experimental(flat: str) -> dict[str, int]:
    out = {}
    for rec in flat.split("\n//\n"):
        m = re.search(r"^AC   (\w+);", rec, re.MULTILINE)
        if m:
            out[m.group(1)] = sum("ECO:0000269" in ln for ln in rec.splitlines())
    return out


def low_complexity_fraction(seq: str) -> float:
    """Share of residues inside a 12-residue window of entropy below SEG's trigger."""
    import math

    if not seq or len(seq) < SEG_WINDOW:
        return 0.0
    flagged = set()
    for i in range(len(seq) - SEG_WINDOW + 1):
        w = seq[i : i + SEG_WINDOW]
        counts = {}
        for aa in w:
            counts[aa] = counts.get(aa, 0) + 1
        h = -sum((c / SEG_WINDOW) * math.log2(c / SEG_WINDOW) for c in counts.values())
        if h < SEG_TRIGGER_BITS:
            flagged.update(range(i, i + SEG_WINDOW))
    return len(flagged) / len(seq)


def repeat_span(feature_text: str) -> int:
    """Residues covered by REPEAT features, from UniProt's feature text."""
    total = 0
    for a, b in re.findall(r"REPEAT (\d+)\.\.(\d+)", feature_text or ""):
        total += int(b) - int(a) + 1
    return total


def pfam_coverage(accessions: list[str], raw: Path) -> dict[str, float]:
    """Share of each sequence covered by a Pfam domain, from the InterPro API.

    --refetch does not reach here: these are hundreds of calls and the answer changes
    only when InterPro releases. Delete raw/interpro/ to fetch them again.
    """
    out = {}
    cache = raw / "interpro"
    cache.mkdir(parents=True, exist_ok=True)
    for i, acc in enumerate(accessions, 1):
        path = cache / f"{acc}.json"
        if not path.exists() or not path.read_text().strip():
            body = ""
            try:
                body, _ = get(f"{INTERPRO}/{acc}?page_size=200", timeout=120)
            except urllib.error.HTTPError as e:
                if e.code not in (204, 404):
                    raise
            # 204 means the entry has no Pfam match, and the body is empty either way.
            path.write_text(body.strip() or '{"results": []}')
            time.sleep(0.12)
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError:
            path.write_text('{"results": []}')
            parsed = {"results": []}
        covered, length = set(), None
        for res in parsed.get("results", []):
            for prot in res.get("proteins", []):
                length = prot.get("protein_length") or length
                for loc in prot.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        covered.update(range(frag["start"], frag["end"] + 1))
        out[acc] = len(covered) / length if length else 0.0
        if i % 50 == 0:
            print(f"  Pfam coverage: {i} of {len(accessions)}", file=sys.stderr)
    return out


def build(cls: str, outdir: Path, raw: Path, refetch: bool) -> dict:
    spec = CLASSES[cls]
    today = dt.date.today().isoformat()
    log, frames = [], []
    searches = {
        **spec["queries"],
        "trembl_pe1": spec["trembl"],
        **spec.get("extra", {}),
    }
    for name, q in searches.items():
        url = f"{STREAM}?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS, "format": "tsv"}
        )
        text, release = fetch(url, raw / f"{cls}_{name}.tsv", refetch)
        if release != RELEASE:
            sys.exit(f"{cls}/{name}: UniProt served {release!r}, expected {RELEASE}")
        t = pl.read_csv(
            raw / f"{cls}_{name}.tsv", separator="\t", infer_schema_length=0
        )
        log.append(
            {
                "class": cls,
                "query_name": name,
                "query": q,
                "url": url,
                "fetched_on": today,
                "uniprot_release": release,
                "n_rows": t.height,
            }
        )
        frames.append(
            t.with_columns(
                matched_by=pl.lit(name),
                tier=(
                    pl.when(pl.col("Reviewed") == "reviewed")
                    .then(pl.lit("Swiss-Prot"))
                    .otherwise(pl.lit("TrEMBL"))
                    if "Reviewed" in t.columns
                    else pl.lit(
                        "TrEMBL"
                        if name in ("trembl_pe1", "d3_control")
                        else "Swiss-Prot"
                    )
                ),
            )
        )
    allrows = pl.concat(frames)
    t = allrows.group_by(pl.exclude("matched_by")).agg(
        pl.col("matched_by").unique().sort().str.join("+")
    )

    dropped = []
    if cls in EXCLUDE_NAME:
        bad = t["Protein names"].str.contains("(?i)" + EXCLUDE_NAME[cls].pattern)
        dropped.append(
            t.filter(bad).with_columns(reason=pl.lit("name excluded by the class rule"))
        )
        t = t.filter(~bad)
    if cls == "adhesive":
        # UniProt writes the note as 3',4'-dihydroxyphenylalanine, with primes, so the
        # count matches the stem rather than the name the search uses.
        n_dopa = (
            pl.col("Modified residue")
            .fill_null("")
            .str.count_matches("dihydroxyphenylalanine")
        )
        thin = (pl.col("matched_by") == "dopa") & (n_dopa < MIN_DOPA_SITES)
        dropped.append(
            t.filter(thin).with_columns(
                reason=pl.format(
                    "kept only by the DOPA search, {} DOPA sites, fewer than "
                    + str(MIN_DOPA_SITES),
                    n_dopa,
                )
            )
        )
        t = t.filter(~thin)
    excluded = (
        pl.concat(dropped)
        if dropped
        else pl.DataFrame(schema={**t.schema, "reason": pl.Utf8})
    )

    t = t.with_columns(
        lineage=pl.Series([lineage_of(cls, r) for r in t.iter_rows(named=True)]),
        Length=pl.col("Length").cast(pl.Int64),
    )

    # The flat file comes in batches: one query naming 476 accessions is past the URL
    # length the API accepts and returns HTTP 400.
    acc = sorted(t["Entry"])
    chunks = []
    for i in range(0, len(acc), FLAT_BATCH):
        part = acc[i : i + FLAT_BATCH]
        url = f"{STREAM}?" + urllib.parse.urlencode(
            {"query": " OR ".join(f"accession:{a}" for a in part), "format": "txt"}
        )
        text, _ = fetch(url, raw / f"{cls}_entries_{i // FLAT_BATCH:03d}.txt", refetch)
        chunks.append(text)
    flat = "\n".join(chunks)
    t = t.with_columns(
        n_experimental_evidence=pl.col("Entry").replace_strict(
            n_experimental(flat), default=0
        )
    )

    if cls == "shell":
        cov = pfam_coverage(acc, raw)
        t = t.with_columns(
            pfam_coverage=pl.col("Entry").replace_strict(cov, default=0.0),
        ).with_columns(
            kind=pl.when(pl.col("pfam_coverage") >= 0.5)
            .then(pl.lit("folded domain"))
            .otherwise(pl.lit("acidic or low complexity"))
        )
    elif cls == "silk":
        t = t.with_columns(
            has_repeat_feature=pl.col("Repeat").is_not_null()
            & (pl.col("Repeat") != ""),
            repeat_span=pl.col("Repeat").map_elements(
                repeat_span, return_dtype=pl.Int64
            ),
            low_complexity_fraction=pl.col("Sequence").map_elements(
                low_complexity_fraction, return_dtype=pl.Float64
            ),
        ).with_columns(
            kind=pl.when(pl.col("has_repeat_feature"))
            .then(pl.lit("repeat feature"))
            .when(pl.col("low_complexity_fraction") >= LOW_COMPLEXITY_MIN)
            .then(pl.lit("low complexity"))
            .otherwise(pl.lit("neither"))
        )
    else:
        t = t.with_columns(kind=pl.lit("adhesive"))

    outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(log).write_csv(outdir / "truth_query_log.tsv", separator="\t")
    t.sort("lineage", "Entry").write_csv(outdir / "truth_set.tsv", separator="\t")
    if excluded.height:
        excluded.select("Entry", "Protein names", "Organism", "reason").write_csv(
            outdir / "dropped_entries.tsv", separator="\t"
        )
    return {"table": t, "excluded": excluded, "log": log}


def report(cls: str, built: dict) -> dict:
    t = built["table"]
    sp = t.filter(pl.col("tier") == "Swiss-Prot")
    pattern = PATTERN_LINEAGES[cls]
    counts = t.group_by("lineage", "tier", "kind").len().sort("lineage", "tier", "kind")
    print(f"\n=== class {CLASSES[cls]['prefix']} ({cls})")
    print(counts)
    if built["excluded"].height:
        print(f"dropped ({built['excluded'].height}), with the reason for each:")
        print(built["excluded"].select("Entry", "Protein names", "Organism", "reason"))

    # Entries that count as cross-lineage truth: Swiss-Prot tier, a pattern lineage, and
    # for class C only the acidic or low-complexity ones.
    usable = sp.filter(pl.col("lineage").is_in(pattern))
    if cls == "shell":
        usable = usable.filter(pl.col("kind") == "acidic or low complexity")
    if cls == "silk":
        usable = usable.filter(pl.col("kind") != "neither")
    with_query = [
        lin for lin in pattern if usable.filter(pl.col("lineage") == lin).height
    ]
    verdict = (
        "benchmark"
        if usable.height >= MIN_CROSS_TARGETS and len(with_query) >= MIN_QUERY_LINEAGES
        else "case study"
    )
    print(
        f"pattern-lineage entries in Swiss-Prot: {usable.height} "
        f"(needs {MIN_CROSS_TARGETS}); pattern lineages with an entry: {len(with_query)} "
        f"(needs {MIN_QUERY_LINEAGES}) -> {verdict}"
    )

    # One query per lineage: most experimental evidence, ties to the longer entry.
    pick_from = (
        sp
        if cls != "shell"
        else sp.filter(pl.col("kind") == "acidic or low complexity")
    )
    show = [
        "lineage",
        "Entry",
        "Protein names",
        "Length",
        "Organism",
        "n_experimental_evidence",
    ]
    if cls == "silk":
        # The brief picks the spider and lepidopteran query by the longest span covered by
        # REPEAT features; with no repeat feature anywhere in a lineage that rule cannot
        # choose, so the longest low-complexity span decides and the column shows which.
        pick_from = pick_from.filter(pl.col("kind") != "neither").with_columns(
            pattern_span=pl.max_horizontal(
                "repeat_span",
                (pl.col("low_complexity_fraction") * pl.col("Length")).cast(pl.Int64),
            )
        )
        order, desc = ["pattern_span", "n_experimental_evidence", "Entry"], [
            True,
            True,
            False,
        ]
        show += ["repeat_span", "low_complexity_fraction", "pattern_span"]
    else:
        order, desc = ["n_experimental_evidence", "Length", "Entry"], [
            True,
            True,
            False,
        ]
    queries = (
        pick_from.filter(~pl.col("lineage").str.ends_with("other"))
        .sort(order, descending=desc)
        .group_by("lineage", maintain_order=True)
        .first()
        .sort("lineage")
    )
    print(queries.select(show))
    return {"verdict": verdict, "n_usable": usable.height, "queries": queries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--classes", default="adhesive,shell,silk")
    args = ap.parse_args()
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(58)
    pl.Config.set_tbl_width_chars(200)

    summary = []
    for cls in args.classes.split(","):
        outdir = REPO / "data" / f"convergence_{cls}"
        built = build(cls, outdir, outdir / "raw", args.refetch)
        r = report(cls, built)
        summary.append(
            {
                "class": CLASSES[cls]["prefix"],
                "name": cls,
                "verdict": r["verdict"],
                "pattern_lineage_entries": r["n_usable"],
                "queries": r["queries"].height,
            }
        )
    print("\n=== Step 0 summary")
    print(pl.DataFrame(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
