"""Labels of kinds 1 to 3 for every set A exon of notebook 342 (Stop 2), no k-mer search.

For each exon, and the same for its context (the 20 residues before plus the 20 after, from
the including protein), writes one row of:

* kind 1, folded-domain overlap: hmmscan (HMMER 3.4, Pfam-A 38.2, --domE 0.001) on the
  including protein; fraction of residues inside a Pfam domain (envelope coordinates), the
  family covering most exon residues, and whether the exon starts or ends inside a domain.
* kind 2, short motifs and modification sites: ELM class regexes (ELM classes TSV, not
  instances) matched on the whole including protein, counting matches that overlap the exon
  and lie within 5 residues of it; UniProt MOD_RES sites of the gene's reviewed human entry,
  placed on the including protein by an exact match of the 11 residues around the site.
* kind 3, composition and pattern class: IUPred long mean (run on the whole protein), kappa,
  Omega_aro, NCPR and FCR (localCIDER 0.1.21), aro_hyd_chg_pol4 class fractions, SEG
  low-complexity fraction (segmasker, BLAST+ 2.16.0, defaults, run on the whole protein),
  and length.

    python scripts/342_exon_labels_kinds_1_3.py

Inputs: tables/342/342_setA_exon_table_patowary2024_grch37.tsv (Stop 1) and notebook 341's
event table for the proteins. Downloads and tool outputs are cached in ~/data/342_exon_function/;
the small ones the notebook reads are copied to data/342_exon_function/.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import condensate_exon_utils as cu  # noqa: E402

CACHE = Path.home() / "data" / "342_exon_function"
DATA = REPO / "data" / "342_exon_function"
TABLES = REPO / "tables" / "342"
EXON_TABLE = TABLES / "342_setA_exon_table_patowary2024_grch37.tsv"
EVENTS = REPO / "data" / "341_condensate_exons" / "341_setA_exon_events.parquet"
OUT = TABLES / "342_setA_exon_labels_kinds1to3.tsv"

HMMSCAN = Path("/Users/olga/anaconda3/envs/hmmer/bin/hmmscan")
SEGMASKER = Path("/Users/olga/anaconda3/envs/orthofinder/bin/segmasker")
PFAM_HMM = Path.home() / "data" / "pfam" / "Pfam-A.hmm"
DOM_E = "0.001"
ELM_URL = "http://elm.eu.org/elms/elms_index.tsv"
UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?query=(organism_id:9606)+AND+(reviewed:true)"
    "&fields=accession,gene_primary,gene_names,sequence,ft_mod_res&format=tsv"
)
#: Residues of flank on each side of the exon for the context value (rule in Stop 1).
FLANK = 20
#: ELM matches count if they lie within this many residues of the exon (prompt, Stop 2).
ELM_FLANK = 5
#: Residues on each side of a MOD_RES site used to place it on the including protein.
MOD_RES_HALF_WINDOW = 5


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path) -> Path:
    """Download once; later runs read the cached file."""
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as response, open(
            tmp, "wb"
        ) as fh:
            shutil.copyfileobj(response, fh)
            release = response.headers.get("X-UniProt-Release")
        tmp.rename(path)
        note = f"{url}\ndownloaded {date.today().isoformat()}\nmd5 {md5(path)}\n"
        if release:
            note += f"UniProt release {release}\n"
        path.with_suffix(path.suffix + ".provenance.txt").write_text(note)
    return path


def load_exons() -> pl.DataFrame:
    proteins = (
        pl.read_parquet(EVENTS)
        .select("including_isoform", "protein_including")
        .unique()
    )
    exons = pl.read_csv(EXON_TABLE, separator="\t").join(
        proteins, on="including_isoform", how="left"
    )
    assert exons["protein_including"].null_count() == 0
    return exons.with_columns(
        pl.format(
            "{}_{}_{}_{}", "gene_name", "including_isoform", "exon_start", "exon_end"
        ).alias("event_id")
    )


def write_fasta(exons: pl.DataFrame) -> Path:
    path = CACHE / "342_setA_including_proteins.fasta"
    rows = (
        exons.select("including_isoform", "protein_including")
        .unique()
        .sort("including_isoform")
    )
    path.write_text("".join(f">{i}\n{p}\n" for i, p in rows.iter_rows()))
    return path


def run_hmmscan(fasta: Path) -> pl.DataFrame:
    domtbl = CACHE / f"342_setA_including_proteins.pfam38.2.domE{DOM_E}.domtbl"
    if not domtbl.exists():
        subprocess.run(
            [
                HMMSCAN,
                "--cpu",
                "8",
                "--domE",
                DOM_E,
                "--noali",
                "-o",
                "/dev/null",
                "--domtblout",
                domtbl,
                PFAM_HMM,
                fasta,
            ],
            check=True,
        )
    with open(domtbl) as src, gzip.open(TABLES / (domtbl.name + ".gz"), "wt") as dst:
        dst.write(src.read())
    rows = []
    for line in open(domtbl):
        if line.startswith("#"):
            continue
        f = line.split()
        # domtblout: target name, accession, ..., query name (3), ..., i-Evalue (12), ...,
        # env from (19), env to (20)
        rows.append(
            {
                "family": f[0],
                "pfam_acc": f[1],
                "including_isoform": f[3],
                "i_evalue": float(f[12]),
                "env_from": int(f[19]),
                "env_to": int(f[20]),
            }
        )
    return pl.DataFrame(rows)


def run_segmasker(fasta: Path) -> dict[str, list[tuple[int, int]]]:
    out = CACHE / "342_setA_including_proteins.segmasker.interval"
    if not out.exists():
        subprocess.run(
            [SEGMASKER, "-in", fasta, "-outfmt", "interval", "-out", out], check=True
        )
    masked, current = {}, None
    for line in open(out):
        if line.startswith(">"):
            current = line[1:].split()[0]
            masked[current] = []
        elif line.strip():
            a, b = (int(x) for x in line.split(" - "))
            masked[current].append(
                (a + 1, b + 1)
            )  # 0-based inclusive -> 1-based inclusive
    return masked


def covered(intervals, n: int) -> np.ndarray:
    """Boolean mask of residues 1..n covered by any (start, end) interval, 1-based inclusive."""
    mask = np.zeros(n + 2, dtype=bool)
    for a, b in intervals:
        mask[a : b + 1] = True
    return mask


def load_elm() -> tuple[list[tuple[str, re.Pattern]], dict]:
    path = fetch(ELM_URL, CACHE / "elm_classes.tsv")
    header = {}
    for line in open(path):
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
    shutil.copy(path, DATA / "elm_classes.tsv")
    classes = pl.read_csv(path, separator="\t", comment_prefix="#", quote_char='"')
    patterns = [
        (name, re.compile(f"(?=({regex}))"))
        for name, regex in classes.select("ELMIdentifier", "Regex").iter_rows()
    ]
    assert len(patterns) == int(header["Num_Classes"]), (len(patterns), header)
    return patterns, {
        "elm_date": header["ELM_Classes_Download_Date"],
        "elm_md5": md5(path),
        "elm_classes": len(patterns),
    }


def load_mod_res() -> pl.DataFrame:
    path = fetch(UNIPROT_URL, CACHE / "uniprot_human_reviewed_mod_res.tsv")
    table = pl.read_csv(path, separator="\t", infer_schema_length=0)
    table.columns = ["accession", "gene_primary", "gene_names", "sequence", "mod_res"]
    return table


def mod_res_sites(
    entries: pl.DataFrame, gene: str, protein: str
) -> tuple[list[int] | None, int, int]:
    """Positions on `protein` of the MOD_RES sites of `gene`'s reviewed entries.

    Returns (positions or None when the gene has no reviewed entry, sites seen, sites placed).
    A site is placed when its 11-residue window occurs exactly once in the protein.
    """
    hits = entries.filter(pl.col("gene_primary") == gene)
    if hits.height == 0:
        hits = entries.filter(pl.col("gene_names").str.split(" ").list.contains(gene))
    if hits.height == 0:
        return None, 0, 0
    placed, seen = set(), 0
    for seq, feats in hits.select("sequence", "mod_res").iter_rows():
        for m in re.finditer(r"MOD_RES (\d+);", feats or ""):
            seen += 1
            p = int(m.group(1))
            a, b = max(0, p - 1 - MOD_RES_HALF_WINDOW), p + MOD_RES_HALF_WINDOW
            window = seq[a:b]
            starts = [
                s.start() for s in re.finditer(f"(?={re.escape(window)})", protein)
            ]
            if len(starts) == 1:
                placed.add(starts[0] + (p - 1 - a) + 1)
    return sorted(placed), seen, len(placed)


def localcider_charge(residues: str) -> dict[str, float]:
    from localcider.sequenceParameters import SequenceParameters

    sp = SequenceParameters(residues)
    return {"ncpr": float(sp.get_NCPR()), "fcr": float(sp.get_FCR())}


def kind3(residues: str) -> dict[str, float]:
    fractions = cu.class_fractions(residues)
    return {
        "length_aa": len(residues),
        "kappa": cu.kappa(residues),
        "omega_aro": cu.omega_aro(residues),
        **localcider_charge(residues),
        **{f"pol4_{k}": v for k, v in fractions.items()},
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    exons = load_exons()
    fasta = write_fasta(exons)
    print(
        f"{exons.height} exons, {exons['including_isoform'].n_unique()} including proteins, "
        f"FASTA md5 {md5(fasta)}",
        flush=True,
    )
    domains = run_hmmscan(fasta)
    print(
        f"hmmscan: {domains.height} Pfam domains at i-E <= {DOM_E} on "
        f"{domains['including_isoform'].n_unique()} proteins",
        flush=True,
    )
    masked = run_segmasker(fasta)
    elm_patterns, elm_info = load_elm()
    uniprot = load_mod_res()
    print(f"ELM {elm_info}; UniProt entries {uniprot.height}", flush=True)

    domains_by_protein = {
        k[0]: v
        for k, v in domains.partition_by("including_isoform", as_dict=True).items()
    }
    iupred_cache, elm_cache, mod_cache = {}, {}, {}
    rows = []
    for n, row in enumerate(exons.iter_rows(named=True)):
        iso, protein = row["including_isoform"], row["protein_including"]
        first, last, L = (
            row["exon_first_residue"],
            row["exon_last_residue"],
            len(protein),
        )
        exon_idx = np.arange(first, last + 1)
        before = np.arange(max(1, first - FLANK), first)
        after = np.arange(last + 1, min(L, last + FLANK) + 1)
        context_idx = np.concatenate([before, after])
        exon_res = protein[first - 1 : last]
        assert exon_res == row["exon_residues"], row["event_id"]
        context_res = "".join(protein[i - 1] for i in context_idx)
        out = {
            "event_id": row["event_id"],
            "gene_name": row["gene_name"],
            "outcome": row["outcome"],
            "half": row["half"],
            "exon_first_residue": first,
            "exon_last_residue": last,
        }

        # Kind 1.
        doms = domains_by_protein.get(iso)
        dom_intervals = (
            [] if doms is None else list(doms.select("env_from", "env_to").iter_rows())
        )
        in_domain = covered(dom_intervals, L)
        out["k1_domain_fraction_exon"] = float(in_domain[exon_idx].mean())
        out["k1_domain_fraction_context"] = (
            float(in_domain[context_idx].mean()) if context_idx.size else None
        )
        out["k1_starts_inside_domain"] = any(a < first <= b for a, b in dom_intervals)
        out["k1_ends_inside_domain"] = any(a <= last < b for a, b in dom_intervals)
        best_family, best_overlap = None, 0
        if doms is not None:
            for fam, a, b in doms.select("family", "env_from", "env_to").iter_rows():
                overlap = max(0, min(b, last) - max(a, first) + 1)
                if overlap > best_overlap:
                    best_family, best_overlap = fam, overlap
        out["k1_family"] = best_family

        # Kind 2: ELM regex matches on the whole protein, kept if they overlap the exon (exon
        # value) or a flank but not the exon (context value), within the windows.
        if iso not in elm_cache:
            elm_cache[iso] = [
                (name, m.start() + 1, m.start() + len(m.group(1)))
                for name, pat in elm_patterns
                for m in pat.finditer(protein)
                if m.group(1)
            ]
        lo, hi = first - ELM_FLANK, last + ELM_FLANK
        elm_exon = [
            (name, a, b)
            for name, a, b in elm_cache[iso]
            if a >= lo and b <= hi and b >= first and a <= last
        ]
        clo, chi = first - FLANK, last + FLANK
        elm_context = [
            (name, a, b)
            for name, a, b in elm_cache[iso]
            if a >= clo and b <= chi and (b < first or a > last)
        ]
        out["k2_elm_count_exon"] = len(elm_exon)
        out["k2_elm_classes_exon"] = ";".join(sorted({x[0] for x in elm_exon}))
        out["k2_elm_count_context"] = len(elm_context)
        key = (row["gene_name"], iso)
        if key not in mod_cache:
            mod_cache[key] = mod_res_sites(uniprot, row["gene_name"], protein)
        sites, seen, placed = mod_cache[key]
        out["k2_mod_res_gene_has_entry"] = sites is not None
        out["k2_mod_res_sites_seen"] = seen
        out["k2_mod_res_sites_placed"] = placed
        out["k2_mod_res_count_exon"] = (
            None if sites is None else sum(first <= s <= last for s in sites)
        )
        out["k2_mod_res_count_context"] = (
            None
            if sites is None
            else sum(int(s) in set(context_idx.tolist()) for s in sites)
        )

        # Kind 3.
        if iso not in iupred_cache:
            iupred_cache[iso] = np.array([0.0] + cu.iupred_long(protein))
        scores = iupred_cache[iso]
        lc = covered(masked.get(iso, []), L)
        out["k3_iupred_mean_exon"] = float(scores[exon_idx].mean())
        out["k3_iupred_mean_context"] = (
            float(scores[context_idx].mean()) if context_idx.size else None
        )
        out["k3_seg_fraction_exon"] = float(lc[exon_idx].mean())
        out["k3_seg_fraction_context"] = (
            float(lc[context_idx].mean()) if context_idx.size else None
        )
        out.update({f"k3_{k}_exon": v for k, v in kind3(exon_res).items()})
        out.update(
            {
                f"k3_{k}_context": v
                for k, v in (kind3(context_res).items() if context_res else [])
            }
        )
        rows.append(out)
        if n % 500 == 0:
            print(f"  {n} of {exons.height}", flush=True)

    labels = pl.DataFrame(rows, infer_schema_length=None)
    labels.write_csv(OUT, separator="\t")
    provenance = {
        "exon table md5": md5(EXON_TABLE),
        "including proteins FASTA md5": md5(fasta),
        "Pfam-A.hmm md5": md5(PFAM_HMM),
        "hmmscan": subprocess.run(
            [HMMSCAN, "-h"], capture_output=True, text=True
        ).stdout.splitlines()[1],
        "segmasker BLAST+": subprocess.run(
            [SEGMASKER.with_name("blastp"), "-version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        **elm_info,
        "UniProt": (CACHE / "uniprot_human_reviewed_mod_res.tsv.provenance.txt")
        .read_text()
        .replace("\n", " | "),
        "labels md5": md5(OUT),
    }
    (TABLES / "342_setA_exon_labels_kinds1to3.provenance.tsv").write_text(
        "".join(f"{k}\t{v}\n" for k, v in provenance.items())
    )
    print(f"{labels.height} rows -> {OUT}")
    for k, v in provenance.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
