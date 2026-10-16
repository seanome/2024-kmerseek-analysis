#!/usr/bin/env python3
"""The MHC and immune-recognition proteins of the Swiss-Prot reference Botryllus was searched
against, sorted by HGNC gene group, and a length-matched control set of the same size, for
notebook 281.

    python3 mhc_target_sets.py <reference.fasta> <reference_pfam.parquet> <uniprot_sprot.dat.gz> <hgnc_dir> <self_nonself_dir> <out_dir>

Categories come from HGNC gene groups (the groups GeneCards shows; downloaded from
genenames.org into data/hgnc/ on the date in data/hgnc/DOWNLOADED_ON). A Swiss-Prot entry is
matched by its gene name (GN Name=, upper-cased) against the approved symbols of each group,
so a mouse or fish orthologue that keeps the human symbol is matched too. Mouse and rat MHC
genes are named H2-* and RT1-*, not HLA-*, so the MHC category also takes any entry carrying a
Pfam MHC domain: PF00129, PF00993, PF00969, PF06623, PF16497 (read from InterPro, 2026-10-12).
Each entry goes to the first category it qualifies for, in this order:

  mhc                      HGNC 588 Histocompatibility complex, or a Pfam MHC domain
  mhc_like_nkg2d_ligand    HGNC 4012 MIC family, 4010 RAET1 (ULBP) family
  minor_histocompatibility HGNC 870 Minor histocompatibility antigens
  antigen_presentation     TAP1, TAP2, TAPBP, B2M, CD74, ERAP1, ERAP2 by symbol; HGNC has no
                           antigen-presentation group, and B2M and TAPBP sit only in C1-set
  nk_receptor              HGNC 620 KIR, 1181 and 1182 LILR, 621 killer cell lectin-like
  self_nonself_nonvertebrate  reviewed UniProt entries mentioning "allorecognition" or
                           "heterokaryon incompatibility", or with keyword KW-0713
                           Self-incompatibility (data/uniprot_self_nonself/, downloaded on the
                           date in DOWNLOADED_ON): fungal het and rcd proteins, Dictyostelium
                           TgrB1/TgrC1, plant S-locus proteins. Found by text, so a few entries
                           (ATG8, a mating-type protein) take part rather than recognise. HGNC
                           covers human genes only, so these systems are not in its groups.
  tcr_cd3                  HGNC 371, 372, 374, 375 T cell receptor loci; 3733 CD3
  c1_set                   HGNC 591 C1-set domain containing: the fold of MHC alpha 3 and
                           immunoglobulin constant domains, broad, kept last

The control set is the same number of reference proteins as all categories together, drawn
without replacement (seed 0) within each length decile so lengths match, from entries in no
category. Writes <out_dir>/mhc_targets.tsv and control_targets.tsv (accession, category,
gene, length, description) and all_targets.txt. Run with the kmerseek image's python.
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl

MHC_PFAM = {"PF00129", "PF00993", "PF00969", "PF06623", "PF16497"}
CATEGORIES = [
    ("mhc", [588]),
    ("mhc_like_nkg2d_ligand", [4012, 4010]),
    ("minor_histocompatibility", [870]),
    ("antigen_presentation", []),
    ("nk_receptor", [620, 1181, 1182, 621]),
    ("self_nonself_nonvertebrate", []),
    ("tcr_cd3", [371, 372, 374, 375, 3733]),
    ("c1_set", [591]),
]
ANTIGEN_PRESENTATION = {"TAP1", "TAP2", "TAPBP", "B2M", "CD74", "ERAP1", "ERAP2"}


def read_lengths(path):
    lengths, name, n = {}, None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def sprot_names_genes(path, wanted):
    """accession -> (RecName, GN Name) for the wanted accessions."""
    out, accs, name, gene = {}, [], None, ""
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("AC   "):
                accs += [a.strip() for a in line[5:].split(";") if a.strip()]
            elif line.startswith("DE   ") and name is None and "Full=" in line:
                name = line.split("Full=", 1)[1].split("{")[0].strip().rstrip(";")
            elif line.startswith("GN   ") and not gene and "Name=" in line:
                gene = line.split("Name=", 1)[1].split(";")[0].split("{")[0].strip()
            elif line.startswith("//"):
                for a in accs:
                    if a in wanted:
                        out[a] = (name or "", gene)
                accs, name, gene = [], None, ""
    return out


def main():
    ref_fasta, pfam_path, dat_path, hgnc_dir, sns_dir, out_dir = sys.argv[1:7]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = {}
    for cat, ids in CATEGORIES:
        s = set()
        for gid in ids:
            g = pl.read_csv(Path(hgnc_dir) / f"hgnc_group_{gid}.tsv", separator="\t", infer_schema_length=0)
            s |= set(g["Approved symbol"].str.to_uppercase().to_list())
        symbols[cat] = s
    symbols["antigen_presentation"] = ANTIGEN_PRESENTATION
    symbols["self_nonself_nonvertebrate"] = set()
    sns_acc = set()
    for f in sorted(Path(sns_dir).glob("*.tsv")):
        sns_acc |= set(pl.read_csv(f, separator="\t", infer_schema_length=0)["Entry"].to_list())

    lengths = read_lengths(ref_fasta)
    info = sprot_names_genes(dat_path, set(lengths))
    pf = {a: set(p) for a, p in pl.read_parquet(pfam_path).group_by("accession").agg(pl.col("pfam_id")).iter_rows()}
    rows = []
    for acc, n in lengths.items():
        desc, gene = info.get(acc, ("", ""))
        g = gene.upper()
        for cat, _ in CATEGORIES:
            if (g in symbols[cat] or (cat == "mhc" and pf.get(acc, set()) & MHC_PFAM)
                    or (cat == "self_nonself_nonvertebrate" and acc in sns_acc)):
                rows.append({"accession": acc, "category": cat, "gene": gene, "length": n, "description": desc})
                break
    mhc = pl.DataFrame(rows)
    print(mhc.group_by("category").len().sort("category"))

    ref = pl.DataFrame({"accession": list(lengths), "length": list(lengths.values())})
    ref = ref.with_columns(pl.col("length").qcut(10, labels=[str(i) for i in range(10)]).alias("decile"))
    need = mhc.join(ref.select("accession", "decile"), on="accession").group_by("decile").len()
    pool = ref.filter(~pl.col("accession").is_in(mhc["accession"].to_list()))
    rng = np.random.default_rng(0)
    picks = []
    for decile, k in need.sort("decile").iter_rows():
        cand = sorted(pool.filter(pl.col("decile") == decile)["accession"].to_list())
        picks += list(rng.choice(cand, size=k, replace=False))
    control = ref.filter(pl.col("accession").is_in(picks)).select("accession", "length").with_columns(
        pl.lit("control").alias("category"),
        pl.col("accession").map_elements(lambda a: info.get(a, ("", ""))[1], return_dtype=pl.String).alias("gene"),
        pl.col("accession").map_elements(lambda a: info.get(a, ("", ""))[0], return_dtype=pl.String).alias("description"))
    mhc.write_csv(out / "mhc_targets.tsv", separator="\t")
    control.select("accession", "category", "gene", "length", "description").write_csv(out / "control_targets.tsv", separator="\t")
    (out / "all_targets.txt").write_text("\n".join(mhc["accession"].to_list() + control["accession"].to_list()) + "\n")
    print(f"categories {mhc.height}, control {control.height}; median length {mhc['length'].median()} vs {control['length'].median()}")


if __name__ == "__main__":
    main()
