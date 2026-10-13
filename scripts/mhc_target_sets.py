#!/usr/bin/env python3
"""The MHC-system proteins of the Swiss-Prot reference Botryllus was searched against, and a
length-matched control set of the same size, for notebook 280.

    python3 mhc_target_sets.py <reference.fasta> <reference_pfam.parquet> <uniprot_sprot.dat.gz> <out_dir>

Three tiers of target, each protein in the first tier it qualifies for:

  mhc_domain        carries a Pfam MHC domain: PF00129 (class I alpha 1 and 2), PF00993 (class
                    II alpha), PF00969 (class II beta), PF06623 (MHC_I C-terminus), PF16497
                    (MHC-I family). Accessions and names read from InterPro on 2026-10-12.
  mhc_pathway       named as part of MHC antigen presentation or as an MHC-binding or
                    self/non-self receptor (the NAME_TERMS below, matched in the Swiss-Prot
                    RecName, case ignored).
  ig_c1_set         carries PF07654 (immunoglobulin C1-set), the fold of MHC alpha 3, beta 2
                    microglobulin and immunoglobulin constant domains; broad, kept apart.

The control set is the same number of reference proteins as the three tiers together, drawn
without replacement (seed 0) within each length decile of the reference so the lengths match,
from proteins in none of the tiers. Writes <out_dir>/mhc_targets.tsv and control_targets.tsv
(accession, tier, length, description) and all_targets.txt (one accession per line).

Run with the kmerseek image's python (polars, numpy).
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl

MHC_PFAM = {"PF00129", "PF00993", "PF00969", "PF06623", "PF16497"}
C1_SET = "PF07654"
NAME_TERMS = ["histocompatibility", "HLA class", "MHC class", "beta-2-microglobulin", "tapasin",
              "antigen peptide transporter", "TAP-binding", "invariant chain",
              "killer cell immunoglobulin-like receptor", "leukocyte immunoglobulin-like receptor",
              "NKG2", "natural killer cells antigen CD94", "T-cell surface glycoprotein CD4",
              "T-cell surface glycoprotein CD8", "T cell receptor", "T-cell receptor",
              "antigen-presenting glycoprotein CD1", "T-cell surface glycoprotein CD1",
              "MHC class I polypeptide-related", "UL16-binding", "endoplasmic reticulum aminopeptidase",
              "zinc-alpha-2-glycoprotein", "hereditary hemochromatosis", "IgG receptor FcRn"]


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


def sprot_names(path, wanted):
    out, accs, name = {}, [], None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("AC   "):
                accs += [a.strip() for a in line[5:].split(";") if a.strip()]
            elif line.startswith("DE   ") and name is None and "Full=" in line:
                name = line.split("Full=", 1)[1].split("{")[0].strip().rstrip(";")
            elif line.startswith("//"):
                for a in accs:
                    if a in wanted:
                        out[a] = name or ""
                accs, name = [], None
    return out


def main():
    ref_fasta, pfam_path, dat_path, out_dir = sys.argv[1:5]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lengths = read_lengths(ref_fasta)
    names = sprot_names(dat_path, set(lengths))
    pfam = pl.read_parquet(pfam_path)
    has = pfam.group_by("accession").agg(pl.col("pfam_id"))
    pf = {a: set(p) for a, p in has.iter_rows()}
    terms = [t.lower() for t in NAME_TERMS]
    rows = []
    for acc, n in lengths.items():
        desc = names.get(acc, "")
        p = pf.get(acc, set())
        if p & MHC_PFAM:
            tier = "mhc_domain"
        elif any(t in desc.lower() for t in terms):
            tier = "mhc_pathway"
        elif C1_SET in p:
            tier = "ig_c1_set"
        else:
            continue
        rows.append({"accession": acc, "tier": tier, "length": n, "description": desc})
    mhc = pl.DataFrame(rows)
    print(mhc.group_by("tier").len().sort("tier"))

    ref = pl.DataFrame({"accession": list(lengths), "length": list(lengths.values())})
    ref = ref.with_columns(pl.col("length").qcut(10, labels=[str(i) for i in range(10)]).alias("decile"))
    need = mhc.join(ref.select("accession", "decile"), on="accession").group_by("decile").len()
    pool = ref.filter(~pl.col("accession").is_in(mhc["accession"].to_list()))
    rng = np.random.default_rng(0)
    picks = []
    for decile, k in need.iter_rows():
        cand = pool.filter(pl.col("decile") == decile)["accession"].to_list()
        picks += list(rng.choice(cand, size=k, replace=False))
    control = ref.filter(pl.col("accession").is_in(picks)).select("accession", "length").with_columns(
        pl.lit("control").alias("tier"), pl.col("accession").replace_strict(names, default="").alias("description"))
    mhc.write_csv(out / "mhc_targets.tsv", separator="\t")
    control.select("accession", "tier", "length", "description").write_csv(out / "control_targets.tsv", separator="\t")
    (out / "all_targets.txt").write_text("\n".join(mhc["accession"].to_list() + control["accession"].to_list()) + "\n")
    print(f"mhc tiers {mhc.height}, control {control.height}; median length {mhc['length'].median()} vs {control['length'].median()}")


if __name__ == "__main__":
    main()
