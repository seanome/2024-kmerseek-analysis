#!/usr/bin/env python3
"""Notebook 359, step 2-5: exon table, blank set, decoys and kmerseek queries.

    build_exons.py --sqanti-dir <qc dir> --filter-dir <filter dir> --prefix <name>
                   --interproscan <tsv> --gencode-gtf <gtf.gz> --seed <int> --out-dir <dir>

Reads SQANTI3 v6.0.1's own outputs (classification with ORF, corrected GTF, the
IsoAnnotLite GFF3 and the filter's pass list) and InterProScan's Pfam TSV, and writes:

  exon_isoform.parquet   one row per (coding isoform surviving the filter, exon)
  blank_exons.parquet    one row per blank alternative exon, with its representative
                         isoform, protein segment, real and decoy protein
  gate.json              the sanity gate: InterProScan recall of IsoAnnotLite Pfam features
  build_summary.json     every count on the way to the blank set
  queries_real.fasta     representative isoforms holding a blank exon (real__<isoform>)
  queries_decoy.fasta    one protein per blank exon, only that exon shuffled (decoy__<exon_id>)

The rules are in notebooks/359_data/preregistered.md; the functions they use are in
notebooks/isoform_blank_exons_359.py. Coordinates in the tables are 0-based end-exclusive
for protein and transcript intervals and 1-based inclusive for genomic ones (as GTF).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import isoform_blank_exons_359 as ib  # noqa: E402

# Lines of IsoAnnotLite's GFF3 that describe the transcript itself, not a feature on it.
STRUCTURE_SOURCES = {"tappAS"}
# NMD is a flag on the whole transcript, not a region of it.
WHOLE_TRANSCRIPT_SOURCES = {"NMD"}


def opener(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def gtf_attr(attrs: str, key: str) -> str | None:
    m = re.search(key + r' "([^"]*)"', attrs)
    return m.group(1) if m else None


def read_exons(gtf: Path, keep: set[str] | None = None) -> dict[str, dict]:
    """transcript_id -> {chrom, strand, exons: [Exon]} from a GTF's exon lines."""
    out: dict[str, dict] = {}
    with opener(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            tid = gtf_attr(f[8], "transcript_id")
            if keep is not None and tid not in keep:
                continue
            rec = out.setdefault(tid, {"chrom": f[0], "strand": f[6], "exons": []})
            rec["exons"].append(ib.Exon(int(f[3]), int(f[4])))
    for rec in out.values():
        rec["exons"].sort(key=lambda e: e.start)
    return out


def ccds_basic_exons(gencode_gtf: Path) -> set[tuple]:
    """(chrom, strand, start, end) of every exon of a GENCODE transcript tagged both
    "basic" and "CCDS"."""
    out = set()
    with opener(gencode_gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "exon":
                continue
            tags = set(re.findall(r'tag "([^"]+)"', f[8]))
            if "basic" in tags and "CCDS" in tags:
                out.add((f[0], f[6], int(f[3]), int(f[4])))
    return out


def read_isoannot(gff3: Path, keep: set[str]) -> pl.DataFrame:
    """Positional features of IsoAnnotLite's GFF3 on the kept transcripts."""
    rows = []
    with open(gff3) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9 or f[0] not in keep:
                continue
            m = re.search(r"PosType=(\w)", f[8])
            pos_type = m.group(1) if m else ""
            idm = re.search(r"ID=([^;]*)", f[8])
            rows.append(
                {
                    "isoform": f[0],
                    "source": f[1],
                    "feature": f[2],
                    "start": int(f[3]) if f[3] not in (".", "") else None,
                    "end": int(f[4]) if f[4] not in (".", "") else None,
                    "pos_type": pos_type,
                    "feature_id": idm.group(1).strip() if idm else "",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "isoform": pl.String,
            "source": pl.String,
            "feature": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "pos_type": pl.String,
            "feature_id": pl.String,
        },
    )


def read_pfam(tsv: Path) -> pl.DataFrame:
    """InterProScan 5 TSV, Pfam rows: protein, accession, 0-based end-exclusive interval."""
    rows = []
    with open(tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[3] != "Pfam":
                continue
            rows.append({"isoform": f[0], "pfam": f[4], "start": int(f[6]) - 1, "end": int(f[7])})
    return pl.DataFrame(rows, schema={"isoform": pl.String, "pfam": pl.String, "start": pl.Int64, "end": pl.Int64})


def main(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc = Path(a.sqanti_dir)
    summary: dict = {"seed": a.seed}

    passed = {l.strip() for l in open(Path(a.filter_dir) / f"{a.prefix}_pass_isoforms.txt") if l.strip()}
    cls = pl.read_csv(qc / f"{a.prefix}_classification.txt", separator="\t", infer_schema_length=0)
    summary["isoforms_qc"] = cls.height
    summary["isoforms_pass_filter"] = len(passed)
    cls = cls.filter(pl.col("isoform").is_in(list(passed)))
    tx = read_exons(qc / f"{a.prefix}_corrected.gtf", keep=passed)
    gene_of = dict(zip(cls["isoform"], cls["associated_gene"]))

    # ---- coding isoforms and their ORFs -----------------------------------------------
    coding = cls.filter((pl.col("coding") == "coding") & ~pl.col("ORF_seq").fill_null("").is_in(["", "NA"]))
    summary["isoforms_pass_coding"] = coding.height
    proteins, cds_start, bad_frame = {}, {}, 0
    for r in coding.iter_rows(named=True):
        prot = ib.usable_orf(r["ORF_seq"], r["CDS_start"], r["CDS_end"])
        if prot is None:
            bad_frame += 1
            continue
        proteins[r["isoform"]] = prot
        cds_start[r["isoform"]] = int(float(r["CDS_start"]))
    summary["isoforms_dropped_orf_length_mismatch"] = bad_frame
    summary["isoforms_coding_used"] = len(proteins)

    # ---- one row per (coding isoform, exon) -------------------------------------------
    rows = []
    for iso, prot in proteins.items():
        t = tx[iso]
        segs = ib.exon_protein_intervals(t["exons"], t["strand"], cds_start[iso], len(prot))
        tints = ib.exon_transcript_intervals(t["exons"], t["strand"])
        for k, (e, seg, ti) in enumerate(zip(ib.transcript_order(t["exons"], t["strand"]), segs, tints)):
            rows.append(
                {
                    "isoform": iso,
                    "gene": gene_of[iso],
                    "chrom": t["chrom"],
                    "strand": t["strand"],
                    "exon_index": k,
                    "n_exons": len(t["exons"]),
                    "g_start": e.start,
                    "g_end": e.end,
                    "t_start": ti[0],
                    "t_end": ti[1],
                    "p_start": seg[0] if seg else None,
                    "p_end": seg[1] if seg else None,
                    "protein_length": len(prot),
                }
            )
    ex = pl.DataFrame(rows).filter(pl.col("p_start").is_not_null())

    # ---- alternative: a cassette exon some spanning surviving isoform of the gene skips --
    by_gene = collections.defaultdict(list)
    for iso in passed:
        if iso in tx and iso in gene_of:
            by_gene[gene_of[iso]].append(iso)
    span = {iso: (tx[iso]["exons"][0].start, tx[iso]["exons"][-1].end) for iso in passed if iso in tx}
    alt = []
    for r in ex.select("gene", "chrom", "strand", "g_start", "g_end").unique().iter_rows(named=True):
        skipped_by = []
        for other in by_gene[r["gene"]]:
            o = tx[other]
            if o["chrom"] != r["chrom"] or o["strand"] != r["strand"]:
                continue
            lo, hi = span[other]
            if not (lo < r["g_start"] and hi > r["g_end"]):
                continue
            if any(e.start <= r["g_end"] and e.end >= r["g_start"] for e in o["exons"]):
                continue
            skipped_by.append(other)
        alt.append({**r, "alternative": bool(skipped_by), "n_skipping_isoforms": len(skipped_by)})
    ex = ex.join(pl.DataFrame(alt), on=["gene", "chrom", "strand", "g_start", "g_end"], how="left")

    # ---- IsoAnnotLite features and Pfam hits overlapping each exon's protein segment ----
    feats = read_isoannot(qc / f"{a.prefix}.gff3", keep=set(proteins))
    summary["isoannot_feature_lines_by_source"] = {
        f"{s}/{p}": n for s, p, n in feats.group_by("source", "pos_type").len().sort("source").iter_rows()
    }
    positional = feats.filter(
        ~pl.col("source").is_in(list(STRUCTURE_SOURCES | WHOLE_TRANSCRIPT_SOURCES))
        & pl.col("pos_type").is_in(["P", "T", "G"])
        & pl.col("start").is_not_null()
    )
    feat_by_iso = collections.defaultdict(list)
    for r in positional.iter_rows(named=True):
        feat_by_iso[r["isoform"]].append(r)
    pfam = read_pfam(Path(a.interproscan))
    pfam_by_iso = collections.defaultdict(list)
    for r in pfam.iter_rows(named=True):
        pfam_by_iso[r["isoform"]].append(r)

    hit_cols = []
    for r in ex.iter_rows(named=True):
        iso, seg = r["isoform"], (r["p_start"], r["p_end"])
        t = tx[iso]
        tseg = ib.protein_to_transcript(seg, cds_start[iso])
        gpieces = ib.protein_to_genomic(seg, t["exons"], t["strand"], cds_start[iso])
        srcs = []
        for f in feat_by_iso.get(iso, []):
            if f["pos_type"] == "P":
                hit = ib.overlap((f["start"] - 1, f["end"]), seg) > 0
            elif f["pos_type"] == "T":
                hit = ib.overlap((f["start"] - 1, f["end"]), tseg) > 0
            else:
                hit = any(f["start"] <= b and f["end"] >= a_ for a_, b in gpieces)
            if hit:
                srcs.append(f"{f['source']}:{f['feature']}")
        pf = sorted({p["pfam"] for p in pfam_by_iso.get(iso, []) if ib.overlap((p["start"], p["end"]), seg) > 0})
        hit_cols.append({"isoannot_features": ";".join(sorted(set(srcs))), "pfam_hits": ";".join(pf)})
    ex = pl.concat([ex, pl.DataFrame(hit_cols)], how="horizontal")

    ccds = ccds_basic_exons(Path(a.gencode_gtf))
    ex = ex.with_columns(
        pl.struct("chrom", "strand", "g_start", "g_end")
        .map_elements(lambda s: (s["chrom"], s["strand"], s["g_start"], s["g_end"]) in ccds, return_dtype=pl.Boolean)
        .alias("in_basic_ccds_isoform")
    )
    ex.write_parquet(out / "exon_isoform.parquet")
    summary["exon_rows"] = ex.height

    # ---- one unit per genomic exon: blank if no isoform holding it has a feature or Pfam --
    key = ["gene", "chrom", "strand", "g_start", "g_end"]
    units = (
        ex.filter(pl.col("alternative"))
        .group_by(key)
        .agg(
            pl.col("isoform").sort().first().alias("rep_isoform"),
            pl.col("isoform").n_unique().alias("n_coding_isoforms_holding"),
            (pl.col("isoannot_features") != "").any().alias("any_isoannot"),
            (pl.col("pfam_hits") != "").any().alias("any_pfam"),
            pl.col("in_basic_ccds_isoform").first(),
        )
        .sort(key)
    )
    summary["alternative_exons"] = units.height
    summary["alternative_exons_with_isoannot"] = int(units["any_isoannot"].sum())
    summary["alternative_exons_with_pfam"] = int(units["any_pfam"].sum())
    blank = units.filter(~pl.col("any_isoannot") & ~pl.col("any_pfam"))
    rep_seg = {
        (r["isoform"], r["chrom"], r["strand"], r["g_start"], r["g_end"]): (r["p_start"], r["p_end"])
        for r in ex.iter_rows(named=True)
    }
    rng = random.Random(a.seed)
    brows = []
    for i, r in enumerate(blank.iter_rows(named=True)):
        seg = rep_seg[(r["rep_isoform"], r["chrom"], r["strand"], r["g_start"], r["g_end"])]
        prot = proteins[r["rep_isoform"]]
        decoy = ib.shuffle_segment(prot, seg, rng)
        brows.append(
            {
                "exon_id": f"E{i + 1:06d}",
                **r,
                "p_start": seg[0],
                "p_end": seg[1],
                "segment_length": seg[1] - seg[0],
                "segment": prot[seg[0] : seg[1]],
                "decoy_segment": decoy[seg[0] : seg[1]],
                "protein": prot,
                "decoy_protein": decoy,
            }
        )
    bl = pl.DataFrame(brows)
    bl.write_parquet(out / "blank_exons.parquet")
    summary["blank_exons"] = bl.height
    summary["blank_genes"] = bl["gene"].n_unique() if bl.height else 0
    summary["blank_exons_in_basic_ccds_isoform"] = int(bl["in_basic_ccds_isoform"].sum()) if bl.height else 0
    summary["decoy_segments_identical_to_real"] = (
        int((bl["segment"] == bl["decoy_segment"]).sum()) if bl.height else 0
    )

    with open(out / "queries_real.fasta", "w") as fh:
        for iso in sorted(set(bl["rep_isoform"])) if bl.height else []:
            fh.write(f">real__{iso}\n{proteins[iso]}\n")
    with open(out / "queries_decoy.fasta", "w") as fh:
        for r in bl.iter_rows(named=True):
            fh.write(f">decoy__{r['exon_id']}\n{r['decoy_protein']}\n")
    summary["queries_real"] = len(set(bl["rep_isoform"])) if bl.height else 0
    summary["queries_decoy"] = bl.height

    # ---- sanity gate: InterProScan recall of IsoAnnotLite's Pfam features ---------------
    ia = feats.filter((pl.col("source") == "PFAM") & (pl.col("feature") == "DOMAIN") & (pl.col("pos_type") == "P"))
    rec_any = rec_same = 0
    for r in ia.iter_rows(named=True):
        seg = (r["start"] - 1, r["end"])
        need = 0.5 * (seg[1] - seg[0])
        hits = [p for p in pfam_by_iso.get(r["isoform"], []) if ib.overlap((p["start"], p["end"]), seg) >= need]
        rec_any += bool(hits)
        acc = r["feature_id"].split(".")[0]
        rec_same += any(p["pfam"].split(".")[0] == acc for p in hits)
    n = ia.height
    gate = {
        "isoannot_pfam_features": n,
        "recovered_by_overlap": rec_any,
        "recall_by_overlap": rec_any / n if n else None,
        "recovered_same_accession": rec_same,
        "recall_same_accession": rec_same / n if n else None,
        "rule": "InterProScan Pfam hit on the same isoform covering >= 50% of the IsoAnnotLite feature",
        "passes": (rec_any / n >= 0.9) if n else False,
    }
    json.dump(gate, open(out / "gate.json", "w"), indent=2)
    json.dump(summary, open(out / "build_summary.json", "w"), indent=2)
    print(json.dumps({**summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sqanti-dir", required=True)
    p.add_argument("--filter-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--interproscan", required=True)
    p.add_argument("--gencode-gtf", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
