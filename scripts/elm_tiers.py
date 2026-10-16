#!/usr/bin/env python3
"""ELM step 4: target-side labels in three tiers, kept in three separate tables.

For every (human ELM instance, species, 1:1 OMA ortholog) pair from Stage 0
(scripts/elm_stage0_flanks.py), the human motif is projected onto the ortholog through the
MAFFT alignment: the projected window is the span of ortholog residues aligned to the
human motif's residues (0-based, end-exclusive; empty when every motif column is a gap in
the ortholog).

regex_on_target.parquet  (its own step; one regex call per pair)
    regex_matches_target: the human instance's ELM class regex matches a stretch of the
    ortholog overlapping the projected window. One boolean per (instance, species) pair. It
    is used in exactly two places, kept apart: as a filter that stratifies results (the C1
    regex-fails stratum), and, in tier_regex_projected only, as a quarantined truth label.
    It is never truth anywhere else.

tier_experimental.parquet
    The ortholog carries its own ELM "true positive" instance of the same class. One row per
    (pair, ortholog instance), with whether that instance overlaps the projected window.
    seq_same_as_elm is whether the QfO 2020_04 ortholog sequence equals the current UniProt
    sequence ELM coordinates refer to; when it differs, a failed regex may be a version
    change, not a substitution, so those rows are excluded from the C1 stratum.

tier_swissprot.parquet
    The ortholog has a Swiss-Prot MOTIF or REGION feature overlapping the projected window.
    One row per feature: type, /note, and the /evidence codes (ECO) as written.

tier_regex_projected.parquet
    The pairs where regex_matches_target is true, renamed motif_regex_projected. For the
    cover check only. It is never a headline, and never joined into
    the other two tables, because the regex is a competitor and a regex-made truth cannot
    lose to the regex.

regex_fails_stratum.parquet (C1)
    tier_experimental rows whose ortholog instance overlaps the projected window, whose
    sequence equals the current UniProt sequence, and whose pair has regex_matches_target
    false: the experimental motif is there, and the class regex does not find it in the
    aligned window.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import polars as pl

ELM_DIR = Path("/Users/olga/data/elm-motif-transfer")
QFO = Path(
    "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
)
SPROT = Path("/Users/olga/data/uniprot/uniprot_sprot.dat.gz")
REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "nextflow-runs"
    / "qfo-pfam-region-benchmark"
    / "assets"
    / "qfo_species.tsv"
)


def read_fasta(path: Path) -> dict[str, str]:
    seqs, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            parts = line[1:].split()[0].split("|")
            acc, buf = (parts[1] if len(parts) >= 2 else parts[0]), []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def regex_hits(regex: str, seq: str, lo: int, hi: int) -> bool:
    """True if the regex matches a non-empty stretch of seq overlapping [lo, hi)."""
    if lo >= hi:
        return False
    pat = re.compile(regex)
    for i in range(max(0, lo - 60), min(len(seq), hi)):
        m = pat.match(seq, i)
        if m and m.end() > m.start() and m.end() > lo and m.start() < hi:
            return True
    return False


def projected_window(aln_path: Path, start: int, end: int) -> tuple[int, int] | None:
    recs = aln_path.read_text().split(">")[1:]
    ha, ta = ["".join(r.splitlines()[1:]).upper() for r in recs]
    hi = ti = 0
    tpos = []
    for a, b in zip(ha, ta):
        if a != "-" and start <= hi < end and b != "-":
            tpos.append(ti)
        hi += a != "-"
        ti += b != "-"
    return (min(tpos), max(tpos) + 1) if tpos else None


def swissprot_features(accs: set[str]) -> list[dict]:
    """MOTIF and REGION features, with /note and /evidence, for the given accessions."""
    rows, cur_accs, feat = [], [], None

    def flush():
        nonlocal feat
        if feat:
            rows.append(feat)
        feat = None

    with gzip.open(SPROT, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "AC":
                cur_accs += [a.strip() for a in line[5:].split(";") if a.strip()]
            elif tag == "FT" and set(cur_accs[:1]) & accs:
                body = line[5:].rstrip("\n")
                if body[:1] != " ":
                    flush()
                    ftype, loc = body[:16].strip(), body[16:].strip()
                    m = re.match(r"<?(\d+)\.\.>?(\d+)$", loc) or re.match(
                        r"(\d+)$", loc
                    )
                    if ftype in ("MOTIF", "REGION") and m:
                        s = int(m.group(1))
                        e = int(m.group(2)) if m.lastindex == 2 else s
                        feat = {
                            "accession": cur_accs[0],
                            "sp_type": ftype,
                            "sp_start": s - 1,
                            "sp_end": e,
                            "sp_note": "",
                            "sp_evidence": "",
                        }
                elif feat is not None:
                    q = body.strip()
                    if q.startswith("/note="):
                        feat["sp_note"] = q[6:].strip('"')
                    elif q.startswith("/evidence="):
                        feat["sp_evidence"] = q[10:].strip('"')
                    elif (
                        not q.startswith("/")
                        and feat["sp_note"]
                        and not feat["sp_evidence"]
                    ):
                        feat["sp_note"] += " " + q.strip('"')
            elif tag == "//":
                flush()
                cur_accs = []
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage0", type=Path, default=ELM_DIR / "stage0")
    ap.add_argument("--out-dir", type=Path, default=ELM_DIR / "tiers")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pairs = pl.read_parquet(args.stage0 / "stage0_pairs.parquet")
    inst = pl.read_parquet(args.stage0 / "stage0_instances.parquet").select(
        "elm_instance", "start", "end", "regex"
    )
    pairs = pairs.join(inst, on="elm_instance")
    elm = pl.read_parquet(ELM_DIR / "elm_instances_tp.parquet")
    uniprot_now = read_fasta(ELM_DIR / "elm_proteins.fasta")
    reg = pl.read_csv(REGISTRY, separator="\t", infer_schema_length=0)
    reg = {r["label"]: r for r in reg.iter_rows(named=True)}
    tseq: dict[str, str] = {}
    for sp in pairs["species"].unique():
        r = reg[sp]
        tseq.update(
            read_fasta(QFO / r["subdir"] / f"{r['proteome']}_{r['taxon']}.fasta")
        )

    proj = []
    for r in pairs.iter_rows(named=True):
        w = projected_window(
            args.stage0
            / "alignments"
            / r["species"]
            / f"{r['accession']}__{r['ortholog']}.fasta",
            r["start"],
            r["end"],
        )
        proj.append(
            {
                "elm_instance": r["elm_instance"],
                "species": r["species"],
                "proj_start": w[0] if w else None,
                "proj_end": w[1] if w else None,
            }
        )
    pairs = pairs.join(pl.DataFrame(proj), on=["elm_instance", "species"])
    base = [
        "elm_instance",
        "elm_class",
        "accession",
        "species",
        "mya",
        "ortholog",
        "motif_length",
        "length_bin",
        "in_midi_plus",
        "proj_start",
        "proj_end",
    ]

    # ---- regex on target: its own step, one call per pair
    rt = []
    for r in pairs.iter_rows(named=True):
        s = tseq.get(r["ortholog"], "")
        ok = r["proj_start"] is not None and regex_hits(
            r["regex"], s, r["proj_start"], r["proj_end"]
        )
        rt.append(
            {
                "elm_instance": r["elm_instance"],
                "species": r["species"],
                "regex_matches_target": ok,
            }
        )
    regex_on_target = pairs.select(*base).join(
        pl.DataFrame(rt), on=["elm_instance", "species"]
    )
    regex_on_target.write_parquet(out / "regex_on_target.parquet")
    rkey = regex_on_target.select("elm_instance", "species", "regex_matches_target")

    # ---- tier_experimental
    t_elm = elm.select(
        pl.col("accession").alias("ortholog"),
        "elm_class",
        pl.col("elm_instance").alias("t_elm_instance"),
        pl.col("start").alias("t_start"),
        pl.col("end").alias("t_end"),
    )
    exp = pairs.join(t_elm, on=["ortholog", "elm_class"], how="inner")
    rows = []
    for r in exp.iter_rows(named=True):
        s = tseq.get(r["ortholog"], "")
        in_range = r["t_end"] <= len(s)
        rows.append(
            {
                "elm_instance": r["elm_instance"],
                "species": r["species"],
                "t_elm_instance": r["t_elm_instance"],
                "overlaps_projection": r["proj_start"] is not None
                and r["t_start"] < r["proj_end"]
                and r["t_end"] > r["proj_start"],
                "seq_same_as_elm": in_range and s == uniprot_now.get(r["ortholog"]),
            }
        )
    exp = exp.select(*base, "t_elm_instance", "t_start", "t_end").join(
        pl.DataFrame(
            rows,
            schema={
                "elm_instance": pl.String,
                "species": pl.String,
                "t_elm_instance": pl.String,
                "overlaps_projection": pl.Boolean,
                "seq_same_as_elm": pl.Boolean,
            },
        ),
        on=["elm_instance", "species", "t_elm_instance"],
    )
    exp.write_parquet(out / "tier_experimental.parquet")

    # ---- tier_swissprot
    feats = pl.DataFrame(
        swissprot_features(set(pairs["ortholog"])),
        schema={
            "accession": pl.String,
            "sp_type": pl.String,
            "sp_start": pl.Int64,
            "sp_end": pl.Int64,
            "sp_note": pl.String,
            "sp_evidence": pl.String,
        },
    )
    sp = (
        pairs.select(*base)
        .join(feats.rename({"accession": "ortholog"}), on="ortholog", how="inner")
        .filter(
            pl.col("proj_start").is_not_null()
            & (pl.col("sp_start") < pl.col("proj_end"))
            & (pl.col("sp_end") > pl.col("proj_start"))
        )
    )
    sp.write_parquet(out / "tier_swissprot.parquet")

    # ---- tier_regex_projected: the quarantined truth use of regex_matches_target
    rpd = regex_on_target.filter("regex_matches_target").rename(
        {"regex_matches_target": "motif_regex_projected"}
    )
    rpd.write_parquet(out / "tier_regex_projected.parquet")

    # ---- C1 stratum
    # The stratifying use of regex_matches_target: a filter, joined on, not copied into exp.
    c1 = exp.join(rkey, on=["elm_instance", "species"]).filter(
        pl.col("overlaps_projection")
        & pl.col("seq_same_as_elm")
        & ~pl.col("regex_matches_target")
    )
    c1.write_parquet(out / "regex_fails_stratum.parquet")

    def n(df):
        return f"{df.height} rows, {df['elm_instance'].n_unique()} human instances"

    print(f"pairs (human instance x species with an OMA ortholog): {pairs.height}")
    print(
        f"tier_experimental: {n(exp)}; overlapping the projected window: "
        f"{n(exp.filter('overlaps_projection'))}"
    )
    print(
        exp.filter("overlaps_projection")
        .join(rkey, on=["elm_instance", "species"])
        .group_by("species")
        .agg(
            rows=pl.len(),
            human_instances=pl.col("elm_instance").n_unique(),
            seq_same_as_elm=pl.col("seq_same_as_elm").sum(),
            regex_matches_target=pl.col("regex_matches_target").sum(),
        )
        .sort("species")
    )
    print(f"tier_swissprot: {n(sp)}")
    print(sp.group_by("species", "sp_type").len().sort("species", "sp_type"))
    print(
        f"regex_on_target: regex_matches_target true in {regex_on_target['regex_matches_target'].sum()} "
        f"of {regex_on_target.height} pairs"
    )
    print(
        regex_on_target.group_by("species")
        .agg(pairs=pl.len(), regex_matches_target=pl.col("regex_matches_target").mean())
        .sort("species")
    )
    print(f"tier_regex_projected: {rpd.height} pairs")
    print(f"C1 regex-fails stratum: {n(c1)}")
    print(
        c1.select(
            "elm_class",
            "accession",
            "species",
            "ortholog",
            "t_start",
            "t_end",
            "length_bin",
        )
    )


if __name__ == "__main__":
    main()
