#!/usr/bin/env python3
"""A second truth set from Swiss-Prot curated features -- neither HMM- nor structure-derived.

Why this exists. Pfam-A domains are DEFINED by profile HMMs, and phmmer/jackhmmer/hhblits
are profile methods, so scoring them against Pfam scores a model class against its own
output. Worse for the hypothesis under test: a region Pfam never annotated is, by
construction, labelled absent, so every correct cryptic-domain rescue is counted as a false
positive. The benchmark punishes the claim it exists to test. That is a measurement error
pointing the wrong way, not conservatism.

Swiss-Prot FT features are literature-curated, residue-ranged, and defined by function
rather than by any sequence model or fold. They are circular with neither the profile
baselines nor the structure baselines, which is what makes them usable for the
invertebrate arm where predicted structures are weakest.

Emits build_domain_truth.py's schema -- accession, pfam_id, domain_start, domain_end,
protein_length -- plus `is_point` and `anchor_pfam`, so the whole existing scoring path
runs against it unchanged. The `pfam_id` column carries the feature type (ACT_SITE,
TRANSMEM, ...) rather than a Pfam accession; the column name is kept for schema
compatibility and the meaning is recorded in the summary.

`anchor_pfam` is the reachability key, and it is why this truth set needs a column the
Pfam truth set does not. Reachability asks whether a human annotation could have been
transferred from this target proteome at all, so that a tool is not charged for missing
something the target does not contain. Under the Pfam truth set the answer key IS the
transfer key: `pfam_id` is a family, and "the target has family F" is a real statement
that varies by species. Here `pfam_id` is one of twelve feature types, and "this proteome
contains at least one TRANSMEM somewhere" is true of every proteome. Joining on it
returned one identical constant for eight of the nine targets, and for ciona that constant
minus the human INTRAMEM features, which is the only type ciona lacks. It measured nothing.

So each row also carries the Pfam families of its OWN protein, on both sides, and
reachable_truth() below joins on the pair. See its docstring for the rule.

Point vs range features behave differently and both are kept, flagged by `is_point`:
  range  DNA_BIND, TRANSMEM, REGION, MOTIF, COILED, ZN_FING, REPEAT, DOMAIN
         -- boundary evaluation, directly comparable to Pfam envelopes
  point  ACT_SITE, BINDING, SITE
         -- catalytic/binding residue recall. A 1-residue truth interval cannot be scored
            by boundary IoU in any meaningful way, so downstream work should filter on
            is_point rather than silently mixing the two.
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import polars as pl

RANGE_FEATURES = {
    "DNA_BIND", "TRANSMEM", "REGION", "MOTIF", "COILED", "ZN_FING", "REPEAT", "DOMAIN",
    "INTRAMEM", "CA_BIND", "NP_BIND",
}
POINT_FEATURES = {"ACT_SITE", "BINDING", "SITE", "METAL"}
DEFAULT_FEATURES = sorted(RANGE_FEATURES | POINT_FEATURES)

# Modern Swiss-Prot FT lines: "FT   TRANSMEM        20..40" or "FT   ACT_SITE        123".
# Positions may carry fuzzy markers (<, >, ?) which are dropped along with the feature,
# since an uncertain boundary cannot serve as boundary truth.
FT_RE = re.compile(r"^FT\s{3}(\w+)\s+([<>?]?\d+)(?:\.\.([<>?]?\d+))?\s*$")


def parse(dat_path: Path, wanted_types: set[str], wanted_acc: set[str] | None):
    acc = None
    seq_len = None
    rows = []
    opener = gzip.open if dat_path.suffix == ".gz" else open
    with opener(dat_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("AC "):
                if acc is None:
                    acc = line[5:].split(";")[0].strip()
            elif line.startswith("ID "):
                acc, seq_len = None, None
                parts = line.split()
                if len(parts) >= 4 and parts[-1] == "AA.":
                    try:
                        seq_len = int(parts[-2])
                    except ValueError:
                        seq_len = None
            elif line.startswith("FT "):
                if acc is None or (wanted_acc is not None and acc not in wanted_acc):
                    continue
                m = FT_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                ftype, start_s, end_s = m.groups()
                if ftype not in wanted_types:
                    continue
                # Fuzzy endpoints are unusable as boundary truth; drop rather than guess.
                if any(c in start_s for c in "<>?") or (end_s and any(c in end_s for c in "<>?")):
                    continue
                start = int(start_s)
                end = int(end_s) if end_s else start
                rows.append((acc, ftype, start, end, seq_len, end_s is None))
            elif line.startswith("//"):
                acc, seq_len = None, None
    return rows


ANCHOR_COL = "anchor_pfam"


def anchor_pairs(target: pl.DataFrame) -> pl.DataFrame:
    """The distinct (feature type, Pfam family) pairs this proteome's proteins hold.

    Split out so a caller scoring thousands of strata against one proteome pays the explode
    once instead of once per stratum.
    """
    return (
        target.select("pfam_id", ANCHOR_COL)
        .explode(ANCHOR_COL)
        .drop_nulls(ANCHOR_COL)
        .unique()
    )


def reachable_truth(truth: pl.DataFrame, target: pl.DataFrame | None = None,
                    pairs: pl.DataFrame | None = None) -> pl.DataFrame:
    """The human feature instances this target proteome could actually have supplied.

    A sequence-search arm produces a call by aligning a human region to a target interval
    and copying that interval's label. So a human annotation is reachable from a target
    proteome when the proteome holds a single annotated protein that supplies both halves
    of that move: it carries the same feature type, and it is homologous to the human
    query. Shared Pfam family membership is the homology proxy, taken from the same
    annotation tables the rest of the benchmark is built on.

    Under the Pfam truth set both halves collapse into one, because `pfam_id` IS the
    family there -- which is why the Pfam and Pfam-N arms keep the plain family join and
    are untouched by this. Here `pfam_id` is one of twelve Swiss-Prot feature types, the
    family half is the only half that varies, and dropping it is what made the denominator
    a constant.

    Eager, and the row index is taken once: a lazy with_row_index feeding a join does not
    survive re-execution of the plan with the same numbering.
    """
    if pairs is None:
        pairs = anchor_pairs(target)
    idx = truth.with_row_index("_reach_i")
    hit = (
        idx.select("_reach_i", "pfam_id", ANCHOR_COL)
        .explode(ANCHOR_COL)
        .join(pairs, on=["pfam_id", ANCHOR_COL], how="semi")
        .select("_reach_i")
        .unique()
    )
    return idx.join(hit, on="_reach_i", how="semi").drop("_reach_i")


def reachable_count(truth: pl.DataFrame, target: pl.DataFrame) -> int:
    return reachable_truth(truth, target).height


def add_anchor(features: pl.DataFrame, anchor: pl.DataFrame, label: str) -> pl.DataFrame:
    """Attach each protein's own Pfam families to its Swiss-Prot features.

    The join is total by construction -- `features` was already restricted to accessions
    that appear in this species' Pfam annotation -- so a null anchor means the two
    annotation tables disagree with each other. A silent empty list would quietly shrink
    every reachability denominator downstream, so fail loudly instead.
    """
    out = features.join(anchor, on="accession", how="left")
    missing = out.filter(pl.col("anchor_pfam").is_null())["accession"].unique().to_list()
    if missing:
        raise SystemExit(
            f"{label}: {len(missing)} accessions carry Swiss-Prot features but no Pfam "
            f"annotation, e.g. {sorted(missing)[:5]}. The reachability key needs both."
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sprot-dat", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path,
                   help="Pfam annotation dir; used only to know which accessions belong "
                        "to which species")
    p.add_argument("--truth-out", required=True, type=Path)
    p.add_argument("--map-outdir", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    p.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = p.parse_args()

    species_acc, species_anchor = {}, {}
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        sp = path.name.replace("_pfam_domains.parquet", "")
        ann = pl.read_parquet(path, columns=["accession", "pfam_id"])
        species_acc[sp] = set(ann["accession"].unique().to_list())
        species_anchor[sp] = ann.group_by("accession").agg(
            pl.col("pfam_id").unique().alias("anchor_pfam")
        )
    if "human" not in species_acc:
        raise SystemExit("no human_pfam_domains.parquet; cannot define the query side")

    all_acc = set().union(*species_acc.values())
    wanted = set(args.features)
    rows = parse(args.sprot_dat, wanted, all_acc)

    df = pl.DataFrame(
        rows,
        schema=["accession", "pfam_id", "domain_start", "domain_end", "protein_length",
                "is_point"],
        orient="row",
    ).unique()

    # Swiss-Prot uses 1-based inclusive coordinates; the Pfam tables this must line up with
    # are already in the same convention, so no shift is applied. Zero-length intervals
    # cannot be scored, and point features are widened by one so an interval exists at all.
    df = df.with_columns(
        pl.when(pl.col("is_point"))
        .then(pl.col("domain_end") + 1)
        .otherwise(pl.col("domain_end"))
        .alias("domain_end")
    ).filter(pl.col("domain_end") > pl.col("domain_start"))

    human = add_anchor(
        df.filter(pl.col("accession").is_in(species_acc["human"])),
        species_anchor["human"], "human",
    )
    human.write_parquet(args.truth_out, compression="zstd")

    args.map_outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.sprot_dat),
        "note": "pfam_id column holds the Swiss-Prot feature type, not a Pfam accession",
        "features_requested": sorted(wanted),
        "human": {
            "n_features": human.height,
            "n_proteins": human["accession"].n_unique(),
            "n_annotated_proteins": len(species_acc["human"]),
            # Share of this species' Pfam-annotated proteins that carry any curated
            # Swiss-Prot feature. It is a property of how deeply curators have worked on
            # the organism, not of the organism, and it spans 0.2% to 93% across the nine
            # targets. Emitted so a species with almost no curation is never read as a
            # species no tool could reach.
            "coverage_fraction": human["accession"].n_unique() / len(species_acc["human"]),
            "n_point": int(human["is_point"].sum()),
            "by_type": {
                r["pfam_id"]: r["len"]
                for r in human.group_by("pfam_id").len().sort("len", descending=True).to_dicts()
            },
        },
    }
    for sp, accs in species_acc.items():
        if sp == "human":
            continue
        sub = add_anchor(df.filter(pl.col("accession").is_in(accs)), species_anchor[sp], sp)
        sub.write_parquet(args.map_outdir / f"{sp}_domain_map.parquet", compression="zstd")
        summary[sp] = {
            "n_features": sub.height,
            "n_proteins": sub["accession"].n_unique(),
            "n_annotated_proteins": len(accs),
            "coverage_fraction": sub["accession"].n_unique() / len(accs),
            # Kept for continuity, and as the thing that shows why it was never a ceiling:
            # a feature type absent from the target cannot be transferred, but with twelve
            # types in the whole vocabulary every proteome has nearly all of them, so this
            # reads 11 or 12 for every species.
            "n_human_types_present": (
                sub.select("pfam_id").unique()
                .join(human.select("pfam_id").unique(), on="pfam_id", how="inner").height
            ),
            # The reachability denominator that replaced it: human feature instances this
            # proteome could actually have supplied, counting an instance as reachable when
            # some annotated target protein carries the same feature type AND shares a Pfam
            # family with the human query. See the module docstring.
            "n_reachable_human_features": reachable_count(human, sub),
        }

    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
