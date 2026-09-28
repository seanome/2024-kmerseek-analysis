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

Point vs range features behave differently and both are kept, flagged by `is_point`:
  range  DNA_BIND, TRANSMEM, REGION, MOTIF, COILED, ZN_FING, REPEAT, DOMAIN
         -- boundary evaluation, directly comparable to Pfam envelopes
  point  ACT_SITE, BINDING, SITE
         -- catalytic/binding residue recall. A 1-residue truth interval cannot be scored
            by boundary IoU in any meaningful way, so downstream work should filter on
            is_point rather than silently mixing the two.

`anchor_pfam` is the reachability key, and it is why this truth set needs a column the
Pfam truth set does not. Reachability asks whether a human annotation could have been
transferred from this target proteome at all, so a tool is not punished for missing
something the target does not contain. Under the Pfam truth set the answer key IS the
transfer key -- `pfam_id` is a family, and "the target has family F" is a real, varying
statement. Here `pfam_id` is a Swiss-Prot feature type with twelve levels, and "the target
proteome contains at least one TRANSMEM somewhere" is true of every proteome. Joining on
it returned the same constant for eight of nine species (7000, and 6991 for ciona, which
is 7000 minus the 9 human INTRAMEM features) -- a denominator that measured nothing.

So each row carries the Pfam families of ITS OWN protein, on both sides. A human feature
instance is reachable in a target proteome when that proteome has a single annotated
protein that both carries the same feature type and shares a Pfam family with the human
query -- the two things one homology transfer needs. Under the Pfam truth set the second
condition is implied by the first, which is why the Pfam and Pfam-N arms keep the plain
family join and are unchanged by any of this.
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


def add_anchor(features: pl.DataFrame, anchor: pl.DataFrame, label: str) -> pl.DataFrame:
    """Attach each protein's Pfam families to its Swiss-Prot features.

    The join is total by construction -- `features` was already restricted to accessions
    that appear in this species' Pfam annotation -- so a null anchor means the annotation
    tables disagree with each other, and a silent empty list would quietly shrink every
    reachability denominator downstream. Fail instead.
    """
    out = features.join(anchor, on="accession", how="left")
    missing = out.filter(pl.col("anchor_pfam").is_null())["accession"].unique().to_list()
    if missing:
        raise SystemExit(
            f"{label}: {len(missing)} accessions carry Swiss-Prot features but no Pfam "
            f"annotation, e.g. {missing[:5]}. The two annotation tables are out of sync."
        )
    return out


def reachable_count(human_instances: pl.DataFrame, target: pl.DataFrame) -> int:
    """How many human feature instances this target proteome could transfer.

    One annotated target protein has to do both jobs: carry the feature type being
    transferred, and be findable from the human query. Sharing a Pfam family stands in for
    findable -- it is the only homology signal available that no tool under test produced,
    so it cannot hand any arm its own difficulty. Requiring both of ONE protein, rather
    than each of some protein, is what a single transfer event actually needs.
    """
    key = ["accession", "pfam_id", "domain_start", "domain_end"]
    target_keys = (
        target.select("pfam_id", "anchor_pfam").explode("anchor_pfam").unique()
    )
    return (
        human_instances.explode("anchor_pfam")
        .join(target_keys, on=["pfam_id", "anchor_pfam"], how="semi")
        .select(key)
        .unique()
        .height
    )


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

    species_acc = {}
    species_anchor = {}
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        sp = path.name.replace("_pfam_domains.parquet", "")
        ann = pl.read_parquet(path).select("accession", "pfam_id").unique()
        species_acc[sp] = set(ann["accession"].unique().to_list())
        species_anchor[sp] = ann.group_by("accession").agg(
            pl.col("pfam_id").unique().sort().alias("anchor_pfam")
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

    human = add_anchor(df.filter(pl.col("accession").is_in(species_acc["human"])),
                       species_anchor["human"], "human")
    human.write_parquet(args.truth_out, compression="zstd")

    # One row per distinct human feature instance, carrying its protein's Pfam families.
    # The summary's reachability counts are computed from this, by exactly the join the
    # scorer performs, so the two can never drift into disagreeing about the denominator.
    human_instances = human.select(
        "accession", "pfam_id", "domain_start", "domain_end", "anchor_pfam"
    ).unique(subset=["accession", "pfam_id", "domain_start", "domain_end"])

    args.map_outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.sprot_dat),
        "note": "pfam_id column holds the Swiss-Prot feature type, not a Pfam accession",
        "features_requested": sorted(wanted),
        "human": {
            "n_features": human.height,
            "n_proteins": human["accession"].n_unique(),
            "n_proteins_pfam_annotated": len(species_acc["human"]),
            "coverage_fraction": (human["accession"].n_unique()
                                  / len(species_acc["human"])) if species_acc["human"] else 0.0,
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
            # Denominator is the Pfam-annotated proteome, which is the set the Pfam truth
            # arm transfers from, so the two arms' coverage is on one scale. Ciona lands at
            # 0.2% against mouse's 76%: Swiss-Prot has 28 reviewed entries for it in total.
            # That is a property of the annotation database, not of any tool, and it is
            # what makes ciona's swissprot call counts collapse while its pfam column is
            # normal. Reported so that collapse is attributable rather than mysterious.
            "n_proteins_pfam_annotated": len(accs),
            "coverage_fraction": (sub["accession"].n_unique() / len(accs)) if accs else 0.0,
            # The old ceiling, kept only to show what it was worth. A feature type absent
            # from the target cannot be transferred -- true, but with twelve types it is
            # 12 for every proteome that has any annotation at all, which is why the
            # denominator it produced was a constant.
            "n_human_types_present": (
                sub.select("pfam_id").unique()
                .join(human.select("pfam_id").unique(), on="pfam_id", how="inner").height
            ),
            "n_reachable_human_features": reachable_count(human_instances, sub),
            "n_human_features": human.height,
        }

    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
