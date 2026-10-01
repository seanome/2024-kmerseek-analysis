#!/usr/bin/env python3
"""Where the P66/CD47 shared k-mers sit, and whether they touch the known residues.

Notebook 241 counted how many k-mers P66 and CD47 share in each arm of its sweep
(one alphabet at one k) but not where they sit. This runs `kmerseek pair` again on
every arm that shares at least one k-mer, keeps the per-k-mer positions, and adds the
two controls.

Numbering. kmerseek reports 0-based positions in the sequences it was given. The P66
record is the 597-residue mature chain, so its positions are mature numbering and
UniProt = mature + 21. The CD47 record is the 323-residue precursor, so its positions
are UniProt numbering and mature = UniProt - 18. Both sequences are checked against
the UniProt entries in tables/245_p66_cd47_annotations.csv before anything is written.

Writes, under tables/:
  245_p66_cd47_shared_kmers.csv  one row per shared k-mer, both numberings on both
                                 proteins, which named residues it covers, and which
                                 UniProt feature of CD47 it lands in.
  245_p66_cd47_regions.csv       the chained regions kmerseek reports, same columns.
  245_p66_cd47_controls.csv      the two controls: the chance rate for a k-mer of that
                                 length landing on each named stretch (binomial test),
                                 and where CD47's count sits among 300 human proteins
                                 of similar length (the list PR #44 drew,
                                 analysis/ranking-metrics-p66/random_ladder.csv).

Usage:
  245_p66_cd47_pair_runs.py [--workers 4] [--force]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TABLES = REPO / "tables"
ANNOTATIONS = TABLES / "245_p66_cd47_annotations.csv"
RANDOM_LADDER = REPO / "analysis" / "ranking-metrics-p66" / "random_ladder.csv"

KMERSEEK = Path("/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek")
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
SWEEP = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
QUERIES = SWEEP / "queries.fa"
ARMS_CSV = SWEEP / "arms.csv"
WORK = SWEEP / "p66_cd47_coordinates"

CD47_HEADER = (
    "ENSP00000355361.5|ENST00000361309.6|ENSG00000196776.18|"
    "OTTHUMG00000044216.4|OTTHUMT00000102793.1|CD47-202|CD47|323"
)

# Offsets from the signal peptides: mature = uniprot - offset.
P66_OFFSET, CD47_OFFSET = 21, 18
P66_LEN_MATURE, CD47_LEN_UNIPROT = 597, 323

# The stretches the literature names, in UniProt numbering (see the fetch script).
P66_LOOP = (202, 208)           # the loop required for integrin binding
CD47_CONTACT_SPAN = (115, 124)  # the SIRP-alpha contact residues that lie in one span
CD47_CONTACT_RESIDUES = (115, 117, 118, 120, 121, 122, 124)

# PR #44's 300 control proteins run 242 to 404 residues, median 322.5, against CD47's
# 323: the set is centred on CD47 but the window is wider than a few residues. The list
# is reused exactly rather than redrawn, so this is a check that it still is that set,
# not a filter.
CONTROL_LEN_MIN, CONTROL_LEN_MAX = 242, 404


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def check_sequences(p66: str, cd47: str) -> None:
    """Stop unless the two sequences are the ones the coordinates were built on."""
    ann = pl.read_csv(ANNOTATIONS)
    full = {r["protein"]: r["sequence"]
            for r in ann.filter(pl.col("feature_type") == "SEQUENCE").iter_rows(named=True)}
    if p66 != full["P66"][P66_OFFSET:]:
        raise SystemExit("the P66 query is not the mature chain of UniProt H7C7N8")
    if cd47 != full["CD47"]:
        raise SystemExit("the CD47 target is not the precursor of UniProt Q08722")
    if len(p66) != P66_LEN_MATURE or len(cd47) != CD47_LEN_UNIPROT:
        raise SystemExit(f"lengths are {len(p66)} and {len(cd47)}, expected "
                         f"{P66_LEN_MATURE} and {CD47_LEN_UNIPROT}")


def run_pair(query_name: str, target_fasta: Path, target_name: str,
             alphabet: str, k: int, out: Path, force: bool) -> dict:
    if out.exists() and not force:
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(KMERSEEK), "pair", "-q", str(QUERIES), "--query-name", query_name,
           "-t", str(target_fasta), "--target-name", target_name,
           "-k", str(k), "-a", alphabet, "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"kmerseek pair failed for {alphabet} k={k} {target_name}:\n{p.stderr}")
    return json.loads(out.read_text())


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Residues shared by two inclusive spans."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def feature_table() -> pl.DataFrame:
    """CD47's UniProt features, smallest first, so the most specific one wins."""
    ann = pl.read_csv(ANNOTATIONS)
    return (ann.filter((pl.col("protein") == "CD47")
                       & (~pl.col("feature_type").is_in(["SEQUENCE", "PUBLISHED", "Chain"])))
            .with_columns((pl.col("end_uniprot") - pl.col("start_uniprot")).alias("width"))
            .sort("width"))


def feature_at(features: pl.DataFrame, pos: int) -> str:
    """The narrowest CD47 feature covering this UniProt position, named for a reader."""
    for r in features.iter_rows(named=True):
        if r["start_uniprot"] <= pos <= r["end_uniprot"]:
            name = r["name"] or r["feature_type"]
            return f"{r['feature_type']}: {name}" if r["name"] else r["feature_type"]
    return "none"


def kmer_rows(arm: dict, pair: dict, features: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    """One row per shared k-mer and one per chained region, in both numberings."""
    a, k, bits = arm["alphabet"], arm["ksize"], arm["bits"]
    qseq, tseq = pair["query"]["sequence"], pair["target"]["sequence"]
    kmers, regions = [], []
    for sk in pair["shared_kmers"]:
        # 0-based start -> 1-based inclusive span, P66 in mature, CD47 in UniProt.
        pm0, pm1 = sk["query_pos"] + 1, sk["query_pos"] + k
        cu0, cu1 = sk["target_pos"] + 1, sk["target_pos"] + k
        p66_res, cd47_res = qseq[pm0 - 1:pm1], tseq[cu0 - 1:cu1]
        pu0, pu1 = pm0 + P66_OFFSET, pm1 + P66_OFFSET
        covered = [p for p in CD47_CONTACT_RESIDUES if cu0 <= p <= cu1]
        kmers.append(dict(
            alphabet=a, ksize=k, bits=bits, kmer_letters=sk["kmer"],
            p66_start_uniprot=pu0, p66_end_uniprot=pu1,
            p66_start_mature=pm0, p66_end_mature=pm1,
            cd47_start_uniprot=cu0, cd47_end_uniprot=cu1,
            cd47_start_mature=cu0 - CD47_OFFSET, cd47_end_mature=cu1 - CD47_OFFSET,
            p66_residues=p66_res, cd47_residues=cd47_res,
            n_identical_residues=sum(x == y for x, y in zip(p66_res, cd47_res)),
            overlaps_p66_loop=overlap((pu0, pu1), P66_LOOP) > 0,
            n_p66_loop_residues_covered=overlap((pu0, pu1), P66_LOOP),
            overlaps_cd47_contact_span=overlap((cu0, cu1), CD47_CONTACT_SPAN) > 0,
            n_cd47_contact_residues_covered=len(covered),
            cd47_feature_hit=feature_at(features, cu0),
        ))
    for rg in pair["regions"]:
        pm0, pm1 = rg["query_start"] + 1, rg["query_end"]
        cu0, cu1 = rg["target_start"] + 1, rg["target_end"]
        pu0, pu1 = pm0 + P66_OFFSET, pm1 + P66_OFFSET
        covered = [p for p in CD47_CONTACT_RESIDUES if cu0 <= p <= cu1]
        regions.append(dict(
            alphabet=a, ksize=k, bits=bits, region_length=rg["length"],
            p66_start_uniprot=pu0, p66_end_uniprot=pu1,
            p66_start_mature=pm0, p66_end_mature=pm1,
            cd47_start_uniprot=cu0, cd47_end_uniprot=cu1,
            cd47_start_mature=cu0 - CD47_OFFSET, cd47_end_mature=cu1 - CD47_OFFSET,
            p66_residues=qseq[pm0 - 1:pm1], cd47_residues=tseq[cu0 - 1:cu1],
            n_identical_residues=sum(x == y for x, y in
                                     zip(qseq[pm0 - 1:pm1], tseq[cu0 - 1:cu1])),
            overlaps_p66_loop=overlap((pu0, pu1), P66_LOOP) > 0,
            n_p66_loop_residues_covered=overlap((pu0, pu1), P66_LOOP),
            overlaps_cd47_contact_span=overlap((cu0, cu1), CD47_CONTACT_SPAN) > 0,
            n_cd47_contact_residues_covered=len(covered),
            cd47_feature_hit=feature_at(features, cu0),
        ))
    return kmers, regions


def chance_rate(seq_len: int, k: int, span: tuple[int, int]) -> float:
    """Share of the places a k-mer can sit in a protein of this length from which it
    touches this span. Every start position is counted once, which is what a k-mer
    drawn uniformly along the sequence does."""
    n_positions = seq_len - k + 1
    if n_positions <= 0:
        return float("nan")
    touching = sum(1 for s in range(1, n_positions + 1) if overlap((s, s + k - 1), span) > 0)
    return touching / n_positions


def control_proteins(human: dict[str, str]) -> dict[str, str]:
    """PR #44's 300 length-matched human proteins, by gene symbol, with the GENCODE
    header each one needs on the kmerseek command line."""
    genes = (pl.read_csv(RANDOM_LADDER)["gene"].unique().sort().to_list())
    by_gene: dict[str, str] = {}
    for header in human:
        parts = header.split("|")
        if len(parts) > 6:
            by_gene.setdefault(parts[6], header)
    out, missing = {}, []
    for g in genes:
        h = by_gene.get(g)
        if h is None:
            missing.append(g)
            continue
        out[g] = h
    if missing:
        print(f"  {len(missing)} of {len(genes)} control genes are not in this GENCODE "
              f"release and are dropped: {', '.join(missing[:10])}", file=sys.stderr)
    bad = {g: len(human[h]) for g, h in out.items()
           if not CONTROL_LEN_MIN <= len(human[h]) <= CONTROL_LEN_MAX}
    if bad:
        raise SystemExit(f"control proteins outside {CONTROL_LEN_MIN}-{CONTROL_LEN_MAX} aa: {bad}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-run pairs already on disk")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    arms = (pl.read_csv(ARMS_CSV, infer_schema_length=None)
            .filter(pl.col("pair_P66_shared_kmers") > 0)
            .select("alphabet", "ksize", "bits", "pair_P66_shared_kmers")
            .sort("alphabet", "ksize"))
    print(f"{arms.height} arms share at least one k-mer between P66 and CD47 "
          f"({arms['pair_P66_shared_kmers'].sum()} k-mers in notebook 241)")

    features = feature_table()

    # The pair runs against CD47.
    kmer_rows_all, region_rows_all = [], []
    for arm in arms.iter_rows(named=True):
        tag = f"{arm['alphabet']}.k{arm['ksize']}"
        pair = run_pair("P66", HUMAN, CD47_HEADER, arm["alphabet"], arm["ksize"],
                        WORK / f"cd47.{tag}.json", args.force)
        check_sequences(pair["query"]["sequence"], pair["target"]["sequence"])
        if len(pair["shared_kmers"]) != arm["pair_P66_shared_kmers"]:
            raise SystemExit(
                f"{tag}: this run found {len(pair['shared_kmers'])} shared k-mers, "
                f"notebook 241 recorded {arm['pair_P66_shared_kmers']}")
        k_rows, r_rows = kmer_rows(arm, pair, features)
        kmer_rows_all += k_rows
        region_rows_all += r_rows

    kmers = pl.DataFrame(kmer_rows_all)
    regions = pl.DataFrame(region_rows_all)
    kmers.write_csv(TABLES / "245_p66_cd47_shared_kmers.csv")
    regions.write_csv(TABLES / "245_p66_cd47_regions.csv")
    print(f"wrote {kmers.height} shared k-mers and {regions.height} regions")

    # Control (a): position. Control (b): 300 length-matched human proteins.
    human = read_fasta(HUMAN)
    controls = control_proteins(human)
    control_fa = WORK / "control_proteins.fa"
    control_fa.write_text("".join(f">{h}\n{human[h]}\n" for h in controls.values()))
    print(f"{len(controls)} control proteins, "
          f"{min(len(human[h]) for h in controls.values())}-"
          f"{max(len(human[h]) for h in controls.values())} aa")

    jobs = [(arm, gene, header) for arm in arms.iter_rows(named=True)
            for gene, header in controls.items()]

    def one(job):
        arm, gene, header = job
        tag = f"{arm['alphabet']}.k{arm['ksize']}"
        pair = run_pair("P66", control_fa, header, arm["alphabet"], arm["ksize"],
                        WORK / "control" / f"{tag}.{gene}.json", args.force)
        return tag, gene, len(pair["shared_kmers"])

    counts: dict[str, dict[str, int]] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (tag, gene, n) in enumerate(ex.map(one, jobs), 1):
            counts.setdefault(tag, {})[gene] = n
            if i % 1000 == 0:
                print(f"  control pairs: {i}/{len(jobs)}", file=sys.stderr, flush=True)

    rows = []
    for arm in arms.iter_rows(named=True):
        a, k, tag = arm["alphabet"], arm["ksize"], f"{arm['alphabet']}.k{arm['ksize']}"
        sub = kmers.filter((pl.col("alphabet") == a) & (pl.col("ksize") == k))
        n = sub.height
        obs_loop = int(sub["overlaps_p66_loop"].sum())
        obs_contact = int(sub["overlaps_cd47_contact_span"].sum())
        p_loop = chance_rate(P66_LEN_MATURE, k, (P66_LOOP[0] - P66_OFFSET, P66_LOOP[1] - P66_OFFSET))
        p_contact = chance_rate(CD47_LEN_UNIPROT, k, CD47_CONTACT_SPAN)
        cd47_n = int(arm["pair_P66_shared_kmers"])
        ctrl = counts[tag]
        vals = list(ctrl.values())
        n_ge = sum(1 for v in vals if v >= cd47_n)
        rows.append(dict(
            alphabet=a, ksize=k, bits=arm["bits"], n_shared_kmers=n,
            # (a) position control
            p66_loop_chance_rate=round(p_loop, 5),
            n_touching_p66_loop=obs_loop,
            expected_touching_p66_loop=round(n * p_loop, 3),
            p66_loop_binomial_p=(float(stats.binomtest(obs_loop, n, p_loop).pvalue)
                                 if n else None),
            cd47_contact_chance_rate=round(p_contact, 5),
            n_touching_cd47_contact_span=obs_contact,
            expected_touching_cd47_contact_span=round(n * p_contact, 3),
            cd47_contact_binomial_p=(float(stats.binomtest(obs_contact, n, p_contact).pvalue)
                                     if n else None),
            # (b) partner control
            n_control_proteins=len(vals),
            control_median_shared_kmers=float(pl.Series(vals).median()),
            control_max_shared_kmers=max(vals),
            n_control_ge_cd47=n_ge,
            cd47_percentile_among_controls=round(100 * (1 - n_ge / len(vals)), 1),
        ))
    controls_df = pl.DataFrame(rows)
    controls_df.write_csv(TABLES / "245_p66_cd47_controls.csv")
    # The raw control counts, so the notebook can draw the distribution.
    pl.DataFrame([{"alphabet": t.rsplit(".k", 1)[0], "ksize": int(t.rsplit(".k", 1)[1]),
                   "gene": g, "shared_kmers": v}
                  for t, d in counts.items() for g, v in d.items()]
                 ).write_csv(TABLES / "245_p66_cd47_control_counts.csv")
    print(f"wrote controls for {controls_df.height} arms")


if __name__ == "__main__":
    main()
