#!/usr/bin/env python3
"""Write the search inputs of the time-split test (notebook 280, Stops 2 and 3).

    make_search_inputs.py reference --qs <query_set> --out <dir>
        reference.fasta      Swiss-Prot 2019_08, header = bare accession
        sprot_domains.tsv    accession, Pfam 33.1 family, start, end (1-based, end included)
    make_search_inputs.py queries --qs <query_set> --sample <proposed_sample_queries.csv>
                          [--n-pilot 200 --controls <identity_sample.csv> --n-controls 20]
                          --out <dir>
        queries.fasta        real queries and one DECOY_ copy each (dipeptide shuffle)
        queries.fasta.md5
        queries.tsv          query, md5, is_decoy, source, half (tune/test), role
        instances.tsv        the kept t1 instances on the real queries
    make_search_inputs.py settings --ranges <tsv> --kappa <tsv> --out <dir>
        settings.tsv         alphabet, ksize, c_opt: every alphabet, each k from k_min to
                             k_max, every other k when there are more than 5

Queries and targets are named by bare accession so kmerseek, MMseqs2 and HMMER all report
the same name (see the project memory on the mmseqs UniProt-header trap).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from time_split_common import (  # noqa: E402
    DECOY_PREFIX,
    DECOY_SEED,
    dipeptide_shuffle,
    pair_counts,
    query_half,
)

PILOT_SEED = 280
CONTROL_SEED = 283
CONTROL_MIN_IDENTITY = 90.0  # positive control: closest t0 Swiss-Prot domain >= 90% identical
CONTROL_MIN_COVERAGE = 0.8  # ... over at least 80% of the t1 domain
MAX_K_PER_ALPHABET = 5  # more k-sizes than this in an alphabet's range: take every other


def wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def reference(qs: Path, out: Path) -> None:
    sp = pl.read_parquet(qs / "sprot_2019_08.parquet")
    with open(out / "reference.fasta", "w") as fh:
        for acc, seq in sp.select("acc", "seq").iter_rows():
            fh.write(f">{acc}\n{wrap(seq)}\n")
    dom = pl.read_parquet(qs / "sprot_2019_08_pfam33.1_domains.parquet").sort("sp_acc", "sp_start")
    dom.write_csv(out / "sprot_domains.tsv", separator="\t")
    # Pfam 38.2 clans, for the same-clan secondary readout.
    clans = qs.parent / "data/Pfam38.2/Pfam-A.clans.tsv.gz"
    if clans.exists():
        pl.read_csv(clans, separator="\t", has_header=False, infer_schema=False).write_csv(
            out / "pfam_clans_t1.tsv", separator="\t", include_header=False)
    print(f"reference: {sp.height:_} proteins; {dom.height:_} Pfam 33.1 domains")


def queries(qs: Path, sample_csv: Path, out: Path, n_pilot: int | None,
            controls_csv: Path | None, n_controls: int) -> None:
    kept = pl.read_parquet(qs / "instances_kept.parquet")
    sample = pl.read_csv(sample_csv)
    roles = sample.select("md5", "acc").with_columns(role=pl.lit("sample"))
    if n_pilot is not None:
        rng = random.Random(PILOT_SEED)
        idx = sorted(rng.sample(range(roles.height), min(n_pilot, roles.height)))
        roles = roles[idx]
    if controls_csv is not None and n_controls:
        ident = pl.read_csv(controls_csv)
        pool = (ident.filter((pl.col("pident") >= CONTROL_MIN_IDENTITY)
                             & (pl.col("query_coverage") >= CONTROL_MIN_COVERAGE))
                .filter(~pl.col("md5").is_in(roles["md5"].implode()))
                .unique("md5").sort("acc"))
        if pool.height < n_controls:
            raise SystemExit(f"only {pool.height} positive-control candidates, need {n_controls}")
        rng = random.Random(CONTROL_SEED)
        pick = pool[sorted(rng.sample(range(pool.height), n_controls))]
        roles = pl.concat([roles, pick.select("md5", "acc").with_columns(role=pl.lit("control"))])
    seqs = (pl.scan_parquet(qs / "pfamseq_33.1.parquet").select("acc", "seq")
            .join(roles.lazy().select("acc"), on="acc", how="semi").collect())
    roles = roles.join(seqs, on="acc").sort("role", "acc")

    rng = random.Random(DECOY_SEED)
    lines, rows = [], []
    for acc, md5, role, seq in roles.select("acc", "md5", "role", "seq").iter_rows():
        decoy = dipeptide_shuffle(seq, rng)
        assert pair_counts(decoy) == pair_counts(seq), acc
        half = query_half(acc)
        lines.append(f">{acc}\n{wrap(seq)}\n>{DECOY_PREFIX}{acc}\n{wrap(decoy)}\n")
        rows.append((acc, md5, False, acc, half, role, len(seq)))
        rows.append((f"{DECOY_PREFIX}{acc}", hashlib.md5(decoy.encode()).hexdigest(), True, acc,
                     half, role, len(decoy)))
    text = "".join(lines)
    (out / "queries.fasta").write_text(text)
    (out / "queries.fasta.md5").write_text(hashlib.md5(text.encode()).hexdigest() + "  queries.fasta\n")
    qtab = pl.DataFrame(rows, schema=["query", "md5", "is_decoy", "source", "half", "role", "length"],
                        orient="row")
    qtab.write_csv(out / "queries.tsv", separator="\t")
    inst = kept.join(roles.select("md5", "role"), on="md5").select(
        pl.col("acc").alias("query"), "role", "pfam_acc", "pfam_id", "pfam_type", "clan_t1",
        "t1_start", "t1_end", "length", "length_bin", "kingdom", "uncharacterized_t0")
    inst.write_csv(out / "instances.tsv", separator="\t")
    print(qtab.group_by("role", "is_decoy", "half").len().sort("role", "is_decoy", "half"))
    print(f"{inst.height:_} instances on {inst['query'].n_unique():_} real queries")


def settings(ranges: Path, kappa: Path, out: Path) -> None:
    r = pl.read_csv(ranges, separator="\t")
    kap = pl.read_csv(kappa, separator="\t", comment_prefix="#").select("alphabet", "c_opt")
    rows = []
    for a, lo, hi in r.select("alphabet", "k_min", "k_max").iter_rows():
        ks = list(range(lo, hi + 1))
        ks = ks if len(ks) <= MAX_K_PER_ALPHABET else ks[::2]
        rows += [(a, k) for k in ks]
    s = pl.DataFrame(rows, schema=["alphabet", "ksize"], orient="row").join(kap, on="alphabet", how="left")
    missing = s.filter(pl.col("c_opt").is_null())["alphabet"].unique().to_list()
    if missing:
        raise SystemExit(f"no c_opt for {missing}")
    s.write_csv(out / "settings.tsv", separator="\t")
    print(s.group_by("alphabet", maintain_order=True).agg("ksize", "c_opt"))
    print(f"{s.height} alphabet-k pairs, {2 * s.height} search settings (exact and extend)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "queries", "settings"])
    ap.add_argument("--qs", type=Path)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--n-pilot", type=int)
    ap.add_argument("--controls", type=Path)
    ap.add_argument("--n-controls", type=int, default=0)
    ap.add_argument("--ranges", type=Path)
    ap.add_argument("--kappa", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "reference":
        reference(a.qs, a.out)
    elif a.mode == "queries":
        queries(a.qs, a.sample, a.out, a.n_pilot, a.controls, a.n_controls)
    else:
        settings(a.ranges, a.kappa, a.out)


if __name__ == "__main__":
    main()
