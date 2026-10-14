"""Build T2 (Pfam domain-order swaps in Swiss-Prot) and the pool N1 is drawn from.

Usage:
    python3 -I build_t2_n1_pool.py DATA_DIR SWISSPROT_FASTA_GZ [--per-group 5] [--max-pairs 3000]
        [--n1-pool-factor 5] [--seed 293]

Inputs:
    DATA_DIR/downloads/pfam/Pfam-A.regions.swissprot_2026_03.tsv  (Pfam 38.2, Swiss-Prot rows)
    SWISSPROT_FASTA_GZ  Swiss-Prot 2026_03

Steps:
    1. Keep Pfam rows whose sequence md5 equals the md5 of the Swiss-Prot 2026_03 sequence,
       so Pfam coordinates are valid on the sequence used.
    2. Drop proteins in which any two Pfam envelopes overlap by more than
       MAX_DOMAIN_OVERLAP residues. Nested or overlapping domains (PF07687 inside PF01546,
       the peptidase M20 dimerisation domain, is the largest case) have no order along the
       sequence, so they cannot be swapped. Architecture of a protein = its Pfam families
       ordered by envelope start. Keep proteins with at least two distinct families.
    3. Group proteins by the multiset of families (each family with its count). Inside a
       group, two proteins whose families come in a different order are a T2 pair, if the
       order also differs when each family is replaced by its Pfam clan
       (Pfam-A.clans.tsv.gz; a family with no clan stands for itself). Without that rule
       the largest groups are EF-hand proteins whose EF-hand pairs are labelled by
       different EF-hand families of clan CL0220 in different proteins: a difference of
       labels, not of order. At least one pair of families must also come in opposite
       orders (by the first copy of each family): this drops pairs that differ only in
       where a repeated domain sits (PF00400-PF23098-PF00400 against
       PF00400-PF00400-PF23098). Because the family multisets are equal, no T2 pair
       differs only by a change in repeat count.
    4. T2 count = every such pair. T2 sample = at most --per-group pairs per multiset group
       (each from a different pair of orders where possible), then at most --max-pairs in
       total, fixed seed.
    5. N1 pool = pairs with the same families in the same order (same architecture string),
       sampled at random, --n1-pool-factor times the T2 sample size. N1 itself is drawn from
       this pool by select_n1.py to match T2's identity distribution.

Outputs in DATA_DIR:
    swissprot_architectures.parquet  one row per protein kept in step 2
    t2_counts.tsv                    counts after each step
    t2_all_pair_counts_by_group.parquet  T2 pairs per multiset group (the full count)
    t2_order_pairs.parquet           every pair of orders counted in T2, with the family
                                     pairs whose order differs
    t2_pairs.parquet                 the T2 sample (pair_set = "T2")
    n1_pool.parquet                  the N1 pool (pair_set = "N1_pool")
"""

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402

MAX_DOMAIN_OVERLAP = (
    10  # residues; Pfam envelopes of adjacent domains often touch or overlap a little
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("swissprot")
    p.add_argument("--per-group", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=3000)
    p.add_argument("--n1-pool-factor", type=int, default=20)
    p.add_argument("--seed", type=int, default=293)
    args = p.parse_args()
    data = Path(args.data_dir)
    rng = random.Random(args.seed)
    counts = []

    seqs = ku.read_fasta(args.swissprot)
    regions = pl.read_csv(
        data / "downloads/pfam/Pfam-A.regions.swissprot_2026_03.tsv",
        separator="\t",
        schema_overrides={"pfamseq_acc": pl.Utf8, "md5": pl.Utf8, "pfamA_acc": pl.Utf8},
    )
    counts.append(("pfam_rows_swissprot", regions.height))
    counts.append(("proteins_with_pfam", regions["pfamseq_acc"].n_unique()))
    md5 = pl.DataFrame(
        {
            "pfamseq_acc": list(seqs),
            "sprot_md5": [hashlib.md5(s.encode()).hexdigest() for s in seqs.values()],
        }
    )
    regions = regions.join(md5, on="pfamseq_acc", how="inner").filter(
        pl.col("md5") == pl.col("sprot_md5")
    )
    counts.append(
        ("proteins_md5_matches_swissprot_2026_03", regions["pfamseq_acc"].n_unique())
    )

    regions = regions.sort("pfamseq_acc", "seq_start", "seq_end", "pfamA_acc")
    overlap = regions.with_columns(
        (
            pl.col("seq_end").cum_max().shift(1).over("pfamseq_acc")
            - pl.col("seq_start")
            + 1
        ).alias("overlap_with_earlier")
    )
    overlapping = overlap.filter(pl.col("overlap_with_earlier") > MAX_DOMAIN_OVERLAP)[
        "pfamseq_acc"
    ].unique()
    regions = regions.filter(~pl.col("pfamseq_acc").is_in(overlapping.implode()))
    counts.append(
        ("proteins_no_domain_overlap_over_10aa", regions["pfamseq_acc"].n_unique())
    )
    arch = (
        regions.group_by("pfamseq_acc", maintain_order=True)
        .agg(
            pl.col("pfamA_acc").alias("families_in_order"),
            pl.col("seq_start").alias("domain_starts"),
            pl.col("seq_end").alias("domain_ends"),
        )
        .with_columns(
            pl.col("families_in_order").list.join("-").alias("architecture"),
            pl.col("families_in_order")
            .list.sort()
            .list.join("+")
            .alias("family_multiset"),
            pl.col("families_in_order").list.n_unique().alias("n_distinct_families"),
        )
        .filter(pl.col("n_distinct_families") >= 2)
        .rename({"pfamseq_acc": "accession"})
    )
    arch = arch.with_columns(
        pl.col("accession")
        .replace_strict({a: len(s) for a, s in seqs.items()})
        .alias("length")
    )
    arch.write_parquet(data / "swissprot_architectures.parquet")
    counts.append(("proteins_with_2plus_distinct_families", arch.height))

    clans = pl.read_csv(
        data / "downloads/pfam/Pfam-A.clans.tsv.gz",
        separator="\t",
        has_header=False,
        new_columns=["family", "clan", "clan_id", "family_id", "description"],
        quote_char=None,
    )
    clan_of = {
        f: (c if c else f) for f, c in clans.select("family", "clan").iter_rows()
    }

    def clan_architecture(architecture: str) -> str:
        return "-".join(clan_of.get(f, f) for f in architecture.split("-"))

    by_group: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for acc, ms, ar in arch.select(
        "accession", "family_multiset", "architecture"
    ).iter_rows():
        by_group[ms][ar].append(acc)

    def swapped_family_pairs(o1: str, o2: str) -> list[str]:
        """Family pairs (by first occurrence) that come in opposite orders in o1 and o2."""
        f1, f2 = o1.split("-"), o2.split("-")
        first1 = {f: f1.index(f) for f in set(f1)}
        first2 = {f: f2.index(f) for f in set(f2)}
        fams = sorted(first1)
        return [
            f"{x}/{y}"
            for i, x in enumerate(fams)
            for y in fams[i + 1 :]
            if (first1[x] < first1[y]) != (first2[x] < first2[y])
        ]

    group_rows = []
    order_pair_rows = []
    t2 = []
    for ms in sorted(by_group):
        orders = by_group[ms]
        if len(orders) < 2:
            continue
        order_pairs = [
            (o1, o2)
            for o1, o2 in combinations(sorted(orders), 2)
            if clan_architecture(o1) != clan_architecture(o2)
            and swapped_family_pairs(o1, o2)
        ]
        if not order_pairs:
            continue
        n_pairs = sum(len(orders[o1]) * len(orders[o2]) for o1, o2 in order_pairs)
        for o1, o2 in order_pairs:
            order_pair_rows.append(
                {
                    "family_multiset": ms,
                    "order_1": o1,
                    "order_2": o2,
                    "n_proteins_order_1": len(orders[o1]),
                    "n_proteins_order_2": len(orders[o2]),
                    "n_t2_pairs": len(orders[o1]) * len(orders[o2]),
                    "swapped_family_pairs": swapped_family_pairs(o1, o2),
                }
            )
        group_rows.append(
            {
                "family_multiset": ms,
                "n_orders": len(orders),
                "n_proteins": sum(len(v) for v in orders.values()),
                "n_order_pairs_differing_by_clan": len(order_pairs),
                "n_t2_pairs": n_pairs,
            }
        )
        rng.shuffle(order_pairs)
        picked: dict[tuple[str, str], None] = {}  # insertion-ordered, so runs repeat
        attempts = 0
        while (
            len(picked) < min(args.per_group, n_pairs)
            and attempts < 20 * args.per_group
        ):
            o1, o2 = order_pairs[attempts % len(order_pairs)]
            picked[(rng.choice(orders[o1]), rng.choice(orders[o2]))] = None
            attempts += 1
        for a, b in picked:
            t2.append(
                {
                    "pair_set": "T2",
                    "accession_a": a,
                    "accession_b": b,
                    "family_multiset": ms,
                }
            )
    pl.DataFrame(order_pair_rows).write_parquet(data / "t2_order_pairs.parquet")
    groups = pl.DataFrame(group_rows)
    groups.write_parquet(data / "t2_all_pair_counts_by_group.parquet")
    counts.append(("t2_family_groups_with_a_swapped_pair_of_clans", groups.height))
    counts.append(("t2_pairs_all", int(groups["n_t2_pairs"].sum())))
    counts.append(("t2_pairs_after_per_group_cap", len(t2)))
    rng.shuffle(t2)
    t2 = t2[: args.max_pairs]
    counts.append(("t2_pairs_sampled", len(t2)))
    pl.DataFrame(t2).write_parquet(data / "t2_pairs.parquet")

    # N1 pool: same architecture, different proteins.
    same_order = [
        v for orders in by_group.values() for v in orders.values() if len(v) >= 2
    ]
    weights = [len(v) * (len(v) - 1) // 2 for v in same_order]
    counts.append(("n1_same_order_pairs_all", sum(weights)))
    pool = set()
    target = args.n1_pool_factor * len(t2)
    # Draw architectures in proportion to their number of pairs, but cap each architecture
    # at --per-group pairs so a few huge families do not fill the pool.
    per_arch: Counter = Counter()
    order_idx = list(range(len(same_order)))
    tries = 0
    while len(pool) < target and tries < 50 * target:
        tries += 1
        i = rng.choices(order_idx, weights=weights)[0]
        if per_arch[i] >= args.per_group:
            continue
        a, b = rng.sample(same_order[i], 2)
        if (a, b) in pool or (b, a) in pool:
            continue
        pool.add((a, b))
        per_arch[i] += 1
    n1_pool = pl.DataFrame(
        [
            {"pair_set": "N1_pool", "accession_a": a, "accession_b": b}
            for a, b in sorted(pool)
        ]
    ).join(
        arch.select(pl.col("accession").alias("accession_a"), "family_multiset"),
        on="accession_a",
    )
    n1_pool.write_parquet(data / "n1_pool.parquet")
    counts.append(("n1_pool_pairs", n1_pool.height))
    pl.DataFrame(counts, schema=["step", "n"], orient="row").write_csv(
        data / "t2_counts.tsv", separator="\t"
    )
    for name, n in counts:
        print(f"{name}\t{n}")


if __name__ == "__main__":
    main()
