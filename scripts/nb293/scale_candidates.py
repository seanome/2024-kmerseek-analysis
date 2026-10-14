"""Step 6 of notebook 293: from sourmash candidate pairs to offset-test calls on Swiss-Prot.

Usage:
    python3 -I scale_candidates.py DATA_DIR SWISSPROT_FASTA_GZ MANYSEARCH_CSV
        [--alphabet protein20] [--k 5] [--n-block 3] [--threads 8]

1. Read the sourmash manysearch output (every Swiss-Prot protein searched against all of
   Swiss-Prot, query containment >= the search threshold). Drop self matches, keep one row
   per unordered pair (the larger containment).
2. Align every candidate pair once with MMseqs2 (C1: default settings, one alignment per
   pair, only the listed pairs are aligned) and keep pairs whose best alignment covers
   less than half of the shorter protein, or that have no alignment at E <= 1e-3.
3. Run the offset test (one setting) on those pairs.
4. Label each called pair: in T1 (the accession pair is a T1 pair), in T2 (same Pfam family
   multiset in a different order by the T2 rules, read from t2_order_pairs.parquet), or
   in neither.
Writes DATA_DIR/scale/candidates_<setting>.parquet and scale/stage_times.tsv.
"""

import argparse
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402

MMSEQS = "/Users/olga/anaconda3/envs/orthofinder/bin/mmseqs"
COLUMNS = "query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"


SEQS: dict[str, str] = {}
SETTING_ARGS: tuple = ()


def test_pair(pair: tuple[str, str]) -> dict:
    a, b = pair
    alphabet, k, n_block = SETTING_ARGS
    return {
        "accession_a": a,
        "accession_b": b,
        **ku.offset_test(SEQS[a], SEQS[b], alphabet, k, [n_block])[0],
    }


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("swissprot")
    p.add_argument("manysearch_csv")
    p.add_argument("--alphabet", default="protein20")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-block", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    data = Path(args.data_dir)
    work = data / "scale" / "mmseqs"
    work.mkdir(parents=True, exist_ok=True)
    times = []

    t0 = time.time()
    ms = pl.scan_csv(args.manysearch_csv).select(
        "query_name", "match_name", "containment"
    )
    acc = r"^\w+\|(\w+)\|"
    cand = (
        ms.with_columns(
            pl.col("query_name").str.extract(acc).alias("q"),
            pl.col("match_name").str.extract(acc).alias("m"),
        )
        .filter(pl.col("q") != pl.col("m"))
        .with_columns(
            pl.min_horizontal("q", "m").alias("accession_a"),
            pl.max_horizontal("q", "m").alias("accession_b"),
        )
        .group_by("accession_a", "accession_b")
        .agg(pl.col("containment").max())
        .sort("accession_a", "accession_b")
        .collect()
    )
    times.append(("read_and_deduplicate_candidates", time.time() - t0, cand.height))

    t0 = time.time()
    if not (work / "spdb.dbtype").exists():
        run([MMSEQS, "createdb", args.swissprot, work / "spdb"])
    key = {
        line.split("\t")[1]: line.split("\t")[0]  # MMseqs2 stores the UniProt accession
        for line in (work / "spdb.lookup").read_text().splitlines()
    }
    with open(work / "pairs.tsv", "w") as fh:
        for a, b in cand.select("accession_a", "accession_b").iter_rows():
            fh.write(f"{key[a]}\t{key[b]}\t0\n")
    for f in work.glob("pref*"):
        f.unlink()
    for f in work.glob("aln*"):
        f.unlink()
    run([MMSEQS, "tsv2db", work / "pairs.tsv", work / "pref", "--output-dbtype", "7"])
    run(
        [
            MMSEQS,
            "align",
            work / "spdb",
            work / "spdb",
            work / "pref",
            work / "aln",
            "-a",
            "--threads",
            args.threads,
        ]
    )
    run(
        [
            MMSEQS,
            "convertalis",
            work / "spdb",
            work / "spdb",
            work / "aln",
            work / "aln.tsv",
            "--format-output",
            COLUMNS,
            "--threads",
            args.threads,
        ]
    )
    aln = pl.read_csv(
        work / "aln.tsv",
        separator="\t",
        has_header=False,
        new_columns=COLUMNS.split(","),
    )
    aln = (
        aln.with_columns(
            pl.col("query").alias("accession_a"),
            pl.col("target").alias("accession_b"),
        )
        .sort("accession_a", "accession_b", "evalue")
        .unique(
            subset=["accession_a", "accession_b"], keep="first", maintain_order=True
        )
        .select(
            "accession_a",
            "accession_b",
            pl.col("fident").alias("c1_identity"),
            (
                (pl.col("qend") - pl.col("qstart") + 1)
                / pl.min_horizontal("qlen", "tlen")
            )
            .clip(upper_bound=1.0)
            .alias("c1_coverage_of_shorter"),
        )
    )
    cand = cand.join(aln, on=["accession_a", "accession_b"], how="left").with_columns(
        pl.col("c1_coverage_of_shorter").fill_null(0.0)
    )
    times.append(("mmseqs2_c1_on_candidates", time.time() - t0, cand.height))

    t0 = time.time()
    low = cand.filter(pl.col("c1_coverage_of_shorter") < 0.5)
    seqs = ku.read_fasta(args.swissprot)
    global SEQS, SETTING_ARGS
    SEQS, SETTING_ARGS = seqs, (args.alphabet, args.k, args.n_block)
    pairs = list(low.select("accession_a", "accession_b").iter_rows())
    with get_context("fork").Pool(args.threads) as pool:  # fork: workers inherit SEQS
        rows = pool.map(test_pair, pairs, chunksize=2_000)
    tested = low.join(
        pl.DataFrame(rows, infer_schema_length=None),
        on=["accession_a", "accession_b"],
        how="left",
    )
    times.append(
        ("offset_test_on_low_coverage_candidates", time.time() - t0, tested.height)
    )

    # Labels.
    t1 = (
        pl.read_parquet(data / "t1_pairs.parquet")
        .select(
            pl.min_horizontal("accession_a", "accession_b").alias("accession_a"),
            pl.max_horizontal("accession_a", "accession_b").alias("accession_b"),
        )
        .with_columns(pl.lit(True).alias("in_t1"))
    )
    arch = pl.read_parquet(data / "swissprot_architectures.parquet").select(
        "accession", "architecture"
    )
    orders = pl.read_parquet(data / "t2_order_pairs.parquet")
    t2_order_pairs = {
        (r[0], r[1]) for r in orders.select("order_1", "order_2").iter_rows()
    }
    t2_order_pairs |= {(b, a) for a, b in t2_order_pairs}
    tested = (
        tested.join(t1, on=["accession_a", "accession_b"], how="left")
        .join(
            arch.rename({"accession": "accession_a", "architecture": "architecture_a"}),
            on="accession_a",
            how="left",
        )
        .join(
            arch.rename({"accession": "accession_b", "architecture": "architecture_b"}),
            on="accession_b",
            how="left",
        )
        .with_columns(pl.col("in_t1").fill_null(False))
    )
    tested = tested.with_columns(
        pl.struct("architecture_a", "architecture_b")
        .map_elements(
            lambda r: (r["architecture_a"], r["architecture_b"]) in t2_order_pairs,
            pl.Boolean,
        )
        .fill_null(False)
        .alias("in_t2")
    )
    setting = f"{args.alphabet}_k{args.k}_n{args.n_block}"
    cand.write_parquet(data / "scale" / "candidates_all.parquet")
    tested.write_parquet(data / "scale" / f"candidates_{setting}.parquet")
    pl.DataFrame(times, schema=["stage", "seconds", "n_pairs"], orient="row").write_csv(
        data / "scale" / "stage_times.tsv", separator="\t"
    )
    called = tested.filter("called")
    print(
        f"candidates={cand.height} low_coverage={low.height} called={called.height} "
        f"in_t1={called['in_t1'].sum()} in_t2={called['in_t2'].sum()} "
        f"neither={called.filter(~pl.col('in_t1') & ~pl.col('in_t2')).height}"
    )
    for row in times:
        print(row)


if __name__ == "__main__":
    main()
