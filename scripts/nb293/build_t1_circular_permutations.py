"""Build T1: PermuStructDB circular-permutation pairs mapped to UniProt with SIFTS.

Inputs (all under DATA_DIR/downloads, see notebook 293 for URLs, dates and md5):
    permustructdb/Dataset 2--TM-Value ... of this work.txt   20,801 chain pairs
    sifts/pdb_chain_uniprot.tsv.gz                             PDB chain -> UniProt segments
    pdb_seqres/pdb_seqres.txt.gz                               chain SEQRES lengths
    uniprot_sequences.fasta                                    written by fetch_uniprot_sequences.py

Outputs in DATA_DIR:
    t1_chain_pairs_mapped.parquet   one row per PermuStructDB chain pair, with every filter flag
    t1_filter_counts.tsv            how many pairs survive each filter, in order
    t1_pairs.parquet                one row per unordered UniProt accession pair that survives

Filters, applied in order:
    1. both chains present in SIFTS
    2. each chain maps to exactly one UniProt accession
    3. the SIFTS segments cover at least MIN_CHAIN_COVERAGE of the chain's SEQRES length
    4. along each chain, the SIFTS segments run in the same order as along its UniProt
       entry. A chain whose segments are out of order (for example 4zf4_B, GFP residues
       51-237 then 4-50) is an engineered circular permutation of its own UniProt entry:
       the permutation is in the construct and not in any UniProt sequence.
    5. both UniProt sequences were retrieved, and each is at least as long as its
       highest mapped UniProt position (the UniProt sequence covers the chain)
    6. the two chains map to different UniProt accessions. When both chains map to the
       same accession, the two UniProt sequences are identical, so there is nothing for a
       sequence method to find.
    7. one row per unordered accession pair (the chain pair with the highest TM-align-CP
       score is kept)

Usage:
    python3 -I build_t1_circular_permutations.py DATA_DIR [--accessions-only]
With --accessions-only, stops after filter 3 and writes t1_accessions_to_fetch.txt.
"""

import gzip
import sys
from pathlib import Path

import polars as pl

MIN_CHAIN_COVERAGE = 0.7  # SEQRES includes tags, linkers and unobserved residues; median chain coverage of mapped T1 chains is 0.75
PERMUSTRUCTDB_PAIRS = (
    "permustructdb/Dataset 2--TM-Value for Circular permutation pairs by TMalign and "
    "TMalign-CP of this work.txt"
)


def read_fasta(path: Path) -> dict[str, str]:
    """Return {first header word: sequence}; UniProt headers sp|ACC|NAME give ACC."""
    seqs: dict[str, list[str]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    name = None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                word = line[1:].split()[0]
                name = word.split("|")[1] if "|" in word else word
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def seqres_lengths(path: Path) -> pl.DataFrame:
    """Chain SEQRES lengths from pdb_seqres.txt.gz headers (protein chains only)."""
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">") and "mol:protein" in line:
                chain, _, length = line[1:].split()[:3]
                rows.append((chain, int(length.split(":")[1])))
    return pl.DataFrame(rows, schema=["chain", "chain_seqres_length"], orient="row")


def main() -> None:
    data_dir = Path(sys.argv[1])
    downloads = data_dir / "downloads"
    accessions_only = "--accessions-only" in sys.argv

    pairs = pl.read_csv(
        downloads / PERMUSTRUCTDB_PAIRS,
        separator=" ",
        has_header=False,
        new_columns=["chain_a", "chain_b", "tm_align", "tm_align_cp"],
    ).with_columns(
        pl.col("chain_a").str.extract(r"download/(\w+)\.pdb"),
        pl.col("chain_b").str.extract(r"download/(\w+)\.pdb"),
    )

    sifts = pl.read_csv(
        downloads / "sifts/pdb_chain_uniprot.tsv.gz",
        separator="\t",
        skip_rows=1,
        schema_overrides={"PDB": pl.Utf8, "CHAIN": pl.Utf8, "SP_PRIMARY": pl.Utf8},
        infer_schema_length=0,
    ).with_columns(
        (pl.col("PDB") + "_" + pl.col("CHAIN")).alias("chain"),
        *[
            pl.col(c).str.extract(r"(-?\d+)").cast(pl.Int64)
            for c in ["RES_BEG", "RES_END", "SP_BEG", "SP_END"]
        ],
    )
    per_chain = (
        sifts.group_by("chain")
        .agg(
            pl.col("SP_PRIMARY").n_unique().alias("n_accessions"),
            pl.len().alias("n_segments"),
            pl.col("SP_PRIMARY").first().alias("accession"),
            (pl.col("RES_END") - pl.col("RES_BEG") + 1).sum().alias("mapped_residues"),
            pl.col("SP_BEG").min().alias("min_uniprot_position"),
            pl.col("SP_END").max().alias("max_uniprot_position"),
            pl.col("SP_BEG")
            .sort_by("RES_BEG")
            .diff()
            .drop_nulls()
            .min()
            .alias("min_sp_step"),
        )
        .with_columns(
            pl.col("min_sp_step").fill_null(1).gt(0).alias("segments_in_uniprot_order")
        )
        .join(
            seqres_lengths(downloads / "pdb_seqres/pdb_seqres.txt.gz"),
            on="chain",
            how="left",
        )
        .with_columns(
            (pl.col("mapped_residues") / pl.col("chain_seqres_length")).alias(
                "chain_coverage"
            )
        )
    )

    df = pairs
    for side in ["a", "b"]:
        df = df.join(
            per_chain.rename({c: f"{c}_{side}" for c in per_chain.columns}),
            on=f"chain_{side}",
            how="left",
        )
    df = df.with_columns(
        (
            pl.col("accession_a").is_not_null() & pl.col("accession_b").is_not_null()
        ).alias("f1_both_in_sifts"),
        ((pl.col("n_accessions_a") == 1) & (pl.col("n_accessions_b") == 1)).alias(
            "f2_one_accession_each"
        ),
        (
            (pl.col("chain_coverage_a") >= MIN_CHAIN_COVERAGE)
            & (pl.col("chain_coverage_b") >= MIN_CHAIN_COVERAGE)
        ).alias("f3_sifts_covers_chain"),
        (
            pl.col("segments_in_uniprot_order_a")
            & pl.col("segments_in_uniprot_order_b")
        ).alias("f4_chain_order_matches_uniprot"),
    )
    pass3 = df.filter(
        "f1_both_in_sifts", "f2_one_accession_each", "f3_sifts_covers_chain"
    )
    if accessions_only:
        accs = sorted(set(pass3["accession_a"]) | set(pass3["accession_b"]))
        (data_dir / "t1_accessions_to_fetch.txt").write_text("\n".join(accs) + "\n")
        print(f"{len(accs)} accessions to fetch")
        return

    seqs = read_fasta(data_dir / "uniprot_sequences.fasta")
    lengths = pl.DataFrame(
        {"accession": list(seqs), "uniprot_length": [len(s) for s in seqs.values()]}
    )
    for side in ["a", "b"]:
        df = df.join(
            lengths.rename({c: f"{c}_{side}" for c in lengths.columns}),
            on=f"accession_{side}",
            how="left",
        )
    df = df.with_columns(
        (
            (pl.col("uniprot_length_a") >= pl.col("max_uniprot_position_a"))
            & (pl.col("uniprot_length_b") >= pl.col("max_uniprot_position_b"))
        )
        .fill_null(False)
        .alias("f5_uniprot_covers_chain"),
        (pl.col("accession_a") != pl.col("accession_b")).alias(
            "f6_different_accessions"
        ),
    )
    df.write_parquet(data_dir / "t1_chain_pairs_mapped.parquet")

    flags = [
        "f1_both_in_sifts",
        "f2_one_accession_each",
        "f3_sifts_covers_chain",
        "f4_chain_order_matches_uniprot",
        "f5_uniprot_covers_chain",
        "f6_different_accessions",
    ]
    counts = [("f0_permustructdb_pairs", df.height)]
    kept = df
    for flag in flags:
        kept = kept.filter(pl.col(flag).fill_null(False))
        counts.append((flag, kept.height))
    t1 = (
        kept.with_columns(
            pl.min_horizontal("accession_a", "accession_b").alias("acc_lo"),
            pl.max_horizontal("accession_a", "accession_b").alias("acc_hi"),
        )
        .sort(["tm_align_cp", "chain_a", "chain_b"], descending=[True, False, False])
        .unique(subset=["acc_lo", "acc_hi"], keep="first", maintain_order=True)
        .drop("acc_lo", "acc_hi")
        .with_columns(pl.lit("T1").alias("pair_set"))
    )
    counts.append(("f7_unique_accession_pairs", t1.height))
    t1.write_parquet(data_dir / "t1_pairs.parquet")
    pl.DataFrame(counts, schema=["filter", "n_pairs"], orient="row").write_csv(
        data_dir / "t1_filter_counts.tsv", separator="\t"
    )
    for name, n in counts:
        print(f"{name}\t{n}")


if __name__ == "__main__":
    main()
