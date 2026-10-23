"""For sequences with no exact 31-mer in oyster_v9 transcripts: is there a
near copy (a different allele of a known transcript) or nothing close?

Added after the 1M results (not one of the rules fixed before results):
breed and wild are different populations, so a sequence can miss every exact
31-mer of the reference only because it carries a SNP or small indel.
BLASTN (blastn task, word size 11, E <= 1e-3) of each such sequence against
oyster_v9 transcripts. Best hit per sequence:
  near_copy        identity >= 90% over >= 90% of the sequence's length
  partial_match    any other hit
  no_close_match   no hit at E <= 1e-3
Writes near_reference.tsv (seq_id, class, best transcript, gene, identity,
mismatches, gaps, aligned length, query length, E-value, bit score).
"""
import argparse
import csv
import gzip
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--rna", required=True)
    p.add_argument("--feature-table", required=True)
    p.add_argument("--blastdb", required=True, help="prefix of a makeblastdb nucleotide database of --rna")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    w = Path(a.work)
    grades = pd.read_csv(w / "grades.tsv", sep="\t", dtype=str)
    absent = set(grades.loc[(grades.level == "sequence") & (grades.status == "absent"), "id"])
    seqs = []
    for name in ("splash_sequences.tsv", "dekupl_sequences.tsv"):
        df = pd.read_csv(w / name, sep="\t", dtype=str)
        seqs.append(df.loc[df.seq_id.isin(absent), ["seq_id", "dna"]])
    seqs = pd.concat(seqs)
    gene = {}
    with gzip.open(a.feature_table, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["product_accession"] and row["GeneID"] and row["# feature"] != "CDS":
                gene[row["product_accession"]] = row["GeneID"]
    with tempfile.TemporaryDirectory() as tmp:
        q = Path(tmp) / "q.fa"
        q.write_text("".join(f">{r.seq_id}\n{r.dna}\n" for r in seqs.itertuples()))
        out = Path(tmp) / "hits.tsv"
        subprocess.run(["blastn", "-task", "blastn", "-word_size", "11", "-evalue", "1e-3",
                        "-query", str(q), "-db", a.blastdb, "-num_threads", str(a.threads),
                        "-max_target_seqs", "5", "-outfmt",
                        "6 qseqid sseqid pident length mismatch gapopen qlen evalue bitscore",
                        "-out", str(out)], check=True)
        cols = ["seq_id", "transcript", "identity", "aligned_length", "mismatches", "gap_opens",
                "query_length", "evalue", "bitscore"]
        hits = pd.read_csv(out, sep="\t", names=cols) if out.stat().st_size else pd.DataFrame(columns=cols)
    # makeblastdb -parse_seqids reports NCBI accessions as "ref|XM_...|".
    hits["transcript"] = hits["transcript"].astype(str).str.replace(r"^ref\|(.+)\|$", r"\1", regex=True)
    hits = hits.sort_values(["seq_id", "bitscore"], ascending=[True, False]).drop_duplicates("seq_id")
    res = seqs[["seq_id"]].merge(hits, on="seq_id", how="left")
    res["gene"] = res["transcript"].map(lambda t: gene.get(t, f"transcript:{t}") if isinstance(t, str) else "")
    cover = res["aligned_length"] / res["query_length"]
    res["class"] = "no_close_match"
    res.loc[res["transcript"].notna(), "class"] = "partial_match"
    res.loc[(res["identity"] >= 90) & (cover >= 0.9), "class"] = "near_copy"
    res.to_csv(w / "near_reference.tsv", sep="\t", index=False)
    print(res["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
