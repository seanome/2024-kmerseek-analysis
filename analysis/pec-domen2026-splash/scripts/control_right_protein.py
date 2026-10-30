#!/usr/bin/env python3
"""Known-answer control: does a changed sequence's peptide find the right protein?

Usage: control_right_protein.py <control_transcripts.fa> <truth_design.tsv>
                                <transcript_align.parquet> <control_contrast_dir> <out_dir>

Right protein of a transcript: its supported new-proteome match with the smallest E-value, if
E <= 1e-10 (transcript_align.parquet, from the transcript's longest stop-free peptide).
A gene's right proteins are those of its transcripts (an isoform pair usually shares one).

Each sequence a channel reports (SPLASH compactors in search_compactors/, DE-kupl contigs in
search_dekupl_limma/) is placed in the gene(s) holding the most of its 31-mers, either strand.
Its answer is the target of its best supported peptide match (smallest E) in the new proteome.
Outcome per sequence:
  right                      answer is a right protein of its gene (or of one of its tied genes)
  wrong                      answer is another protein
  no_supported_match         no peptide of the sequence has a supported match
  gene_without_right_protein the gene's transcript found no protein at E <= 1e-10
  not_placed                 no 31-mer of the sequence is in any control transcript
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

K = 31
COMP = str.maketrans("ACGT", "TGCA")


def read_fasta(p):
    seqs, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        else:
            seqs[name].append(line.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def gene_of(t):
    return t.rsplit("_iso", 1)[0] if "_iso" in t else t


def main():
    fa, truth_p, tx_align, cdir, out = sys.argv[1:6]
    cdir, out = Path(cdir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tx = read_fasta(fa)
    truth = pl.read_csv(truth_p, separator="\t")
    gene_class = {gene_of(t): c for t, c in zip(truth["transcript"], truth["class"])}

    a = pl.read_parquet(tx_align)
    right_tx = (a.filter((pl.col("supported") == True) & (pl.col("evalue") <= 1e-10))
                .sort(["query", "evalue", "target"]).group_by("query", maintain_order=True).head(1))
    right = defaultdict(set)
    tx_right = {}
    for q, t, pid in zip(right_tx["query"], right_tx["target"], right_tx["pct_identity"]):
        right[gene_of(q)].add(t)
        tx_right[q] = (t, pid)
    summary = dict(transcripts=len(tx), transcripts_with_right_protein=len(tx_right),
                   genes_with_right_protein=len(right),
                   right_protein_identity_quantiles={str(p): float(right_tx["pct_identity"].quantile(p))
                                                     for p in (0.05, 0.25, 0.5, 0.75)} if right_tx.height else {},
                   right_proteins_shared_by_two_or_more_genes=sum(
                       1 for v in Counter(t for s in right.values() for t in s).values() if v > 1),
                   channels={})

    index = defaultdict(set)
    for t, s in tx.items():
        g = gene_of(t)
        for seq in (s, s.translate(COMP)[::-1]):
            for i in range(len(seq) - K + 1):
                index[seq[i:i + K]].add(g)

    def place(seq):
        seq = seq.upper()
        c = Counter()
        for i in range(len(seq) - K + 1):
            for g in index.get(seq[i:i + K], ()):
                c[g] += 1
        if not c:
            return set()
        m = max(c.values())
        return {g for g, n in c.items() if n == m}

    rows = []
    for channel, folder, seq_col in (("splash_compactors", "search_compactors", "compactor"),
                                     ("dekupl_limma_contigs", "search_dekupl_limma", "contig")):
        d = cdir / folder
        if not (d / "search.done").exists():
            continue
        pep = pl.read_csv(d / "pep.peptides.tsv", separator="\t", infer_schema_length=0)
        al = pl.read_parquet(d / "align.proteome.parquet")
        best = (al.filter(pl.col("supported") == True).join(
                    pep.select(pl.col("peptide_id").alias("query"), "seq_id"), on="query")
                .sort(["seq_id", "evalue", "target"]).group_by("seq_id", maintain_order=True).head(1))
        answer = dict(zip(best["seq_id"], best["target"]))
        seqs = pep.select("seq_id", seq_col).unique("seq_id")
        # sequences with no peptide at all are listed in dna_lengths
        dl = pl.read_csv(d / "pep.dna_lengths.tsv", separator="\t")
        all_ids = dl["seq_id"].to_list()
        seq_of = dict(zip(seqs["seq_id"], seqs[seq_col]))
        if channel == "dekupl_limma_contigs":
            src = pl.read_csv(d / "dekupl_contigs.tsv", separator="\t", columns=["contig"])
            for i, s in enumerate(src["contig"], start=1):
                seq_of.setdefault(f"c{i}", s)
        elif (cdir / "compactors/compactors.tsv").exists():
            src = pl.read_csv(cdir / "compactors/compactors.tsv", separator="\t", columns=["compactor"])
            for i, s in enumerate(src["compactor"], start=1):
                seq_of.setdefault(f"c{i}", s)
        outcome = Counter()
        by_class = defaultdict(Counter)
        for sid in all_ids:
            genes = place(seq_of.get(sid, ""))
            if not genes:
                o, cls = "not_placed", "none"
            else:
                rp = set().union(*(right.get(g, set()) for g in genes))
                cls = "/".join(sorted({gene_class.get(g, "?") for g in genes}))
                if not rp:
                    o = "gene_without_right_protein"
                elif sid not in answer:
                    o = "no_supported_match"
                elif answer[sid] in rp:
                    o = "right"
                else:
                    o = "wrong"
            outcome[o] += 1
            by_class[cls][o] += 1
            rows.append(dict(channel=channel, seq_id=sid, genes=",".join(sorted(genes)), truth_class=cls,
                             answer=answer.get(sid, ""), outcome=o))
        decided = outcome["right"] + outcome["wrong"]
        summary["channels"][channel] = dict(sequences=len(all_ids), outcome=dict(outcome),
                                            right_fraction_of_answered=round(outcome["right"] / decided, 4) if decided else None,
                                            by_truth_class={k: dict(v) for k, v in by_class.items()})
    pl.DataFrame(rows).write_csv(out / "control_right_protein.tsv", separator="\t")
    right_tx.write_csv(out / "control_transcript_right_protein.tsv", separator="\t")
    (out / "control_right_protein_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
