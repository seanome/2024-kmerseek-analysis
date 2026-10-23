"""Grade sequences against oyster_v9 transcripts by exact 31-mer lookup.

The reference is used only here, for grading. Each input sequence (SPLASH
compactor, DE-kupl contig) is cut into canonical 31-mers; each 31-mer is
looked up in every oyster_v9 transcript, and the transcript's gene comes from
the NCBI feature table (any RNA feature: product_accession -> GeneID).

Per sequence:
  absent      no 31-mer occurs in any oyster_v9 transcript
  strict      >= 2 distinct 31-mers that occur in exactly one gene, all in the
              same gene, and no 31-mer unique to any other gene
  multi_gene  31-mers unique to two or more genes, or every matching 31-mer
              is shared by several genes (paralogs, repeats)
  weak        one gene, but fewer than 2 distinct 31-mers unique to it
Input TSV needs columns seq_id and dna; group_id (optional) grades a group of
sequences (all compactors of one SPLASH anchor) by the union of their 31-mers.
"""
import argparse
import csv
import gzip
import json
import time
from collections import Counter, defaultdict

K = 31
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s.translate(COMP)[::-1]


def kmers(seq):
    seq = seq.upper()
    for i in range(len(seq) - K + 1):
        k = seq[i:i + K]
        if set(k) <= set("ACGT"):
            yield min(k, rc(k))


def read_fasta(path):
    name, chunks = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip().upper())
    if name:
        yield name, "".join(chunks)


def transcript_genes(feature_table):
    mapping = {}
    with gzip.open(feature_table, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["product_accession"] and row["GeneID"] and row["# feature"] != "CDS":
                mapping[row["product_accession"]] = row["GeneID"]
    return mapping


def classify(kms, kmer_genes):
    hits = [k for k in kms if k in kmer_genes]
    unique = Counter()
    for k in set(hits):
        genes = kmer_genes[k]
        if len(genes) == 1:
            unique[next(iter(genes))] += 1
    all_genes = sorted({g for k in set(hits) for g in kmer_genes[k]})
    if not hits:
        status, gene = "absent", ""
    elif len(unique) == 1 and next(iter(unique.values())) >= 2:
        status, gene = "strict", next(iter(unique))
    elif len(unique) >= 2 or (not unique and len(all_genes) >= 2):
        status, gene = "multi_gene", ""
    else:
        status, gene = "weak", ""
    return dict(status=status, gene=gene, kmers=len(set(kms)), kmers_in_reference=len(set(hits)),
                unique_kmers_per_gene=dict(unique.most_common()), genes_hit=all_genes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", required=True)
    p.add_argument("--rna", required=True)
    p.add_argument("--feature-table", required=True)
    p.add_argument("--out", required=True, help="TSV, one row per seq_id and per group_id")
    a = p.parse_args()
    start = time.monotonic()
    seqs, groups = {}, defaultdict(list)
    for path in a.sequences:
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                seqs[row["seq_id"]] = row["dna"]
                if row.get("group_id"):
                    groups[row["group_id"]].append(row["seq_id"])
    wanted = {}
    for dna in seqs.values():
        for k in kmers(dna):
            wanted[k] = k
            wanted[rc(k)] = k
    print(f"{len(seqs)} sequences, {len(wanted) // 2} canonical 31-mers to look up", flush=True)
    tx_gene = transcript_genes(a.feature_table)
    kmer_genes = defaultdict(set)
    n_tx = no_gene = 0
    for name, seq in read_fasta(a.rna):
        n_tx += 1
        gene = tx_gene.get(name)
        if gene is None:
            no_gene += 1
            gene = "transcript:" + name
        for i in range(len(seq) - K + 1):
            canon = wanted.get(seq[i:i + K])
            if canon is not None:
                kmer_genes[canon].add(gene)
    print(f"scanned {n_tx} transcripts ({no_gene} without a GeneID)", flush=True)
    with open(a.out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "level", "status", "gene", "kmers", "kmers_in_reference",
                    "unique_kmers_per_gene", "genes_hit"])
        for sid, dna in seqs.items():
            c = classify(list(kmers(dna)), kmer_genes)
            w.writerow([sid, "sequence", c["status"], c["gene"], c["kmers"], c["kmers_in_reference"],
                        json.dumps(c["unique_kmers_per_gene"]), ";".join(c["genes_hit"])])
        for gid, members in groups.items():
            c = classify([k for m in members for k in kmers(seqs[m])], kmer_genes)
            w.writerow([gid, "group", c["status"], c["gene"], c["kmers"], c["kmers_in_reference"],
                        json.dumps(c["unique_kmers_per_gene"]), ";".join(c["genes_hit"])])
    print(json.dumps(dict(sequences=len(seqs), groups=len(groups), transcripts=n_tx,
                          transcripts_without_geneid=no_gene,
                          seconds=round(time.monotonic() - start, 1))), flush=True)


if __name__ == "__main__":
    main()
