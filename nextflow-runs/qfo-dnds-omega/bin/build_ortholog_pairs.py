#!/opt/conda/bin/python3
"""Build 1:1 human<->target ortholog pairs with codon sequences, straight from the
QfO release files. One species per invocation.

Why the pairing comes from the shipped .idmapping and not from a search:
the downstream consumer regresses benchmark performance on omega, so the ortholog
call must not come from the same k-mer search whose performance is being explained.
Reciprocal-best-hit pairs would make that circular. The QfO release ships UniProt
cross-references to external ortholog databases, which are independent of anything
this repo computes.

OMA is the default source. An OMA group holds at most one protein per species by
construction, so "same group" already means 1:1 orthologs rather than "same family".
Measured on this release, 11_584 of 11_594 shared human-mouse OMA groups are exactly
1:1, and the 10 exceptions are dropped rather than resolved. Ensembl GeneTree is
offered as an alternative because chicken's OMA cross-references are sparse in this
release (5_038 of 17_837 chicken proteins carry one, against 8_559 1:1 GeneTree pairs
versus 3_871 OMA). GeneTree is a weaker claim: a gene tree contains paralogs too, and
"exactly one member per species in this tree" is a proxy for 1:1 orthology, not a
statement of it.

CDS usability is decided by what codeml needs, not by agreement with the protein
FASTA. Each record is translated by this pipeline, so a record whose translation
differs from the UniProt canonical sequence is still a valid CDS. It is kept, and
flagged: `human_cds_exact_match` records whether the human CDS translates to the
exact UniProt sequence the benchmark indexes. It does so for 13_013 of 19_351 usable
human records; among the rest the median identity to the canonical sequence is 0.853,
and 15% fall below 0.5 identity. Anything joining omega onto a human accession and
caring that it refers to the same protein should filter on that flag.

Terminal stop codons are stripped here rather than in the codeml step. Not every
proteome supplies them: 10_211 of 16_674 ciona records carry a terminal stop against
essentially all of mouse's, so a step that assumes the last codon is a stop would
silently truncate two-thirds of ciona's proteins by one real codon.

Output: a manifest TSV, and chunked 2-records-per-pair FASTA files. Chunking exists
because per-pair Nextflow task overhead is on the order of the codeml run itself; at
~15_000 pairs one task per pair spends most of the wall clock on scheduling.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from Bio.Seq import Seq

STOP_CODONS = {"TAA", "TAG", "TGA"}


def read_fasta(path):
    name, buf = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].rstrip("\n"), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def accession(header: str) -> str:
    """QfO headers are UniProt style, `db|ACCESSION|rest`. The DNA FASTA puts an
    Ensembl protein id where the protein FASTA puts an entry name, so the middle
    field is the only part that joins the two files. It does join completely: across
    all ten proteomes every DNA record's accession is present in the protein FASTA,
    with no duplicated accessions on either side."""
    parts = header.split("|")
    return parts[1] if len(parts) >= 2 else header.split()[0]


def load_usable_cds(path: Path, min_codons: int) -> dict[str, str]:
    """Keep records codeml can actually use: in frame, unambiguous bases, and no stop
    codon before the end. A terminal stop is stripped if present and not required."""
    usable = {}
    for header, seq in read_fasta(path):
        seq = seq.upper()
        if len(seq) % 3 != 0:
            continue
        codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
        if codons and codons[-1] in STOP_CODONS:
            codons = codons[:-1]
        if len(codons) < min_codons:
            continue
        if any(c in STOP_CODONS for c in codons):
            continue
        body = "".join(codons)
        if any(base not in "ACGT" for base in body):
            continue
        usable[accession(header)] = body
    return usable


def load_xref(path: Path, database: str, keep: set[str]) -> dict[str, set[str]]:
    """group id -> accessions, restricted to accessions that already have a usable CDS.
    The .idmapping file covers every UniProtKB entry for the taxon, which is far more
    than the reference proteome (75_075 human entries against 20_600 proteome records),
    so restricting here also restricts to the proteome."""
    groups = defaultdict(set)
    with open(path) as handle:
        for line in handle:
            acc, db, value = line.rstrip("\n").split("\t")
            if db == database and acc in keep:
                groups[value].add(acc)
    return groups


def load_gene_names(path: Path, keep: set[str]) -> dict[str, str]:
    """Gene symbol per accession. Kept so the output can be joined either on accession
    (direct, and what the benchmark keys on) or on symbol, which is the route
    build_query_covariates.py currently takes through HGNC."""
    names = {}
    with open(path) as handle:
        for line in handle:
            acc, db, value = line.rstrip("\n").split("\t")
            if db == "Gene_Name" and acc in keep and acc not in names:
                names[acc] = value
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human_dna", required=True, type=Path)
    ap.add_argument("--human_protein", required=True, type=Path)
    ap.add_argument("--human_idmapping", required=True, type=Path)
    ap.add_argument("--target_dna", required=True, type=Path)
    ap.add_argument("--target_idmapping", required=True, type=Path)
    ap.add_argument("--species", required=True)
    ap.add_argument("--mya", type=int, default=0, help="divergence estimate, carried through to output")
    ap.add_argument("--ortholog_source", default="OMA", choices=["OMA", "GeneTree", "OrthoDB", "TreeFam", "eggNOG"])
    ap.add_argument("--min_codons", type=int, default=50)
    ap.add_argument("--chunk_size", type=int, default=250)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 = no cap; >0 truncates for smoke tests")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    human_cds = load_usable_cds(args.human_dna, args.min_codons)
    target_cds = load_usable_cds(args.target_dna, args.min_codons)

    human_protein = {accession(h): s.upper() for h, s in read_fasta(args.human_protein)}
    human_exact = {
        acc
        for acc, cds in human_cds.items()
        if str(Seq(cds).translate()) == human_protein.get(acc)
    }

    human_groups = load_xref(args.human_idmapping, args.ortholog_source, set(human_cds))
    target_groups = load_xref(args.target_idmapping, args.ortholog_source, set(target_cds))
    human_names = load_gene_names(args.human_idmapping, set(human_cds))
    target_names = load_gene_names(args.target_idmapping, set(target_cds))

    shared = set(human_groups) & set(target_groups)
    one_to_one = sorted(
        group
        for group in shared
        if len(human_groups[group]) == 1 and len(target_groups[group]) == 1
    )
    # Counted before any smoke-test truncation, so the summary keeps describing the
    # release rather than describing --max_pairs.
    n_one_to_one = len(one_to_one)
    if args.max_pairs:
        one_to_one = one_to_one[: args.max_pairs]

    if not one_to_one:
        # Nextflow would otherwise report this as a missing-output glob, which says
        # nothing about the cause. ecoli reaches here with every source except KEGG
        # Orthology: the release ships no OrthoDB, OMA, GeneTree or TreeFam
        # cross-references for it, and its eggNOG groups share none with human.
        raise SystemExit(
            f"No 1:1 {args.ortholog_source} ortholog pairs between human and "
            f"{args.species} ({len(shared)} groups shared, none of them 1:1 with a "
            f"usable CDS on both sides). This species cannot be paired from this "
            f"source; it is not a transient failure."
        )

    manifest_path = args.outdir / f"pairs.{args.species}.tsv"
    fields = [
        "pair_id", "species", "mya", "ortholog_source", "ortholog_group",
        "human_accession", "human_gene", "target_accession", "target_gene",
        "human_cds_exact_match", "human_n_codons", "target_n_codons", "chunk",
    ]

    chunk_handle, chunk_index, in_chunk = None, -1, args.chunk_size
    with open(manifest_path, "w") as manifest:
        manifest.write("\t".join(fields) + "\n")
        for group in one_to_one:
            human_acc = next(iter(human_groups[group]))
            target_acc = next(iter(target_groups[group]))
            # The pair id has to survive being a FASTA record name and a filename.
            pair_id = f"{human_acc}__{target_acc}"

            if in_chunk >= args.chunk_size:
                if chunk_handle is not None:
                    chunk_handle.close()
                chunk_index += 1
                chunk_handle = open(args.outdir / f"chunk_{chunk_index:04d}.{args.species}.fa", "w")
                in_chunk = 0
            chunk_handle.write(f">{pair_id}|query\n{human_cds[human_acc]}\n")
            chunk_handle.write(f">{pair_id}|target\n{target_cds[target_acc]}\n")
            in_chunk += 1

            row = [
                pair_id, args.species, str(args.mya), args.ortholog_source, group,
                human_acc, human_names.get(human_acc, ""),
                target_acc, target_names.get(target_acc, ""),
                "true" if human_acc in human_exact else "false",
                str(len(human_cds[human_acc]) // 3),
                str(len(target_cds[target_acc]) // 3),
                f"{chunk_index:04d}",
            ]
            manifest.write("\t".join(row) + "\n")
    if chunk_handle is not None:
        chunk_handle.close()

    summary = dict(
        species=args.species,
        ortholog_source=args.ortholog_source,
        n_human_usable_cds=len(human_cds),
        n_target_usable_cds=len(target_cds),
        n_human_cds_exact_match=len(human_exact),
        n_shared_groups=len(shared),
        n_pairs_1to1=n_one_to_one,
        n_pairs_dropped_not_1to1=len(shared) - n_one_to_one,
        n_pairs_written=len(one_to_one),
        n_chunks=chunk_index + 1,
    )
    (args.outdir / f"summary.{args.species}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
