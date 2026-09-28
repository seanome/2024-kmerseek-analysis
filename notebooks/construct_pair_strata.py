#!/usr/bin/env python3
"""
Stratify the Pfam subdomain benchmark's positive (label=True) pairs into
'hard' and 'easy' strata, based on whether DIAMOND --ultra-sensitive at
E<=10 finds any HSP for the pair at all.

  easy — DIAMOND found >=1 HSP (sanity control: the relationship IS
         detectable by a fast local aligner)
  hard — DIAMOND found nothing (the capability-claim stratum: does
         kmerseek localize a shared domain in pairs where no sequence
         method finds the relationship at all?)

Only positive (shared-domain) pairs are stratified — negative pairs share
no Pfam domain, so there is no relationship for DIAMOND to find or miss.

DIAMOND here is used ONLY as a hard/easy detector, never as an identity
source. Global %identity is computed via a full Needleman-Wunsch global
alignment (Bio.Align.PairwiseAligner, mode="global", BLOSUM62, gap
open/extend -11/-1) on every positive pair, run in parallel.

20% of each (species, stratum) group is held out (seed=42, matching
construct_subdomain_pairs.py's convention) and flagged is_holdout=True.
Nothing downstream should touch holdout rows until final numbers.

Prerequisite: fresh DIAMOND --ultra-sensitive (E<=10) results per species
at --diamond-dir (see nextflow-runs/pfam-subdomain-hard/Makefile). The
pfam-benchmark-tools pipeline's own DIAMOND output is NOT usable here —
verified empty (20-byte stub) for 6/9 species and missing for the other 2,
a container-execution bug in that older run, not a biological result.

Usage:
    python construct_pair_strata.py --species all
    python construct_pair_strata.py --species mouse --force
"""

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from Bio.Align import PairwiseAligner, substitution_matrices

sys.path.insert(0, str(Path(__file__).parent))
from build_pfam_architectures import SPECIES_METADATA, extract_accession

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.20
DIAMOND_EVALUE = 10.0


def load_diamond_hits(diamond_gz: Path) -> pl.DataFrame:
    """Best (max bitscore / min evalue) HSP per (human_accession, species_accession)."""
    df = pl.read_csv(
        diamond_gz, separator="\t", has_header=False,
        new_columns=["query_raw", "target_raw", "bitscore", "evalue"],
        infer_schema_length=100_000,
    )
    df = df.with_columns([
        pl.col("query_raw").map_elements(extract_accession, return_dtype=pl.String).alias("human_accession"),
        pl.col("target_raw").map_elements(extract_accession, return_dtype=pl.String).alias("species_accession"),
    ])
    return (
        df.group_by(["human_accession", "species_accession"])
        .agg([
            pl.col("bitscore").max().alias("diamond_best_bitscore"),
            pl.col("evalue").min().alias("diamond_best_evalue"),
        ])
    )


def load_sequences(species: str, qfo_dir: Path) -> dict:
    """accession -> sequence string, from the QfO FASTA."""
    meta = SPECIES_METADATA[species]
    fasta = qfo_dir / meta["qfo_subdir"] / f"{meta['qfo_proteome']}_{meta['taxon_id']}.fasta"
    seqs: dict[str, str] = {}
    acc = None
    parts: list[str] = []
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(parts)
                acc = extract_accession(line)
                parts = []
            else:
                parts.append(line.strip())
        if acc is not None:
            seqs[acc] = "".join(parts)
    return seqs


# ---------------------------------------------------------------------------
# NW alignment (module-level for multiprocessing pickling; one aligner built
# per worker process, cached in that process's globals)
# ---------------------------------------------------------------------------

_ALIGNER = None
_ALPHABET = None


def _get_aligner() -> PairwiseAligner:
    global _ALIGNER, _ALPHABET
    if _ALIGNER is None:
        matrix = substitution_matrices.load("BLOSUM62")
        a = PairwiseAligner()
        a.substitution_matrix = matrix
        a.mode = "global"
        a.open_gap_score = -11
        a.extend_gap_score = -1
        _ALIGNER = a
        _ALPHABET = set(matrix.alphabet)
    return _ALIGNER


def _sanitize(seq: str) -> str:
    """Replace any residue outside BLOSUM62's alphabet (e.g. U/O for
    selenocysteine/pyrrolysine, rare in real UniProt sequences) with 'X'."""
    return "".join(c if c in _ALPHABET else "X" for c in seq)


def nw_identity(task: tuple) -> tuple:
    h_acc, s_acc, h_seq, s_seq = task
    if not h_seq or not s_seq:
        return h_acc, s_acc, None, None, None
    aligner = _get_aligner()
    h_seq, s_seq = _sanitize(h_seq), _sanitize(s_seq)
    aln = aligner.align(h_seq, s_seq)[0]
    a1, a2 = str(aln[0]), str(aln[1])
    aln_len = len(a1)
    matches = sum(1 for x, y in zip(a1, a2) if x == y and x != "-")
    identity = matches / aln_len if aln_len else 0.0
    return h_acc, s_acc, identity, float(aln.score), aln_len


# ---------------------------------------------------------------------------

def process_species(
    species: str,
    pairs_dir: Path,
    diamond_dir: Path,
    qfo_dir: Path,
    strata_dir: Path,
    n_workers: int,
    force: bool = False,
) -> dict | None:
    out_path = strata_dir / f"human_vs_{species}_hard_easy.parquet"
    if out_path.exists() and not force:
        print(f"  {species}: already done — skipping (use --force)")
        return None

    print(f"\n=== {species} ===")
    gt = pl.read_parquet(pairs_dir / f"human_vs_{species}_ground_truth.parquet")
    pos = gt.filter(pl.col("label")).select([
        "human_accession", "species_accession", "shared_pfam_ids", "n_shared_domains",
        "human_protein_length", "species_protein_length",
    ])
    print(f"  positive (shared-domain) pairs: {len(pos):,}")

    diamond_gz = diamond_dir / f"human_vs_{species}.diamond.tsv.gz"
    if not diamond_gz.exists():
        print(f"  SKIPPING: missing {diamond_gz}", file=sys.stderr)
        return None
    hits = load_diamond_hits(diamond_gz)
    print(f"  DIAMOND ultra-sensitive hit pairs (E<={DIAMOND_EVALUE}): {len(hits):,}")

    merged = pos.join(hits, on=["human_accession", "species_accession"], how="left")
    merged = merged.with_columns(
        pl.when(pl.col("diamond_best_evalue").is_not_null())
        .then(pl.lit("easy")).otherwise(pl.lit("hard")).alias("stratum")
    )
    n_hard = merged.filter(pl.col("stratum") == "hard").height
    n_easy = merged.filter(pl.col("stratum") == "easy").height
    print(f"  N_HARD={n_hard:,}   N_EASY={n_easy:,}")

    # ---- Full NW global identity on every positive pair ----
    h_seqs = load_sequences("human", qfo_dir)
    s_seqs = load_sequences(species, qfo_dir)
    tasks = [
        (row["human_accession"], row["species_accession"],
         h_seqs.get(row["human_accession"], ""), s_seqs.get(row["species_accession"], ""))
        for row in merged.iter_rows(named=True)
    ]
    n_missing_seq = sum(1 for t in tasks if not t[2] or not t[3])
    if n_missing_seq:
        print(f"  WARNING: {n_missing_seq} pairs missing a FASTA sequence — nw_identity=null for those", file=sys.stderr)

    print(f"  Computing full NW global alignment for {len(tasks):,} pairs ({n_workers} workers)...")
    nw_by_pair = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for h_acc, s_acc, identity, score, aln_len in ex.map(nw_identity, tasks, chunksize=50):
            nw_by_pair[(h_acc, s_acc)] = (identity, score, aln_len)

    identities, scores, aln_lens = [], [], []
    for row in merged.iter_rows(named=True):
        v = nw_by_pair[(row["human_accession"], row["species_accession"])]
        identities.append(v[0]); scores.append(v[1]); aln_lens.append(v[2])
    merged = merged.with_columns([
        pl.Series("nw_identity", identities, dtype=pl.Float64),
        pl.Series("nw_score", scores, dtype=pl.Float64),
        pl.Series("nw_alignment_length", aln_lens, dtype=pl.Int32),
    ])

    # ---- 20% holdout, stratified by (stratum), seeded ----
    rng = random.Random(RANDOM_SEED)
    holdout_flags = [False] * merged.height
    idx_by_stratum: dict[str, list[int]] = {}
    for i, s in enumerate(merged["stratum"]):
        idx_by_stratum.setdefault(s, []).append(i)
    for s, idxs in idx_by_stratum.items():
        idxs = sorted(idxs)
        rng.shuffle(idxs)
        n_hold = round(len(idxs) * HOLDOUT_FRACTION)
        for i in idxs[:n_hold]:
            holdout_flags[i] = True
    merged = merged.with_columns(pl.Series("is_holdout", holdout_flags))

    strata_dir.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="snappy")
    print(f"  Saved: {out_path}")

    summary = {
        "species": species,
        "n_positive_pairs": merged.height,
        "n_hard": n_hard,
        "n_easy": n_easy,
        "n_hard_holdout": merged.filter((pl.col("stratum") == "hard") & pl.col("is_holdout")).height,
        "n_easy_holdout": merged.filter((pl.col("stratum") == "easy") & pl.col("is_holdout")).height,
        "median_nw_identity_hard": merged.filter(pl.col("stratum") == "hard")["nw_identity"].median(),
        "median_nw_identity_easy": merged.filter(pl.col("stratum") == "easy")["nw_identity"].median(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", nargs="+", default=["all"])
    parser.add_argument("--pairs-dir", type=Path, default=Path("results/pfam_benchmark/pairs"))
    parser.add_argument("--diamond-dir", type=Path, default=Path("results/pfam_benchmark/diamond_fresh"))
    parser.add_argument(
        "--qfo-dir", type=Path,
        default=Path.home() / "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143",
    )
    parser.add_argument("--strata-dir", type=Path, default=Path("results/pfam_benchmark/pairs_stratified"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    non_human = [s for s in SPECIES_METADATA if s != "human"]
    species_list = non_human if args.species == ["all"] else [s for s in args.species if s != "human"]

    summaries = []
    for sp in species_list:
        s = process_species(sp, args.pairs_dir, args.diamond_dir, args.qfo_dir, args.strata_dir, args.workers, force=args.force)
        if s:
            summaries.append(s)

    if summaries:
        total_hard = sum(s["n_hard"] for s in summaries)
        total_easy = sum(s["n_easy"] for s in summaries)
        total_hard_holdout = sum(s["n_hard_holdout"] for s in summaries)
        total_easy_holdout = sum(s["n_easy_holdout"] for s in summaries)
        print(f"\n=== TOTAL across {len(summaries)} species ===")
        print(f"  N_HARD={total_hard:,}  N_EASY={total_easy:,}")
        print(f"  N_HARD_HOLDOUT={total_hard_holdout:,}  N_EASY_HOLDOUT={total_easy_holdout:,}")

        summary_path = args.strata_dir / "strata_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "per_species": summaries,
                "total_n_hard": total_hard,
                "total_n_easy": total_easy,
                "total_n_hard_holdout": total_hard_holdout,
                "total_n_easy_holdout": total_easy_holdout,
                "holdout_fraction": HOLDOUT_FRACTION,
                "random_seed": RANDOM_SEED,
                "diamond_evalue_threshold": DIAMOND_EVALUE,
            }, f, indent=2)
        print(f"  Saved: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
