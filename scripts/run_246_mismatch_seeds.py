"""Tables for notebook 246: seeds that allow mismatches, on BCL2/CED-9 and P66/CD47.

Two measurements, both with every scheme in ``mismatch_seed_utils.default_schemes()`` and
both two-letter alphabets:

1. Search. CED-9 (C. elegans) and P66 (Borrelia burgdorferi) are each compared, along
   every diagonal, with every protein of the GENCODE v49 canonical human proteome. A
   protein's score under a scheme is the most seed positions on any one of its diagonals
   (for ``extend``, its best segment score). The partner (BCL2 for CED-9, CD47 for P66)
   is ranked among all 19,732 proteins.
2. Reach. On the Pfam seed-alignment pairs of notebook 230, the share of pairs whose own
   alignment diagonal holds at least one seed position, by identity bin (pairs with at
   least 50 aligned positions, as in notebooks 230 and 232). Not defined for ``extend``,
   which has a score and no seed.

Plus the index size each keyed scheme would need on the human proteome.

Run (numba is not in the 2025-kmerseek-analysis env, so pixi supplies it):
    pixi exec --spec numba --spec numpy --spec polars -- python scripts/run_246_mismatch_seeds.py
Inputs can be moved with NB246_PROTEOME and NB246_PFAM_PAIRS.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import mismatch_seed_utils as ms  # noqa: E402

PROTEOME = Path(os.environ.get(
    "NB246_PROTEOME", "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"))
PFAM_PAIRS = Path(os.environ.get(
    "NB246_PFAM_PAIRS", "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"))
OUT = ROOT / "tables"
# The queries notebook 241 searched (PR 48, notebooks/241_queries.fa): CED-9, 280 aa, and
# mature P66, 597 aa (signal peptide removed).
QUERIES = OUT / "246_queries_from_241.fasta"

# query name in QUERIES -> (query label, partner gene in the human proteome)
CASES = {"Ced9": ("CED-9", "BCL2"), "P66": ("P66", "CD47")}
# BH1: BCL2 130-166 against CED-9 154-190 on one gapless diagonal (kmerseek docs Part 2,
# notebook 245). With CED-9 as the query the diagonal is BCL2 position - CED-9 position.
BH1_DIAGONAL = 130 - 154
N_TOP = 10  # top-scoring human proteins kept per search
N_NULL_DRAWS = 20_000
IDENTITY_BINS = [0.2, 0.3, 0.4, 0.6]  # same cut points as hp_conservation_utils
IDENTITY_LABELS = ["<20%", "20-30%", "30-40%", "40-60%", ">=60%"]
MIN_ALIGNED = 50  # aligned positions; notebooks 230 and 232 used the same filter


def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    out.append((name, "".join(buf)))
    return out


def rank_of(scores: np.ndarray, value: float) -> tuple[int, int]:
    """1 + proteins scoring strictly higher, and how many others tie with ``value``."""
    return int((scores > value).sum()) + 1, int((scores == value).sum()) - 1


def scheme_table(schemes: list[ms.Scheme]) -> pl.DataFrame:
    return pl.DataFrame([{"scheme": s.name, "family": s.family, "scaled": s.scaled,
                          "weight": s.weight, "span": s.span, "pattern": s.pattern,
                          "k": s.k, "n_pieces": s.n, "min_match": s.min_match,
                          "penalty": s.penalty} for s in schemes])


def search(schemes, proteome, genes, names) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows, top, diag, null, draws_out = [], [], [], [], []
    queries = dict(read_fasta(QUERIES))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(proteome), N_NULL_DRAWS)
    for stem, (qlabel, partner) in CASES.items():  # stem = the FASTA record name
        qseq = queries[stem]
        p_idx = genes.index(partner)
        for alphabet in ms.ALPHABETS:
            q = ms.encode(qseq, alphabet)
            db, starts = ms.concat_database([ms.encode(s, alphabet) for s in proteome])
            t0 = time.time()
            score, total, pieces = ms.run_database(q, db, starts, schemes)
            secs = time.time() - t0
            cells = float(len(q)) * float(len(db))
            print(f"{qlabel} {alphabet}: {secs:.0f} s", flush=True)
            per_diag = ms.run_pair(q, ms.encode(proteome[p_idx], alphabet), schemes)
            ranks = np.empty((len(proteome), len(schemes)))
            for si, s in enumerate(schemes):
                col = score[:, si]
                pv = col[p_idx]
                rank, tied = rank_of(col, pv)
                others = np.delete(np.arange(len(proteome)), p_idx)
                best_d = int(per_diag[:, si].argmax())
                rows.append({
                    "query": qlabel, "partner": partner, "alphabet": alphabet, "scheme": s.name,
                    "partner_score": pv, "partner_rank": rank if pv > 0 else None,
                    "partner_n_tied": tied if pv > 0 else None,
                    "n_proteins": len(proteome),
                    "n_proteins_with_seed": int((col > 0).sum()) if s.family != "extend" else None,
                    "chance_seed_positions_per_search": int(total[others, si].sum())
                    if s.family != "extend" else None,
                    "index_pieces_returned": int(pieces[:, si].sum()) if s.family == "chained" else None,
                    "partner_best_diagonal": best_d - (len(q) - 1) if pv > 0 else None,
                    "query_length": len(q), "database_residues": len(db),
                    "diagonal_positions_scanned": cells, "seconds": secs,
                })
                # Rank of every protein, ties at the best rank of the tie, unscored at the bottom.
                order = np.sort(col)[::-1]
                r = np.searchsorted(-order, -col, side="left") + 1.0
                r[col == 0] = len(proteome)
                ranks[:, si] = r
                for j in np.argsort(-col, kind="stable")[:N_TOP]:
                    if col[j] > 0:
                        top.append({"query": qlabel, "alphabet": alphabet, "scheme": s.name,
                                    "rank": rank_of(col, col[j])[0], "gene": genes[j],
                                    "protein": names[j].split("|")[0], "length": len(proteome[j]),
                                    "score": col[j], "is_partner": j == p_idx})
                for d in np.flatnonzero(per_diag[:, si]):
                    diag.append({"query": qlabel, "alphabet": alphabet, "scheme": s.name,
                                 "diagonal": int(d) - (len(q) - 1), "value": per_diag[d, si]})
            # Random-protein null for the best rank over all schemes of this alphabet.
            best_partner = ranks[p_idx].min()
            best_random = ranks[draws].min(axis=1)
            null.append({"query": qlabel, "partner": partner, "alphabet": alphabet,
                         "n_schemes": len(schemes),
                         "partner_best_rank_any_scheme": best_partner,
                         "random_protein_median_best_rank": float(np.median(best_random)),
                         "p_random_protein_at_least_as_good": float((best_random <= best_partner).mean()),
                         "n_draws": N_NULL_DRAWS})
            draws_out.append(pl.DataFrame({"query": qlabel, "alphabet": alphabet,
                                           "random_protein_best_rank": best_random}))
    return rows, top, diag, null, pl.concat(draws_out)


def pfam_reach(schemes) -> pl.DataFrame:
    pairs = pl.read_parquet(PFAM_PAIRS).filter(pl.col("lali") >= MIN_ALIGNED).with_columns(
        pl.col("seqid_ali").cut(IDENTITY_BINS, labels=IDENTITY_LABELS).alias("identity_bin"))
    out = []
    for alphabet in ms.ALPHABETS:
        tab = ms.class_table(alphabet)
        qa = [tab[np.frombuffer(s.encode(), np.uint8)] for s in pairs["qaln"]]
        ta = [tab[np.frombuffer(s.encode(), np.uint8)] for s in pairs["taln"]]
        vals = ms.run_aligned(qa, ta, schemes)
        df = pl.DataFrame({"identity_bin": pairs["identity_bin"].cast(pl.String)})
        for si, s in enumerate(schemes):
            if s.family == "extend":
                continue
            sub = df.with_columns(pl.Series("v", vals[:, si]))
            agg = sub.group_by("identity_bin").agg(
                pl.len().alias("n_pairs"),
                (pl.col("v") > 0).mean().alias("fraction_of_pairs_with_seed"),
                pl.col("v").median().alias("median_value"))
            out.append(agg.with_columns(pl.lit(alphabet).alias("alphabet"),
                                        pl.lit(s.name).alias("scheme")))
    return pl.concat(out).select("alphabet", "scheme", "identity_bin", "n_pairs",
                                 "fraction_of_pairs_with_seed", "median_value") \
        .sort("alphabet", "scheme", "identity_bin")


def bh1_fire_positions(schemes) -> pl.DataFrame:
    """Every position on the BH1 diagonal of CED-9 against BCL2 where each scheme fires."""
    queries = dict(read_fasta(QUERIES))
    bcl2 = next(s for n, s in read_fasta(PROTEOME) if n.split("|")[6] == "BCL2")
    out = []
    for alphabet in ms.ALPHABETS:
        q, t = ms.encode(queries["Ced9"], alphabet), ms.encode(bcl2, alphabet)
        fired = ms.fire_positions(q, t, BH1_DIAGONAL, schemes)
        i0 = max(0, -BH1_DIAGONAL)
        for x, si in zip(*np.nonzero(fired)):
            out.append({"alphabet": alphabet, "scheme": schemes[si].name,
                        "ced9_position": int(i0 + x + 1),  # 1-based, where the seed ends
                        "bcl2_position": int(i0 + x + 1 + BH1_DIAGONAL)})
    return pl.DataFrame(out)


def main() -> None:
    schemes = ms.default_schemes()
    scheme_table(schemes).write_csv(OUT / "246_seed_schemes.tsv", separator="\t")
    fasta = read_fasta(PROTEOME)
    names = [n for n, _ in fasta]
    proteome = [s for _, s in fasta]
    genes = [n.split("|")[6] for n in names]  # GENCODE header field 7 is the gene name

    idx = []
    for alphabet in ms.ALPHABETS:
        db, starts = ms.concat_database([ms.encode(s, alphabet) for s in proteome])
        entries = ms.run_index_entries(db, starts, schemes)
        idx += [{"alphabet": alphabet, "scheme": s.name, "index_entries": int(e),
                 "database_residues": len(db)} for s, e in zip(schemes, entries)]
    pl.DataFrame(idx).write_csv(OUT / "246_human_proteome_index_entries.tsv", separator="\t")

    reach = pfam_reach(schemes)
    reach.write_csv(OUT / "246_pfam_seed_pairs_reach.tsv", separator="\t")
    print("Pfam reach done", flush=True)

    bh1_fire_positions(schemes).write_csv(OUT / "246_bh1_diagonal_seed_positions.tsv",
                                          separator="\t")
    rows, top, diag, null, draws = search(schemes, proteome, genes, names)
    (draws.group_by("query", "alphabet", "random_protein_best_rank").agg(pl.len().alias("n_draws"))
     .sort("query", "alphabet", "random_protein_best_rank")
     .write_csv(OUT / "246_random_protein_best_rank_draws.tsv", separator="\t"))
    pl.DataFrame(rows).write_csv(OUT / "246_human_proteome_search.tsv", separator="\t")
    pl.DataFrame(top).write_csv(OUT / "246_human_proteome_top_targets.tsv", separator="\t")
    pl.DataFrame(diag).write_csv(OUT / "246_partner_diagonals.tsv", separator="\t")
    pl.DataFrame(null).write_csv(OUT / "246_random_protein_null.tsv", separator="\t")


if __name__ == "__main__":
    main()
