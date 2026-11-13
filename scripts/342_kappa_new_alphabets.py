"""Copy rate (Cohen's kappa) and mismatch penalty C for aro_hyd_chg_pol4 and
aro_hyd_arg_chg_pol5, which notebook 230's pair table does not have.

Recomputes notebook 230's per-pair class agreement for the two new alphabets, plus
polarity4 as a check against its existing row, on the same Pfam-A seed 38.2 pairs, with
the same code (``hp_conservation_utils.pair_stats``). Then the dark-set tool
``tools/kappa-for-alphabet`` (olgabot/dark-set-kmerseek-0.4) turns each alphabet's pairs
into one row of ``kappa_by_alphabet.tsv``.

    python scripts/342_kappa_new_alphabets.py \
        ~/data/pfam/230_pfam_seed_pair_alignments.parquet \
        tables/342/230_pair_kappa_new_alphabets.parquet

The class lists are those of seanome/kmerseek PR 146 (src/rust/alphabets.rs at db906a3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import hp_conservation_utils as hcu  # noqa: E402

#: seanome/kmerseek PR 146, src/rust/alphabets.rs at db906a3.
NEW_ALPHABETS = {
    "aro_hyd_chg_pol4": ["FWY", "AILMV", "DEKR", "SNQTGCPH"],
    "aro_hyd_arg_chg_pol5": ["FWY", "AILMV", "R", "DEK", "SNQTGCPH"],
}
#: An alphabet already in notebook 230's table, recomputed here to check the code path.
CHECK_ALPHABET = "polarity4"
KEY_COLS = ["family", "family_id", "query", "target", "seqid_ali"]


def main(alignments: str, out: str) -> None:
    """Write one row per (pair, alphabet) for the pairs kappa-for-alphabet reads."""
    for name, clusters in NEW_ALPHABETS.items():
        assert sorted("".join(clusters)) == sorted(hcu.AA20), name
        hcu.TABLES[name] = hcu._table(clusters)
        hcu.SIZES[name] = len(clusters)
    pairs = pl.read_parquet(alignments).filter(
        pl.col("seqid_ali") > 0.2, pl.col("seqid_ali") <= 0.3
    )
    table = hcu.compute_pair_table(
        pairs, KEY_COLS, alphabets=[*NEW_ALPHABETS, CHECK_ALPHABET], n_shuffle=1
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.select(
        *KEY_COLS, "alphabet", "n_cols", "agree", "expected", "kappa"
    ).write_parquet(out)
    print(f"{pairs.height} pairs at 20-30% identity, {table.height} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
