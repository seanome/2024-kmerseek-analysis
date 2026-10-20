"""Notebook 287, part 2: chain scores for the dipeptide-shuffled copies of Ced9, P66 and BHF
(notebook 284's shuffled-query search, notebooks/284_shuffle_agreement.py), the null for
notebook 287's decision rule.

Each copy keeps its query's length, first and last residue and the count of every adjacent
residue pair (Altschul & Erickson 1985), so it has the query's composition and short
repeats and no homology. 284 searched 20 copies of each against the human proteome at
notebook 241's 152 settings with every region kept; it stores query_name, target_name,
region_start, region_end, target_start, region_evalue, gene, alphabet and ksize_arm,
which is all the chain score needs, given N per setting
(notebooks/287_settings_N_human_index_logs.csv: the build logs of the indexes 284 read).

For every query and copy, and the four main variants of notebook 287 (across settings or
one setting; each call's E-value times 71, the settings with an E-value fit, or not; gap
at most 200):
  the partner's (BCL2 for Ced9, CD47 for P66) chain and single E-value and rank,
  how many human proteins reach chain E < 1 and single E < 1.

Runs where 284's regions are (Sherlock): NB284_DIR (default $SCRATCH/284-shuffle-agreement).
Output: $NB284_DIR/287_shuffle_chains.csv (small; copy back to notebooks/).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import sum_statistics_utils as ss  # noqa: E402

NB284 = Path(
    os.environ.get(
        "NB284_DIR", Path(os.environ.get("SCRATCH", "/tmp")) / "284-shuffle-agreement"
    )
)
PARTNER = {"Ced9": "BCL2", "P66": "CD47", "BHF": None}
MAX_GAP = 200


def read_fasta(p: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = []
        else:
            out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def main() -> None:
    t0 = time.time()
    queries = read_fasta(NB284 / "queries.fa")
    n_tab = (
        pl.read_csv(ROOT / "notebooks/287_settings_N_human_index_logs.csv")
        .filter(pl.col("fitted"))
        .select("setting", pl.col("n_db").cast(pl.Float64))
    )
    n_settings = n_tab.height
    pen = math.log(n_settings)
    x = f"x{n_settings}"
    variants = [
        (f"across settings, {x}", ss.ChainRules(MAX_GAP, True, pen)),
        ("across settings, no factor", ss.ChainRules(MAX_GAP, True, 0.0)),
        (f"one setting, {x}", ss.ChainRules(MAX_GAP, False, pen)),
        ("one setting, no factor", ss.ChainRules(MAX_GAP, False, 0.0)),
    ]
    files = sorted((NB284 / "regions").glob("*.parquet"))
    if not files:
        sys.exit(f"no regions in {NB284 / 'regions'}; run 284's search step first")
    rows = []
    for q in queries:
        case = q.split("_shuf")[0]
        calls = (
            pl.scan_parquet(files)
            .filter((pl.col("query_name") == q) & pl.col("region_evalue").is_finite())
            .with_columns(
                setting=pl.col("alphabet") + ".k" + pl.col("ksize_arm").cast(pl.String),
                n_t=pl.col("target_name").str.split("|").list.last().cast(pl.Float64),
            )
            .join(n_tab.lazy(), on="setting", how="left")
        )
        calls = ss.add_normalised_score(calls).collect()
        missing = calls.filter(pl.col("n_db").is_null()).height
        if missing:
            sys.exit(
                f"{q}: {missing} calls with an E-value in a setting with no fit in 241's arms.csv"
            )
        for name, rules in variants:
            s = ss.score_pairs(calls, rules).with_columns(
                gene=pl.col("target_name").str.split("|").list.get(6)
            )
            p = PARTNER[case]
            prow = s.filter(pl.col("gene") == p) if p else s.clear()
            rows.append(
                dict(
                    query=q,
                    case=case,
                    real=q == case,
                    variant=name,
                    partner=p,
                    n_proteins_with_a_call=s.height,
                    n_chain_e_below_1=int((s["chain_e"] < 1).sum()),
                    n_single_e_below_1=int((s["single_e"] < 1).sum()),
                    partner_chain_e=prow["chain_e"][0] if prow.height else None,
                    partner_chain_r=prow["chain_r"][0] if prow.height else None,
                    partner_rank_chain=(
                        int((s["chain_log_e"] < prow["chain_log_e"][0]).sum()) + 1
                        if prow.height
                        else None
                    ),
                    partner_single_e=prow["single_e"][0] if prow.height else None,
                    partner_rank_single=(
                        int((s["single_log_e"] < prow["single_log_e"][0]).sum()) + 1
                        if prow.height
                        else None
                    ),
                )
            )
        print(f"{q} done, {time.time() - t0:.0f} s", flush=True)
    out = NB284 / "287_shuffle_chains.csv"
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
