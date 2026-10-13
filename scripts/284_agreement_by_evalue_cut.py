"""Notebook 284, part 2: alphabet agreement when each alphabet's calls must pass an E-value
cut, and with alphabet families counted instead of alphabets.

For Ced9 (partner BCL2) and P66 (partner CD47), at each cut in alphabet_agreement_utils.E_CUTS,
the partner's agreement and the number of human proteins with more, for the three counts
(alphabets; families with the H/P alphabets as one; families with spots protein20 also calls
counted as 1). BHF has no known partner, so only the proteome's top value is given.

Input: notebook 241's regions.parquet (see 284_alphabet_agreement.py).
Output: notebooks/284_agreement_by_evalue_cut.csv
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import alphabet_agreement_utils as au  # noqa: E402

DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
CASES = [("Ced9", "BCL2"), ("P66", "CD47"), ("BHF", None)]

rows = []
for q, partner in CASES:
    d = au.prepare(pl.scan_parquet(DATA / "regions.parquet").filter(pl.col("query_name") == q))
    for cut in au.E_CUTS:
        s = au.spots(d, cut)
        for c in au.COUNTS:
            b = au.best_per_target(s, c)
            top = int(b[c].max()) if b.height else 0
            row = dict(query=q, partner=partner, e_cut="all regions" if cut is None else f"E < {cut:g}",
                       count=c, proteins_with_a_call=b.height, proteome_max=top,
                       proteins_at_max=int(b.filter(pl.col(c) == top).height) if b.height else 0,
                       genes_at_max=",".join(sorted(b.filter(pl.col(c) == top)["gene"].unique().to_list())[:5]))
            if partner:
                p = b.filter(pl.col("gene") == partner)
                v = int(p[c].max()) if p.height else 0
                row.update(partner_value=v, proteins_with_more=int(b.filter(pl.col(c) > v).height))
            rows.append(row)
            print(row)
pl.DataFrame(rows).write_csv(ROOT / "notebooks/284_agreement_by_evalue_cut.csv")
