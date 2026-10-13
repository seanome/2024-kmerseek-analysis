"""Notebook 284, part 4: where Ced9's best calls land on the human BCL-2 family, against
UniProt's BH motifs.

For each family member, the call with the lowest E-value over any alphabet and k
(notebook 241's regions.parquet), in 1-based coordinates, and the offset (target start
minus query start) that would put Ced9's motif on the target's same motif. A call is in
register when its own offset equals the motif offset within 1 residue.

BH motifs: UniProt P41958 (CED-9), P10415 (BCL2), Q07817 (BCL2L1), Q07820 (MCL1),
Q16611 (BAK1), Q07812 (BAX), fetched 2026-10-13.

Output: notebooks/284_ced9_bcl2_family_hits.csv
"""
from pathlib import Path

import polars as pl

DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
OUT = Path(__file__).resolve().parents[1] / "notebooks/284_ced9_bcl2_family_hits.csv"
CED9 = {"BH4": (80, 99), "BH1": (160, 179), "BH2": (213, 229)}
MOTIFS = {
    "BCL2": {"BH4": (10, 30), "BH3": (93, 107), "BH1": (136, 155), "BH2": (187, 202)},
    "BCL2L1": {"BH4": (4, 24), "BH3": (86, 100), "BH1": (129, 148), "BH2": (180, 195)},
    "MCL1": {"BH3": (209, 223), "BH1": (252, 272), "BH2": (304, 319)},
    "BAK1": {"BH3": (74, 88), "BH1": (117, 136), "BH2": (169, 184)},
    "BAX": {"BH3": (59, 73), "BH1": (98, 118), "BH2": (150, 165)},
}

r = (pl.scan_parquet(DATA / "regions.parquet")
     .filter((pl.col("query_name") == "Ced9") & pl.col("gene").is_in(list(MOTIFS)) & pl.col("region_evalue").is_finite())
     .collect())
rank = (pl.scan_parquet(DATA / "regions.parquet").filter((pl.col("query_name") == "Ced9") & pl.col("region_evalue").is_finite())
        .group_by("gene").agg(e=pl.col("region_evalue").min()).collect()
        .with_columns(rank_by_e=pl.col("e").rank("min")))
rows = []
for gene, motifs in MOTIFS.items():
    g = r.filter(pl.col("gene") == gene).sort("region_evalue", "alphabet", "ksize_arm", "region_start")
    for c in g.head(3).iter_rows(named=True):
        qs, qe, ts = int(c["region_start"]) + 1, int(c["region_end"]), int(c["target_start"]) + 1
        offset = ts - qs
        hit = [f"Ced9 {m} on {gene} {m}" for m in CED9 if m in motifs
               and abs((motifs[m][0] - CED9[m][0]) - offset) <= 1 and qs <= CED9[m][1] and qe >= CED9[m][0]]
        rows.append(dict(gene=gene, alphabet=c["alphabet"], k=c["ksize_arm"], ced9_start=qs, ced9_end=qe,
                         target_start=ts, target_end=int(c["target_end"]), mismatches=int(c["region_n_mismatches"]),
                         evalue=c["region_evalue"], in_register="; ".join(hit) or "no",
                         gene_rank_by_lowest_e=int(rank.filter(pl.col("gene") == gene)["rank_by_e"][0])))
df = pl.DataFrame(rows)
df.write_csv(OUT)
print(df.to_pandas().to_string())
