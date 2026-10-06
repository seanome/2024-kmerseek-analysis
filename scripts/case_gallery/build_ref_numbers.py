"""The numbers the reference-pair tab quotes in its text, computed from sources.D241.

Ced9 and P66: the known partner's best rank over every alphabet, k and the five ranking
statistics notebook 241 uses, and notebook 241's random-protein null for that best rank
(alphabet_ranking_utils.random_protein_null, imported from the notebook-241 checkout).
BHF: how many searches put some human protein under E = 1 and under E = 0.01, and the
lowest E-values. BHF has no known human homologue, so a calibrated E-value would give
about one protein under E = 1 per search and about one under E = 0.01 in all searches.
Writes tables/case_gallery/num241.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from sources import D241, KMERSEEK_COMMIT, KMERSEEK_PR, NB241, NB241_COMMIT

ROOT = Path(__file__).resolve().parents[2]
head = subprocess.run(["git", "-C", str(NB241), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if head != NB241_COMMIT:
    sys.exit(f"{NB241} is at {head}, expected {NB241_COMMIT}")
sys.path.insert(0, str(NB241 / "notebooks"))
import alphabet_ranking_utils as au  # noqa: E402

ranks = pl.read_csv(D241 / "ranks.csv", schema_overrides={
    "rank": pl.Int64, "n_tied": pl.Int64, "partner_value": pl.Float64, "best_value": pl.Float64,
    "partner_found": pl.Boolean, "top_gene": pl.Utf8})
out = {"kmerseek_pr": KMERSEEK_PR, "kmerseek_commit": KMERSEEK_COMMIT}
for q in ["Ced9", "P66"]:
    null = au.random_protein_null(ranks, q)
    at = (ranks.filter(pl.col("query") == q, pl.col("metric").is_in(au.METRICS5), pl.col("partner_found"),
                       pl.col("rank") == null["observed_best_rank"])
          .sort("n_targets", "alphabet", "ksize").row(0, named=True))
    out[q] = {**null, "alphabet": at["alphabet"], "k": at["ksize"], "metric": at["metric"],
              "pct_random_as_good": round(100 * null["p_random_at_least_as_good"])}

reg = pl.scan_parquet(D241 / "regions.parquet").filter(pl.col("query_name") == "BHF")
per = (reg.group_by("alphabet", "ksize_arm", "gene").agg(pl.col("region_evalue").min().alias("E")).collect())
arms = per.group_by("alphabet", "ksize_arm").agg(lt1=(pl.col("E") < 1).sum(), lt001=(pl.col("E") < 0.01).sum())
low = per.sort("E").head(3)
out["BHF"] = {
    "n_arms": arms.height,
    "n_arms_lt1": int((arms["lt1"] > 0).sum()),
    "n_proteins_lt1": int(arms["lt1"].sum()),
    "n_proteins_lt001": int(arms["lt001"].sum()),
    "lowest": [{"gene": r["gene"], "E": r["E"], "alphabet": r["alphabet"], "k": int(r["ksize_arm"])}
               for r in low.iter_rows(named=True)],
}
(ROOT / "tables" / "case_gallery" / "num241.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
