#!/usr/bin/env python3
"""Generate notebooks/352_kmerseek_index_scale_UniParc.ipynb.

The notebook reads only the small result files copied back from Sherlock into
notebooks/352_data/sherlock/ (see notebooks/352_submit_all.sh). Until they exist it says
that nothing has been measured, and shows no number.
"""

import json
from pathlib import Path

PREREG_COMMIT = "0d029b350c35805d00576c10b2784ca1f9dd2c66"
REPO = "https://github.com/seanome/2024-kmerseek-analysis"
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "352_kmerseek_index_scale_UniParc.ipynb"

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {}, "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append(
        {
            "cell_type": "code",
            "id": f"code-{len(cells):02d}",
            "execution_count": None,
            "metadata": {"jupyter": {"source_hidden": True}},
            "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True),
        }
    )


md(
    rf"""
# 352. kmerseek index build and query time at 1, 10 and up to 100 billion residues (UniParc)

**Question.** When the database kmerseek searches grows from 1 to 10 billion and up to 100 billion amino
acids, does the time to answer one query grow in proportion to the index size, or faster?

**Decision it informs.** Whether the paper and every description of kmerseek may say it
scales to petabase-sized databases, and how large a database later experiments can
search in two days of compute.

**Rule, fixed before any data** ([preregistered.yaml]({REPO}/blob/{PREREG_COMMIT}/notebooks/352_data/preregistered.yaml),
commit [{PREREG_COMMIT[:8]}]({REPO}/commit/{PREREG_COMMIT})). Fit a straight line to
log10(median query time) against log10(index size on disk) over the measured sizes. The
slope is the exponent: 1 means query time grows in proportion to index size, above 1
means faster. If the lower end of the slope's 95% interval is above 1, query time grows
faster than linearly, and the petabase claim is removed from every description of the
method until it is measured directly. Lower is better: an exponent at or below 1 is what
the claim needs.

**The fit is arithmetic on three points.** With three sizes, one degree of freedom is
left, and the 95% interval is about 13 standard errors wide on each side. An interval
that includes 1 does not show the growth is linear; it shows three points cannot tell.

**kmerseek setting** (one combination of alphabet, k, scaled, extension and mask):
`hp_pbotc_1st_ed2`, k = 19, scaled 1 (every k-mer kept), extension with mismatch penalty
1.59 and give-up margin 6.36 (the X-drop), chained regions up to 30 residues apart, mask
off. Hydrophobic/polar alphabet `hp_pbotc_1st_ed2`: each residue becomes h or p.
kmerseek 0.4.0, image `olgabot/kmerseek:0.4.0-rc5`.

**Data.** Target databases: nested random subsets of UniParc active sequences (UniProt
2026_03), chosen by a hash of each sequence's UPI accession, so the 1 billion residue set
is inside the 10 billion set, which is inside the largest. Queries: 1,000 sequences from
the 1 billion residue set, the same at every size. Machine: one whole Sherlock node per
size (AMD EPYC 7742, 128 cores, 1,024 GB, no GPU), index on the node's own SSD.
Sources and checksums: [data/SOURCES.md]({REPO}/blob/main/data/SOURCES.md).
"""
)

md(
    r"""
**Stated change to the preregistration (2026-10-20).** The preregistration did not say how
many of the 1,000 queries must finish for a size to count. As first written, the code
dropped a size if even one query failed. Olga set the rule on 2026-10-20, after the
preregistration was committed and before any job ran: a size counts when at least 990 of
its 1,000 queries finish (exit 0). Queries that fail or are cut off by the time limit are
left out of the median and 95th percentile, and their number is printed in the table. If
the slow queries are the ones that fail, the counted times are lower than the true ones;
`n_queries_failed` and `n_queries_not_run` show how many were left out. The fit and the
rule are otherwise unchanged.
"""
)

code(
    r"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent / "scripts"))
import kmerseek_scale_352 as ks

ks.check_cpu_only()
RESULTS = Path("352_data/sherlock/results")
FIG = Path("../figures")
summaries = sorted(RESULTS.glob("size_*/summary.json")) if RESULTS.exists() else []
if not summaries:
    print("No measurement yet: the Sherlock jobs (notebooks/352_submit_all.sh) have not run, "
          "or their results have not been copied to notebooks/352_data/sherlock/results.")
else:
    budget = RESULTS.parent / "budget.tsv"
    print(f"{len(summaries)} database sizes with a summary; wall time used: "
          f"{ks.budget_total_h(budget):.2f} of {ks.BUDGET_HOURS:.0f} h")
"""
)

md(
    r"""
## 1. Measured index build and query times

Every number in this table was measured on Sherlock; the column names are the ones
`scripts/kmerseek_scale_352.py run-size` writes. `index_bytes_apparent` is the index size
on disk (sum of file sizes, bytes). `query_wall_s_median` and `query_wall_s_p95` are the
median and 95th percentile of the 1,000 per-query wall times (seconds, from process start
to exit, opening the index included). `complete` says whether all 1,000 queries
finished. `counted` says whether at least 990 did; only counted sizes enter the fit (see
"Stated change" above). The median and 95th percentile are over the queries that finished.

Figure SF1 (below the table): x, index size on disk; y, per-query wall time; both axes
spaced by ratio, so equal steps are equal factors. Filled circles are measured sizes. The
black line is the least-squares fit, drawn only between the smallest and largest measured
index; nothing beyond the last point is drawn. **The fitted exponent is arithmetic on
three points.**
"""
)

code(
    r"""
if summaries:
    rows = [json.loads(p.read_text()) for p in summaries]
    # summary.json is written before the build ends and again after the queries, so a failed
    # or unfinished size lacks some keys: take each column with .get (None when missing).
    cols = [
        "setting", "size_residues", "n_residues", "n_sequences", "index_build_wall_s",
        "index_build_peak_rss_gb", "index_bytes_apparent", "index_bytes_allocated",
        "n_queries_ok", "n_queries_failed", "n_queries_not_run", "complete", "counted",
        "query_wall_s_median", "query_wall_s_p95", "query_peak_rss_gb_max", "host",
    ]
    measured = pl.DataFrame({c: [r.get(c) for r in rows] for c in cols}).sort("size_residues")
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(measured)
    full = measured.filter(pl.col("counted").fill_null(False))
    if full.height >= 3:
        import scale_352_figure as sf

        x_gb = (full["index_bytes_apparent"] / 1e9).to_list()
        fit_median = ks.loglog_fit(full["index_bytes_apparent"], full["query_wall_s_median"])
        fit_p95 = ks.loglog_fit(full["index_bytes_apparent"], full["query_wall_s_p95"])
        fig = sf.plot_sf1(x_gb, full["query_wall_s_median"], full["query_wall_s_p95"], fit_median, fit_p95, ks.SETTING_NAME)
        import pubfig as pf
        pf.save(fig, FIG / "352_SF1_query_time_vs_index_size_kmerseek_UniParc")
        fits = pl.DataFrame([dict(series="median", **fit_median), dict(series="p95", **fit_p95)])
        print("Fits (arithmetic on three points):")
        print(fits)
        print(f"Rule fires (lower end of the 95% interval of the median exponent > 1): {ks.rule_fires(fit_median)}")
    else:
        print(f"{full.height} counted sizes: a slope with an interval needs three; no fit.")
"""
)

md(
    r"""
## 2. Verdict

Written after the measurements are in, under the rule above. Not written yet: nothing has
been measured.
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT}")
