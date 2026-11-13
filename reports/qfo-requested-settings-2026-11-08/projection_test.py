"""Does the C2H2 zinc-finger lead come from how target domains are moved onto the query?

transfer_domains() gives each call the whole query span of the alignment. An MMseqs2
alignment over a tandem array of zinc fingers (about 140 residues on C2H2 proteins, median)
then becomes one call per target finger, each spanning the whole array, and each fails the
IoU >= 0.5 test against a single 23-residue finger. kmerseek regions are short (median 13
residues for dayhoff6 k9 on C2H2 proteins), so its calls are not cut down the same way.

This rescores both tools twice, from the same region files:
  span       transfer_domains() as the pipeline runs it
  projected  each target domain placed at its own position on the query, by the
             alignment's offset (query start - target start), clipped to the aligned span.
             Gaps inside an alignment are ignored, so a projected domain can sit a few
             residues off; the score_calls IoU test still decides whether it counts.

Everything after the transfer (score_calls, classify_scoreable, subset, protein-centric Fmax)
is the pipeline's own code. The span arm must reproduce the published Fmax, or it stops.

Decision rule, written before the first run (2026-11-09):
  - If MMseqs2's projected C2H2 Fmax rises above the typical kmerseek setting's (ratio below 1),
    the C2H2 lead comes from the transfer step, not from kmerseek finding zinc fingers better.
  - If kmerseek stays above MMseqs2 after projection, the lead is kmerseek's own.

Run on Sherlock, from the repo root:
  python reports/qfo-requested-settings-2026-11-08/projection_test.py \
      --results <run>/results --kmerseek-dirs <run>/results/kmerseek <run>/results-hp-k35-k40/kmerseek \
      --settings-file reports/qfo-requested-settings-2026-11-08/settings_in_figure.txt \
      --published <run>/results/metrics <run>/results-hp-k35-k40/metrics --cpus 16 --out <dir>
"""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
sys.path.insert(
    0, str(HERE.parents[1] / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin")
)
import cafa_metrics as cm  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402

MIN_OVERLAP = 0.5  # params.min_overlap
GRAY_MIN = 0.5  # --gray-min-annotated-fraction default
CUTS = [("all", "all"), ("geneset", "zinc_finger_c2h2")]
NINE = [
    "arabidopsis",
    "chicken",
    "ciona",
    "ecoli",
    "fly",
    "mouse",
    "worm",
    "yeast",
    "zebrafish",
]


def projected_transfer(regions: pl.LazyFrame, domain_map: pl.LazyFrame) -> pl.LazyFrame:
    """transfer_domains(), but each target domain lands at its own place on the query."""
    joined = regions.join(
        domain_map.select(
            pl.col("accession").alias("target_acc"),
            "pfam_id",
            pl.col("domain_start").alias("t_dom_start"),
            pl.col("domain_end").alias("t_dom_end"),
        ),
        on="target_acc",
        how="inner",
    )
    offset = pl.col("qstart") - pl.col("tstart")
    return (
        joined.with_columns(
            edc.overlap_expr("tstart", "tend", "t_dom_start", "t_dom_end").alias(
                "t_overlap"
            ),
            (pl.col("t_dom_end") - pl.col("t_dom_start")).alias("t_dom_len"),
        )
        # The same keep rule as transfer_domains, so both arms start from the same set.
        .filter(pl.col("t_overlap") >= MIN_OVERLAP * pl.col("t_dom_len"))
        .with_columns(
            pl.max_horizontal("qstart", pl.col("t_dom_start") + offset).alias(
                "p_start"
            ),
            pl.min_horizontal("qend", pl.col("t_dom_end") + offset).alias("p_end"),
        )
        .select(
            "query_acc",
            "pfam_id",
            pl.col("p_start").alias("qstart"),
            pl.col("p_end").alias("qend"),
            "score",
        )
    )


def score_species(job, truth, results):
    """Fmax for every (search, arm, cut) in one species."""
    species, searches = job
    truth_lf = truth.lazy()
    ic = cm.information_content(truth)
    map_lf = pl.read_parquet(results / "truth" / f"{species}_domain_map.parquet").lazy()
    rows = []
    for search, path in searches:
        regions = edc.load_regions(Path(path), False)
        for arm in ("span", "projected"):
            if regions is None:
                scored = None
            else:
                calls = (
                    edc.transfer_domains(regions, map_lf, MIN_OVERLAP)
                    if arm == "span"
                    else projected_transfer(regions, map_lf)
                )
                scored = edc.score_calls(
                    calls, truth_lf, MIN_OVERLAP, "alignment", "cover"
                )
                scored = edc.classify_scoreable(scored, truth, GRAY_MIN)
            for axis, value in CUTS:
                if scored is None:
                    fmax, n = 0.0, 0
                else:
                    t_sub, c_sub = edc.subset(truth, scored, "all", axis, value)
                    fmax = cm.cafa_scalars(cm.protein_centric_curve(c_sub, t_sub, ic))[
                        "fmax"
                    ]
                    n = c_sub.height
                rows.append(
                    {
                        "species": species,
                        "search": search,
                        "arm": arm,
                        "stratum": value,
                        "fmax": fmax,
                        "n_calls": n,
                    }
                )
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--kmerseek-dirs", type=Path, nargs="+", required=True)
    p.add_argument(
        "--settings-file",
        type=Path,
        required=True,
        help="one kmerseek setting per line, e.g. dayhoff6_k9",
    )
    p.add_argument("--species", nargs="+", default=NINE)
    p.add_argument(
        "--published",
        type=Path,
        nargs="+",
        required=True,
        help="metrics dirs to check the span arm against",
    )
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    truth = pl.read_parquet(a.results / "truth" / "human_domain_truth.parquet")
    cov = pl.read_parquet(a.results / "truth" / "human_query_covariates.parquet")
    truth = edc.attach_strata(truth, cov, keep_zinc_finger=True)

    settings = a.settings_file.read_text().split()
    jobs = {}
    for sp in a.species:
        mm = (
            a.results
            / "regions"
            / "mmseqs2_seqseq"
            / f"human_vs_{sp}.mmseqs2_seqseq.tsv.gz"
        )
        jobs[sp] = [("mmseqs2", mm)]
        for setting in settings:
            alphabet, k = setting.rsplit("_k", 1)
            name = f"human_vs_{sp}.{alphabet}.k{k}.lcfalse.regions.parquet"
            found = [d / name for d in a.kmerseek_dirs if (d / name).exists()]
            if found:
                jobs[sp].append((setting, found[0]))
        print(f"{sp}: {len(jobs[sp]) - 1} kmerseek settings", file=sys.stderr)
    work = partial(score_species, truth=truth, results=a.results)
    # spawn, not fork: see subset_null.py (job 51664847 hung after forking under polars).
    with multiprocessing.get_context("spawn").Pool(a.cpus) as pool:
        rows = [
            r for part in pool.imap_unordered(work, sorted(jobs.items())) for r in part
        ]
    scores = pl.DataFrame(rows)
    scores.write_csv(a.out / "projection_scores.csv")

    check(scores, a)
    summary = summarise(scores)
    summary.write_csv(a.out / "projection_summary.csv")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(summary)


def check(scores, a):
    """The span arm is the pipeline's own transfer, so it must give the published Fmax."""
    want = []
    for d in a.published:
        for f in d.glob("pfam.*.metrics.parquet"):
            parts = f.name.split(".")
            if ".dedup." in f.name:
                continue
            if parts[1] == "kmerseek" and parts[2].endswith("_lcFalse"):
                search = parts[2].removesuffix("_lcFalse")
            elif parts[1:3] == ["mmseqs2_seqseq", "s7"]:
                search = "mmseqs2"
            else:
                continue
            m = pl.read_parquet(f).filter(
                pl.col("split") == "all",
                pl.struct("stratum_axis", "stratum").is_in(
                    [{"stratum_axis": x, "stratum": y} for x, y in CUTS]
                ),
            )
            for r in m.select("stratum", "fmax").iter_rows():
                want.append(
                    {
                        "species": parts[3],
                        "search": search,
                        "stratum": r[0],
                        "published": r[1],
                    }
                )
    j = scores.filter(pl.col("arm") == "span").join(
        pl.DataFrame(want), on=["species", "search", "stratum"]
    )
    bad = j.filter((pl.col("fmax") - pl.col("published")).abs() > 1e-6)
    j.write_csv(a.out / "projection_published_check.csv")
    print(f"published check: {j.height} compared, {bad.height} differ", file=sys.stderr)
    if j.height == 0 or bad.height:
        print(bad.head(20), file=sys.stderr)
        raise SystemExit("span arm does not reproduce the published Fmax")


def summarise(scores):
    """other_metrics.py's typical ratio, per arm and cut, plus MMseqs2's own Fmax."""
    mm = scores.filter(pl.col("search") == "mmseqs2").select(
        "species", "arm", "stratum", pl.col("fmax").alias("mm")
    )
    ratio = (
        scores.filter(pl.col("search") != "mmseqs2", pl.col("n_calls") > 0)
        .join(mm, on=["species", "arm", "stratum"])
        .with_columns((pl.col("fmax") / pl.col("mm")).alias("ratio"))
        .filter(pl.col("ratio").is_finite())
        .group_by("search", "arm", "stratum")
        .agg(
            pl.col("ratio").median(),
            pl.col("fmax").median(),
            pl.len().alias("n_species"),
        )
        .filter(pl.col("n_species") >= 5)
        .group_by("arm", "stratum")
        .agg(
            pl.col("ratio").median().alias("typical_ratio"),
            pl.col("fmax").median().alias("kmerseek_typical_fmax"),
            pl.len().alias("n_settings"),
        )
    )
    mm_med = mm.group_by("arm", "stratum").agg(
        pl.col("mm").median().alias("mmseqs2_fmax")
    )
    return ratio.join(mm_med, on=["arm", "stratum"]).sort("stratum", "arm")


if __name__ == "__main__":
    main()
