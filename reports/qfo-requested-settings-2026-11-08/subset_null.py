"""Is kmerseek's lead on C2H2 zinc fingers more than what any 32 proteins would give?

C2H2 zinc fingers ranked first of 59 protein subsets on kmerseek / MMseqs2 Fmax, picked
after looking. This rescores random protein sets with the pipeline's own Fmax code
(evaluate_domain_calls.subset + cafa_metrics.protein_centric_curve, on the published
per-search call files), and summarises each set exactly as other_metrics.py does: ratio per
setting and species, median over species (at least 5), median over settings ("typical").

Cuts scored:
  observed_c2h2           the 32 C2H2 zinc-finger proteins (geneset axis)
  observed_16_30          domains of 16-30 aa (feature_length_bin axis)
  observed_16_30_no_c2h2  domains of 16-30 aa on proteins that are not C2H2 zinc fingers
  random_any_<i>          N random query proteins, N = the number of C2H2 proteins
  random_short_<i>        N random proteins that have a 16-30 aa domain and are not C2H2

The observed cuts must reproduce the published Fmax; the script stops if they do not.

Decision rule, written before the first run (2026-11-09):
  - C2H2 is a kmerseek strength only if its typical ratio is above every random_any set
    but at most 1 in 100 (one-sided p <= 0.01, p = (1 + n at or above) / (1 + n draws)).
  - If it also beats random_short at p <= 0.01, the lead belongs to zinc fingers. If not,
    it is a short-domain effect that C2H2 proteins happen to carry.
  - The 16-30 aa lead stands on its own only if observed_16_30_no_c2h2 stays above 1.

Run on Sherlock (compute node), from the repo root:
  python reports/qfo-requested-settings-2026-11-08/subset_null.py \
      --results <run>/results --calls-dir <run>/results/calls <run>/results-hp-k35-k40/calls \
      --draws 1000 --cpus 16 --out <dir>
"""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"))
import cafa_metrics as cm  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402

NINE = ["arabidopsis", "chicken", "ciona", "ecoli", "fly", "mouse", "worm", "yeast", "zebrafish"]
MMSEQS2 = "pfam.mmseqs2_seqseq.s7.{species}.calls.parquet"
MIN_SPECIES = 5


def load_truth(results: Path) -> pl.DataFrame:
    """The answer key with the strata scoreDomainCalls attaches, built by the same functions."""
    truth = pl.read_parquet(results / "truth" / "human_domain_truth.parquet")
    cov = pl.read_parquet(results / "truth" / "human_query_covariates.parquet")
    truth = edc.attach_strata(truth, cov, keep_zinc_finger=True)
    return edc.attach_feature_length(truth)


def cuts(truth: pl.DataFrame, n_draws: int, seed: int) -> dict[str, tuple[pl.Series, bool]]:
    """Each cut is (in-cut mask over truth rows, instance-level?)."""
    c2h2 = pl.col("stratum_geneset") == "zinc_finger_c2h2"
    short = pl.col("stratum_feature_length_bin") == "16-30"
    c2h2_acc = set(truth.filter(c2h2)["accession"].to_list())
    any_pool = sorted(truth["accession"].unique().to_list())
    short_pool = sorted(set(truth.filter(short)["accession"].to_list()) - c2h2_acc)
    n = len(c2h2_acc)
    rng = np.random.default_rng(seed)
    out = {
        "observed_c2h2": (truth.select(c2h2.fill_null(False)).to_series(), False),
        "observed_16_30": (truth.select(short.fill_null(False)).to_series(), True),
        "observed_16_30_no_c2h2": (
            truth.select((short & ~pl.col("accession").is_in(c2h2_acc)).fill_null(False)).to_series(), True),
    }
    for name, pool in (("any", any_pool), ("short", short_pool)):
        for i in range(n_draws):
            pick = rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()
            out[f"random_{name}_{i}"] = (truth.select(pl.col("accession").is_in(pick)).to_series(), False)
    print(f"C2H2 proteins {n}; random_any pool {len(any_pool)}; random_short pool {len(short_pool)}",
          file=sys.stderr)
    return out


def score_species(species: str, truth: pl.DataFrame, cut_items: list, call_files: dict,
                  results: Path) -> list[dict]:
    """Fmax and call count for every (cut, search) in one species."""
    ic = cm.information_content(truth)
    rows = []
    calls = {name: pl.read_parquet(path) for name, path in call_files[species].items()}
    for cut, (mask, instance_level) in cut_items:
        t = truth.with_columns(mask.alias("stratum_cut"))
        axes = frozenset({"cut"}) if instance_level else frozenset()
        for search, c in calls.items():
            t_sub, c_sub = edc.subset(t, c, "all", "cut", True, axes)
            pc = cm.protein_centric_curve(c_sub, t_sub, ic)
            rows.append({"cut": cut, "species": species, "search": search,
                         "fmax": cm.cafa_scalars(pc)["fmax"], "n_calls": c_sub.height})
    return rows


def summarise(scores: pl.DataFrame) -> pl.DataFrame:
    """other_metrics.py's typical ratio, per cut."""
    mm = scores.filter(pl.col("search") == "mmseqs2").select("cut", "species", pl.col("fmax").alias("mm"))
    return (
        scores.filter(pl.col("search") != "mmseqs2", pl.col("n_calls") > 0)
        .join(mm, on=["cut", "species"])
        .with_columns((pl.col("fmax") / pl.col("mm")).alias("ratio"))
        .filter(pl.col("ratio").is_finite())
        .group_by("cut", "search")
        .agg(pl.col("ratio").median(), pl.len().alias("n_species"))
        .filter(pl.col("n_species") >= MIN_SPECIES)
        .group_by("cut")
        .agg(pl.col("ratio").median().alias("typical_ratio"), pl.len().alias("n_settings"))
        .sort("cut")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--calls-dir", type=Path, nargs="+", required=True)
    p.add_argument("--published-metrics", type=Path, nargs="+", default=None,
                   help="metrics dirs to check the observed cuts against (default: <calls-dir>/../metrics)")
    p.add_argument("--species", nargs="+", default=NINE)
    p.add_argument("--settings", nargs="*", default=None, help="limit kmerseek settings (for a quick test)")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    truth = load_truth(a.results)
    call_files = {}
    for sp in a.species:
        files = {}
        for d in a.calls_dir:
            for f in d.glob(f"pfam.kmerseek.*_lcFalse.{sp}.calls.parquet"):
                setting = f.name.split(".")[2].removesuffix("_lcFalse")
                if a.settings is None or setting in a.settings:
                    files[setting] = f
        mm = next(d / MMSEQS2.format(species=sp) for d in a.calls_dir if (d / MMSEQS2.format(species=sp)).exists())
        files["mmseqs2"] = mm
        call_files[sp] = files
        print(f"{sp}: {len(files) - 1} kmerseek settings", file=sys.stderr)

    items = list(cuts(truth, a.draws, a.seed).items())
    chunks = [(sp, items[i::a.cpus]) for sp in a.species for i in range(a.cpus)]
    work = partial(_run_chunk, truth=truth, call_files=call_files, results=a.results)
    with Pool(a.cpus) as pool:
        rows = [r for part in pool.imap_unordered(work, chunks) for r in part]
    scores = pl.DataFrame(rows)
    scores.write_parquet(a.out / "subset_null_scores.parquet")

    check_published(scores, a)
    summary = summarise(scores)
    summary.write_csv(a.out / "subset_null_summary.csv")
    report(summary, a.out)


def _run_chunk(chunk, truth, call_files, results):
    species, cut_items = chunk
    return score_species(species, truth, cut_items, call_files, results)


def check_published(scores: pl.DataFrame, a) -> None:
    """The observed cuts must give the Fmax the pipeline published, or the null means nothing."""
    dirs = a.published_metrics or [d.parent / "metrics" for d in a.calls_dir]
    want = []
    for cut, axis, stratum in (("observed_c2h2", "geneset", "zinc_finger_c2h2"),
                               ("observed_16_30", "feature_length_bin", "16-30")):
        for d in dirs:
            for f in d.glob("pfam.*.metrics.parquet"):
                parts = f.name.split(".")
                if ".dedup." in f.name or parts[3] not in a.species:
                    continue
                if parts[1] == "kmerseek":
                    if not parts[2].endswith("_lcFalse"):
                        continue
                    search = parts[2].removesuffix("_lcFalse")
                elif parts[1:3] == ["mmseqs2_seqseq", "s7"]:
                    search = "mmseqs2"
                else:
                    continue
                m = pl.read_parquet(f).filter(pl.col("split") == "all", pl.col("stratum_axis") == axis,
                                              pl.col("stratum") == stratum)
                if m.height:
                    want.append({"cut": cut, "species": parts[3], "search": search,
                                 "published": m["fmax"][0]})
    j = scores.join(pl.DataFrame(want), on=["cut", "species", "search"])
    bad = j.filter((pl.col("fmax") - pl.col("published")).abs() > 1e-6)
    print(f"published check: {j.height} (cut, species, search) compared, {bad.height} differ", file=sys.stderr)
    j.write_csv(a.out / "published_check.csv")
    if j.height == 0 or bad.height:
        print(bad.head(20), file=sys.stderr)
        raise SystemExit("observed cuts do not reproduce the published Fmax; the null is not comparable")


def report(summary: pl.DataFrame, out: Path) -> None:
    obs = dict(summary.filter(pl.col("cut").str.starts_with("observed")).select("cut", "typical_ratio").iter_rows())
    lines = [f"{k}: typical ratio {v:.3f}" for k, v in sorted(obs.items())]
    for name in ("any", "short"):
        null = summary.filter(pl.col("cut").str.starts_with(f"random_{name}_"))["typical_ratio"].to_numpy()
        x = obs["observed_c2h2"]
        p = (1 + (null >= x).sum()) / (1 + len(null))
        lines.append(f"random_{name}: {len(null)} draws, median {np.median(null):.3f}, "
                     f"95th {np.quantile(null, 0.95):.3f}, max {null.max():.3f}; C2H2 p = {p:.4f}")
    (out / "subset_null_verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
