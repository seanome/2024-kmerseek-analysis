#!/usr/bin/env python3
"""Stage 2 of the Pfam candidate table: one unfiltered row per (human Pfam instance, species).

Reads stage 1's per-(setting or tool, species) files (scripts/pfam_instance_calls.py) and the
Pfam answer key, and writes:

  pfam_fair_table.parquet     every human Pfam-A instance x every target species, with the
                              closed IoU of each tool's best same-family call (0 when none),
                              for the six aligners and kmerseek under two ways of choosing its
                              setting. No row is dropped for what any tool did.
  pfam_setting_choice.csv     the setting each scheme chose, per length bin and per clan, with
                              the number of selection-half instances behind each choice.
  pfam_summary_heldout.csv    per length bin, species and tool, on the held-out half: number
                              of instances, mean IoU, share with IoU >= 0.5, share landed.
  pfam_candidates_<view>.csv  labelled views of the held-out rows (see VIEWS). Each view is
                              written with its mirror image, so a list of kmerseek wins never
                              appears without the matching list of aligner wins.
  pfam_candidates_counts.csv  the size of every view per length bin, both directions.
  pfam_settings_left_out.csv  kmerseek settings the run did not finish on every species,
                              left out of the choice (written only when there are any).

Choosing kmerseek's setting (mask on only, as in notebook 244 Stage 0):

  score     share of selection-half (instance, species) rows the setting lands on, pooled over
            species; ties go to the higher mean IoU, then to the setting name.
  length    one setting per domain-length bin (LENGTH_BINS); a bin with no selection-half
            instance uses the one setting best over all lengths (kmerseek_length_source).
  clan      one setting per Pfam clan, chosen on the clan's selection-half instances. The
            pipeline splits by family, so held-out families are new to the choice; a clan
            with fewer than --min-clan-instances selection instances, a clan no setting
            touches on the selection half, or a family in no clan, uses its length bin's
            setting instead (column kmerseek_clan_source says which).

"Lands" is notebook 244's rule, applied to any overlapping call of the family (not only the
best-IoU one): at least 80% of the call inside the domain and the call
covering at least 30% of it, on closed intervals. IoU is closed and 1-based for every tool.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import polars as pl

ALIGNERS = {
    "hmmer3_phmmer": "phmmer",
    "mmseqs2_seqseq": "MMseqs2",
    "mmseqs2_iterative": "MMseqs2 iterative",
    "foldseek": "Foldseek",
    "prostt5": "ProstT5",
    "reseek": "Reseek",
}
LENGTH_BINS = [(0, 59, "<60 aa"), (60, 149, "60-149 aa"), (150, 10**9, ">=150 aa")]
KEY = ["accession", "pfam_id", "domain_start", "domain_end"]


def length_bin_expr() -> pl.Expr:
    L = pl.col("domain_end") - pl.col("domain_start") + 1
    e = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, name in reversed(LENGTH_BINS):
        e = pl.when((L >= lo) & (L <= hi)).then(pl.lit(name)).otherwise(e)
    return e


def read_clans(path: Path | None) -> pl.DataFrame:
    """Pfam-A.clans.tsv(.gz): pfam_id, clan_id, clan name, family id, family description."""
    if path is None:
        return pl.DataFrame({"pfam_id": [], "clan": []}, schema={"pfam_id": pl.Utf8, "clan": pl.Utf8})
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[1] and f[1] != "\\N":
                rows.append((f[0], f[1]))
    return pl.DataFrame(rows, schema={"pfam_id": pl.Utf8, "clan": pl.Utf8}, orient="row")


def choose(calls_km: pl.DataFrame, inst_sp: pl.DataFrame, group: str) -> pl.DataFrame:
    """Best mask-on setting per group on the selection half (landed share, then mean IoU)."""
    sel = inst_sp.filter(pl.col("split") == "selection")
    denom = sel.group_by(group).agg(n_rows=pl.len())
    arms = calls_km.select("arm").unique()
    grid = denom.join(arms, how="cross")
    hits = (
        calls_km.join(sel.select(KEY + ["species", group]), on=KEY + ["species"], how="inner")
        .group_by([group, "arm"])
        .agg(n_landed=pl.col("any_landed").sum(), sum_iou=pl.col("best_iou").sum())
    )
    scored = (
        grid.join(hits, on=[group, "arm"], how="left")
        .with_columns(pl.col("n_landed").fill_null(0), pl.col("sum_iou").fill_null(0.0))
        .with_columns(landed_share=pl.col("n_landed") / pl.col("n_rows"),
                      mean_iou=pl.col("sum_iou") / pl.col("n_rows"))
    )
    return (
        scored.sort([group, "landed_share", "mean_iou", "arm"],
                    descending=[False, True, True, False])
        .group_by(group, maintain_order=True)
        .first()
        .select(group, "arm", "n_rows", "landed_share", "mean_iou")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls-dir", type=Path, required=True, help="stage 1 output directory")
    ap.add_argument("--truth", type=Path, required=True,
                    help="results/truth/human_domain_truth.parquet")
    ap.add_argument("--clans", type=Path, default=None,
                    help="Pfam-A.clans.tsv.gz from the same Pfam release as the run")
    ap.add_argument("--min-clan-instances", type=int, default=20)
    ap.add_argument("--n-species", type=int, default=9,
                    help="expected target species; stage 2 stops if the inputs have fewer")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build the table even if some (setting or tool, species) files are "
                         "missing; their rows then read IoU 0, so do this only for tests")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    truth = (
        pl.read_parquet(args.truth)
        .select(KEY + ["protein_length", "split"])
        .unique(KEY)
        .with_columns(length_aa=pl.col("domain_end") - pl.col("domain_start") + 1,
                      length_bin=length_bin_expr())
    )
    clans = read_clans(args.clans)
    truth = truth.join(clans, on="pfam_id", how="left").with_columns(
        pl.col("clan").fill_null("no clan"))

    calls = pl.read_parquet(sorted(args.calls_dir.glob("*.pfam_calls.parquet")))
    calls = calls.rename({"query_acc": "accession", "true_start": "domain_start",
                          "true_end": "domain_end"})
    species = sorted(calls["species"].unique().to_list())
    # A missing (setting or tool, species) file would read as "found nothing" (IoU 0), so
    # check the inputs are complete before building anything.
    files = pl.DataFrame({"f": [p.name for p in args.calls_dir.glob("*.pfam_calls.parquet")]}).with_columns(
        arm=pl.col("f").str.replace(r"\.[a-z]+\.pfam_calls\.parquet$", ""),
        species=pl.col("f").str.extract(r"\.([a-z]+)\.pfam_calls\.parquet$"))
    per_arm = files.group_by("arm").agg(n=pl.col("species").n_unique())
    tools_seen = {a.split(".")[0] for a in per_arm["arm"].to_list()}
    problems = []
    if len(species) < args.n_species or files["species"].n_unique() < args.n_species:
        problems.append(f"{files['species'].n_unique()} species in the inputs, {args.n_species} expected")
    short = per_arm.filter(pl.col("n") < files["species"].n_unique())
    # A kmerseek setting the run did not finish on every species (gbmr7 at k 9-10 runs out of
    # memory on the larger targets) cannot be compared with the others on the same rows, so
    # it is left out of the choice and listed. A missing aligner file still stops the build.
    km_short = short.filter(pl.col("arm").str.starts_with("kmerseek."))
    if km_short.height:
        km_short.rename({"n": "n_species"}).sort("arm").write_csv(
            args.out_dir / "pfam_settings_left_out.csv")
        print(f"left out {km_short.height} kmerseek settings missing some species: "
              f"{km_short['arm'].sort().to_list()}")
        calls = calls.filter(~pl.col("arm").is_in(km_short["arm"].to_list()))
        short = short.filter(~pl.col("arm").str.starts_with("kmerseek."))
    if short.height:
        problems.append(f"{short.height} settings or tools lack some species, e.g. {short['arm'][:5].to_list()}")
    missing_tools = [t for t in ALIGNERS if t not in tools_seen]
    if missing_tools:
        problems.append(f"no files for {missing_tools}")
    if problems and not args.allow_missing:
        raise SystemExit("inputs incomplete: " + "; ".join(problems) +
                         ". Rerun the missing stage-1 tasks, or pass --allow-missing for a test.")
    for p_ in problems:
        print("WARNING", p_)
    inst_sp = truth.join(pl.DataFrame({"species": species}), how="cross")

    km = calls.filter((pl.col("tool") == "kmerseek") & pl.col("arm").str.ends_with("_lcTrue"))

    by_len = choose(km, inst_sp, "length_bin")
    overall = choose(km, inst_sp.with_columns(all_lengths=pl.lit("all lengths")), "all_lengths")
    # A clan where no setting touches any selection instance would get the first setting by
    # name (every setting ties at zero), so it uses its length bin's setting instead.
    by_clan = choose(km, inst_sp, "clan").filter(
        (pl.col("n_rows") >= args.min_clan_instances * len(species)) & (pl.col("clan") != "no clan")
        & (pl.col("mean_iou") > 0))
    pl.concat([
        by_len.rename({"length_bin": "group"}).with_columns(scheme=pl.lit("length bin")),
        overall.rename({"all_lengths": "group"}).with_columns(scheme=pl.lit("fallback: all lengths")),
        by_clan.rename({"clan": "group"}).with_columns(scheme=pl.lit("clan")),
    ]).with_columns(n_selection_instances=pl.col("n_rows") // len(species)).drop("n_rows").write_csv(
        args.out_dir / "pfam_setting_choice.csv")

    # Which setting each (instance) uses under each scheme.
    use = (
        truth.select(KEY + ["length_bin", "clan"])
        .join(by_len.select("length_bin", pl.col("arm").alias("arm_length_only")), on="length_bin", how="left")
        .with_columns(arm_length=pl.coalesce("arm_length_only", pl.lit(overall["arm"][0])),
                      kmerseek_length_source=pl.when(pl.col("arm_length_only").is_null())
                      .then(pl.lit("all-lengths fallback")).otherwise(pl.lit("length bin")))
        .join(by_clan.select("clan", pl.col("arm").alias("arm_clan_only")), on="clan", how="left")
        .with_columns(
            arm_clan=pl.coalesce("arm_clan_only", "arm_length"),
            kmerseek_clan_source=pl.when(pl.col("arm_clan_only").is_null())
            .then(pl.lit("length bin fallback")).otherwise(pl.lit("clan")),
        )
        .drop("arm_clan_only", "arm_length_only", "length_bin", "clan")
    )

    table = inst_sp.join(use, on=KEY, how="left")
    cols = ["best_iou", "any_landed", "best_qstart", "best_qend", "best_target_acc",
            "best_tstart", "best_tend"]

    def attach(t: pl.DataFrame, sub: pl.DataFrame, prefix: str, on_arm: str | None = None):
        sub = sub.select(KEY + ["species", "arm"] + cols) if on_arm else sub.select(KEY + ["species"] + cols)
        if on_arm:
            sub = sub.rename({"arm": on_arm})
            t = t.join(sub, on=KEY + ["species", on_arm], how="left")
        else:
            t = t.join(sub, on=KEY + ["species"], how="left")
        ren = {c: f"{prefix}_{c.removeprefix('best_').removeprefix('any_')}" for c in cols}
        t = t.rename(ren)
        return t.with_columns(pl.col(f"{prefix}_iou").fill_null(0.0),
                              pl.col(f"{prefix}_landed").fill_null(False))

    table = attach(table, km, "kmerseek_length", on_arm="arm_length")
    table = attach(table, km, "kmerseek_clan", on_arm="arm_clan")
    for tool, label in ALIGNERS.items():
        sub = calls.filter(pl.col("tool") == tool)
        table = attach(table, sub, label.replace(" ", "_"))
    # The best of ALL kmerseek settings per row: an upper bound, chosen after seeing the answer.
    oracle = (km.sort("best_iou", descending=True).group_by(KEY + ["species"]).first()
              .select(KEY + ["species", pl.col("best_iou").alias("kmerseek_best_any_setting_ORACLE_iou"),
                             pl.col("arm").alias("kmerseek_best_any_setting_ORACLE_arm")]))
    table = table.join(oracle, on=KEY + ["species"], how="left").with_columns(
        pl.col("kmerseek_best_any_setting_ORACLE_iou").fill_null(0.0))
    al_iou = [f"{a.replace(' ', '_')}_iou" for a in ALIGNERS.values()]
    # best_aligner_iou is the best of six per row, chosen after seeing the answer; it is
    # against kmerseek, which gets one pre-chosen setting. Keep that in mind in the views.
    table = table.with_columns(best_aligner_iou=pl.max_horizontal(al_iou))
    assert table.height == inst_sp.height, (table.height, inst_sp.height)
    table.write_parquet(args.out_dir / "pfam_fair_table.parquet")

    held = table.filter(pl.col("split") == "heldout")
    tools = {"kmerseek (length-bin setting)": "kmerseek_length",
             "kmerseek (clan setting)": "kmerseek_clan",
             **{v: v.replace(" ", "_") for v in ALIGNERS.values()}}
    summ = []
    for label, p in tools.items():
        summ.append(held.group_by(["length_bin", "species"]).agg(
            n=pl.len(), mean_iou=pl.col(f"{p}_iou").mean(),
            share_iou_ge_0_5=(pl.col(f"{p}_iou") >= 0.5).mean(),
            share_landed=pl.col(f"{p}_landed").mean()).with_columns(tool=pl.lit(label)))
        summ.append(held.group_by(["length_bin"]).agg(
            n=pl.len(), mean_iou=pl.col(f"{p}_iou").mean(),
            share_iou_ge_0_5=(pl.col(f"{p}_iou") >= 0.5).mean(),
            share_landed=pl.col(f"{p}_landed").mean()).with_columns(
            species=pl.lit("all species"), tool=pl.lit(label)))
    pl.concat(summ, how="diagonal").sort(["length_bin", "species", "tool"]).write_csv(
        args.out_dir / "pfam_summary_heldout.csv")

    counts = []
    for scheme in ("kmerseek_length", "kmerseek_clan"):
        k = pl.col(f"{scheme}_iou")
        a = pl.col("best_aligner_iou")
        views = {
            "only_kmerseek_places": (k >= 0.5) & (a == 0),
            "only_aligners_place": (a >= 0.5) & (k == 0),
            "kmerseek_tighter_by_0.3": (k > 0) & (a > 0) & (k - a > 0.3),
            "aligner_tighter_by_0.3": (k > 0) & (a > 0) & (a - k > 0.3),
            "both_place": (k >= 0.5) & (a >= 0.5),
            "neither_touches": (k == 0) & (a == 0),
        }
        for name, cond in views.items():
            v = held.filter(cond)
            v.sort(["length_bin", "pfam_id", "accession", "species"]).write_csv(
                args.out_dir / f"pfam_candidates_{scheme}_{name}.csv")
            counts.append(held.group_by("length_bin").agg(
                n_heldout_rows=pl.len(), n_in_view=cond.sum()).with_columns(
                scheme=pl.lit(scheme), view=pl.lit(name)))
    pl.concat(counts).sort(["scheme", "view", "length_bin"]).write_csv(
        args.out_dir / "pfam_candidates_counts.csv")
    print(f"{truth.height} human Pfam instances x {len(species)} species = {table.height} rows; "
          f"held out: {held.height}")


if __name__ == "__main__":
    main()
