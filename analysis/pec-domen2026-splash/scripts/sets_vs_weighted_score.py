#!/usr/bin/env python3
"""Version A (weighted) and version B (sets only) scores (design/sets_vs_weighted/decision_rules.md).

Usage: sets_vs_weighted_score.py <work dir> <analysis dir> <out dir>

<work dir> holds botryllus/<set>/{up,down}.codes.tsv, human/<run>.counts.tsv.gz and
human_sets/<name>/{up,down}.<alphabet>.<k>.codes.gz, made by the three sets_vs_weighted_*.sbatch
jobs. <analysis dir> is analysis/pec-domen2026-splash, for the design files.

One score per Botryllus set, kmerseek setting and human comparison.

Version A is D of design/de2de/README.md: the human change (second group minus first) of the
Botryllus up k-mers minus that of the down k-mers, in log2 counts per million k-mers, over k-mers
with a total count of at least 10 in the comparison's runs. It is computed for the 11 human
comparisons and, as a check against PR 201, for the three Fig. 4 groupings scored there.

Version B is (same - reversed) / (|Q_up| + |Q_down|), where same = |Q_up & D_up| + |Q_down & D_down|
and reversed = |Q_up & D_down| + |Q_down & D_up|. It is computed for the 11 human comparisons and
their 18 label swaps. A k-mer in both the up and the down set of one side is dropped from both.

Each real Botryllus comparison is also scored with its up and down sets exchanged ("fullswap").
Every score must change sign and keep its size; if one does not, this script stops.

Writes to <out dir>:
  scores.tsv            one row per Botryllus set, setting and human set
  full_swap_check.tsv   each score beside its exchanged-labels score
  pr201_check.tsv       D here beside D in exports/de2de/de2de_results.tsv
  sizes_b.tsv           k-mers and bytes of each version B table (tables/*.tsv.gz)
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl

MIN_COUNT = 10
ABSENT = np.log2(0.5)  # log2 counts per million of a k-mer with count 0
FIG4 = {"gse99963_fig4_post_vs_pre": ("GSE99963", True), "gse144304_fig4_fit_vs_young": ("GSE144304", False),
        "gse120862_fig4_trained_vs_untrained": ("GSE120862", True)}


def exclusive(up, down):
    """Up-only codes, down-only codes, and how many codes were in both. Inputs sorted and unique."""
    both = np.intersect1d(up, down, assume_unique=True)
    return np.setdiff1d(up, both, assume_unique=True), np.setdiff1d(down, both, assume_unique=True), len(both)


def b_score(q_up, q_down, d_up, d_down):
    """same, reversed, denominator and score of version B. Score is None when the denominator is 0,
    and when the human set has no k-mers (reads shorter than k): a test with no k-mers is missing."""
    same = int(np.isin(q_up, d_up).sum() + np.isin(q_down, d_down).sum())
    rev = int(np.isin(q_up, d_down).sum() + np.isin(q_down, d_up).sum())
    den = len(q_up) + len(q_down)
    return same, rev, den, ((same - rev) / den if den and len(d_up) + len(d_down) else None)


def member(sorted_codes, x):
    """Which of x are in sorted_codes (sorted, unique). Same answer as np.isin, without sorting the
    large array again for every Botryllus set."""
    if len(sorted_codes) == 0:
        return np.zeros(len(x), bool)
    i = np.minimum(np.searchsorted(sorted_codes, x), len(sorted_codes) - 1)
    return sorted_codes[i] == x


def sorted_unique(a):
    """a as a sorted array with no repeats; kmer-codes already writes it so, which is checked, not assumed."""
    return a if len(a) < 2 or bool((a[1:] > a[:-1]).all()) else np.unique(a)


def run_weights(design, paired):
    """{run: weight} so that D = sum of weight x (a run's up mean minus down mean), real labels.

    Not paired: mean of the second group minus mean of the first. Paired: each subject's second-group
    mean minus its first-group mean, averaged over subjects. As weights() in scripts/de2de_test.py.
    """
    samples = design["sample"].to_list()
    second = (design["condition"] == "PEC").to_numpy()
    if not paired:
        w = np.where(second, 1 / second.sum(), -1 / (~second).sum())
    else:
        subj = np.array(design["subject"].to_list())
        subjects = sorted(set(subj))
        w = np.zeros(len(samples))
        for s in subjects:
            m = subj == s
            w += (np.where(m & second, 1 / (m & second).sum(), 0) - np.where(m & ~second, 1 / (m & ~second).sum(), 0))
        w /= len(subjects)
    return dict(zip(samples, w))


def d_score(sums, weights, n_up, n_down):
    """D from each run's (sum_up, present_up, sum_down, present_down). None when a set kept no k-mer."""
    if n_up == 0 or n_down == 0:
        return None
    d = 0.0
    for run, w in weights.items():
        su, pu, sd, pd = sums[run]
        d += w * ((su + (n_up - pu) * ABSENT) / n_up - (sd + (n_down - pd) * ABSENT) / n_down)
    return float(d)


def set_table(up, down):
    """A version B table: one row per k-mer with its direction."""
    return pl.concat([pl.DataFrame({"code": up}).with_columns(direction=pl.lit("up")),
                      pl.DataFrame({"code": down}).with_columns(direction=pl.lit("down"))])


def write_table(df, path):
    """Writes a table as gzip-compressed TSV and returns its size in bytes. Every table whose size is
    compared (version A and version B) goes through this function."""
    with gzip.open(path, "wb", compresslevel=6) as f:
        df.write_csv(f, separator="\t")
    return Path(path).stat().st_size


def read_botryllus(path, settings):
    d = pl.read_csv(path, separator="\t", schema_overrides={"code": pl.Int64})
    return {(a, k): np.unique(d.filter((pl.col("alphabet") == a) & (pl.col("k") == k))["code"].to_numpy())
            for a, k in settings}


def read_human_codes(folder, direction, a, k):
    """Sorted unique codes of one human set, direction and setting. A setting longer than the reads
    has no file, which is an empty set; a folder without codes.done is an unfinished job."""
    if not (folder / "codes.done").exists():
        raise SystemExit(f"{folder} has no codes.done: the human sets job did not finish")
    p = folder / f"{direction}.{a}.{k}.codes.gz"
    if not p.exists():
        return np.empty(0, np.int64)
    return sorted_unique(pl.read_csv(p, has_header=False, new_columns=["code"], schema_overrides={"code": pl.Int64})["code"].to_numpy())


def main(work, analysis, out):
    work, analysis, out = Path(work), Path(analysis), Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    design_dir = analysis / "design" / "sets_vs_weighted"
    settings = [(r["alphabet"], r["k"]) for r in pl.read_csv(design_dir / "settings.tsv", separator="\t").iter_rows(named=True)]
    bot = pl.read_csv(design_dir / "botryllus_sets.tsv", separator="\t").to_dicts()
    hum = pl.read_csv(design_dir / "human_sets.tsv", separator="\t").to_dicts()
    runs = pl.read_csv(design_dir / "human_runs.tsv", separator="\t")

    # Botryllus sets: the listed ones, then each real comparison with up and down exchanged
    q_sets = {}  # name -> {setting: (up-only, down-only, in both)}
    info = {}
    for b in bot:
        up = read_botryllus(work / "botryllus" / b["name"] / "up.codes.tsv", settings)
        down = read_botryllus(work / "botryllus" / b["name"] / "down.codes.tsv", settings)
        q_sets[b["name"]] = {s: exclusive(up[s], down[s]) for s in settings}
        info[b["name"]] = (b["comparison"], b["labels"])
        if b["labels"] == "real":
            q_sets[b["name"] + "__fullswap"] = {s: exclusive(down[s], up[s]) for s in settings}
            info[b["name"] + "__fullswap"] = (b["comparison"], "fullswap")
    names = list(q_sets)

    # Human comparisons scored by version A: real labels only, plus the Fig. 4 groupings as a check
    a_comp = {h["name"]: (h["series"], h["paired"], analysis / h["design"]) for h in hum if h["labels"] == "real"}
    for c, (series, paired) in FIG4.items():
        a_comp[c] = (series, paired, analysis / "design" / "human" / f"contrast_{c}.design.tsv")
    weights = {c: run_weights(pl.read_csv(p, separator="\t"), paired) for c, (_, paired, p) in a_comp.items()}
    comps_of = {}
    for c, w in weights.items():
        for run in w:
            comps_of.setdefault(run, []).append(c)
    missing = set(comps_of) - set(runs["run"])
    assert not missing, f"runs in a design but not in human_runs.tsv: {sorted(missing)[:5]}"

    # Query k-mers per setting, and each set's label on them: 1 up, -1 down, 0 neither
    q, lab = {}, {}
    for s in settings:
        q[s] = np.unique(np.concatenate([q_sets[n][s][i] for n in names for i in (0, 1)] + [np.empty(0, np.int64)]))
        for n in names:
            lab[s, n] = (np.isin(q[s], q_sets[n][s][0]).astype(np.int8) - np.isin(q[s], q_sets[n][s][1]).astype(np.int8))
        print(*s, "query k-mers", len(q[s]), flush=True)

    # The human counts were made against query_codes.tsv. A code missing there was never counted.
    qt = pl.read_csv(work / "query_codes.tsv", separator="\t", schema_overrides={"code": pl.Int64})
    for a, k in settings:
        have = qt.filter((pl.col("alphabet") == a) & (pl.col("k") == k))["code"].to_numpy()
        n_missing = len(q[a, k]) - int(np.isin(q[a, k], have).sum())
        assert n_missing == 0, f"{a} {k}: {n_missing} query k-mers are not in query_codes.tsv; rebuild it and recount"
    del qt

    def read_run(run):
        d = pl.read_csv(work / "human" / f"{run}.counts.tsv.gz", separator="\t",
                        schema_overrides={"code": pl.Int64, "count": pl.Int64})
        for a, k in settings:
            s = d.filter((pl.col("alphabet") == a) & (pl.col("k") == k))
            total = s.filter(pl.col("code") == -1)["count"]
            assert len(total) == 1, f"{run}: no total row for {a} {k}; it was counted with another settings list"
            s = s.filter(pl.col("code") >= 0)
            codes = s["code"].to_numpy()
            idx = np.searchsorted(q[a, k], codes)
            ok = idx < len(q[a, k])
            ok[ok] = q[a, k][idx[ok]] == codes[ok]
            # query_codes.tsv may hold codes in both sets of every Botryllus set: counted, not used
            yield (a, k), idx[ok], s["count"].to_numpy()[ok], total.item()

    # Pass 1: each k-mer's total count over each comparison's runs
    tot = {(s, c): np.zeros(len(q[s]), np.int64) for s in settings for c in a_comp}
    no_kmers = {(s, c): 0 for s in settings for c in a_comp}  # runs whose reads are shorter than k
    for run in comps_of:
        for s, idx, cnt, total in read_run(run):
            for c in comps_of[run]:
                tot[s, c][idx] += cnt
                no_kmers[s, c] += total == 0
    print("pass 1 done", flush=True)

    # Pass 2: per run, sums of log2 counts per million over the kept up and down k-mers
    sums = {}
    for run in comps_of:
        for s, idx, cnt, total in read_run(run):
            x = np.log2(cnt / total * 1e6 + 0.5) if total else np.zeros(len(cnt))
            for c in comps_of[run]:
                kept = tot[s, c][idx] >= MIN_COUNT
                xk = x[kept]
                for n in names:
                    lb = lab[s, n][idx][kept]
                    sums.setdefault((s, c, n), {})[run] = (xk[lb == 1].sum(), int((lb == 1).sum()),
                                                           xk[lb == -1].sum(), int((lb == -1).sum()))
    print("pass 2 done", flush=True)

    a_rows = {}
    for s in settings:
        for c in a_comp:
            keep = tot[s, c] >= MIN_COUNT
            for n in names:
                n_up, n_down = int((keep & (lab[s, n] == 1)).sum()), int((keep & (lab[s, n] == -1)).sum())
                d = d_score(sums.get((s, c, n), {r: (0.0, 0, 0.0, 0) for r in weights[c]}), weights[c], n_up, n_down)
                a_rows[s, c, n] = dict(a_up_kept=n_up, a_down_kept=n_down, a_D=d, a_runs=len(weights[c]),
                                       a_runs_without_kmers=int(no_kmers[s, c]))
    del tot, sums

    # Version B, one human set and setting at a time; and the size of each real set's table
    rows, sizes = [], []
    for h in hum:
        folder = work / "human_sets" / h["name"]
        for s in settings:
            a, k = s
            d_up, d_down, d_both = exclusive(read_human_codes(folder, "up", a, k), read_human_codes(folder, "down", a, k))
            if h["labels"] == "real":
                table = set_table(d_up, d_down)
                path = out / "tables" / f"human.{h['name']}.{a}.{k}.tsv.gz"
                sizes.append(dict(side="human", name=h["name"], alphabet=a, k=k, version="B", kmers=len(table),
                                  bytes=write_table(table, path)))
                del table
            in_up, in_down = member(d_up, q[s]), member(d_down, q[s])
            for n in names:
                q_up, q_down, q_both = q_sets[n][s]
                # b_score() on the union of query k-mers: lab is 1 for this set's up k-mers, -1 for down
                up_n, down_n = lab[s, n] == 1, lab[s, n] == -1
                same = int((in_up & up_n).sum() + (in_down & down_n).sum())
                rev = int((in_down & up_n).sum() + (in_up & down_n).sum())
                den = len(q_up) + len(q_down)
                # no human k-mers at this setting (reads shorter than k): missing, as in version A
                score = (same - rev) / den if den and len(d_up) + len(d_down) else None
                row = dict(botryllus_set=n, botryllus_comparison=info[n][0], botryllus_labels=info[n][1], alphabet=a, k=k,
                           human_set=h["name"], human_comparison=h["comparison"], human_labels=h["labels"], kind=h["kind"],
                           q_up=len(q_up), q_down=len(q_down), q_in_both=q_both,
                           d_up=len(d_up), d_down=len(d_down), d_in_both=d_both,
                           b_same=same, b_reversed=rev, b_denominator=den, b_score=score,
                           a_up_kept=None, a_down_kept=None, a_D=None, a_runs=None, a_runs_without_kmers=None)
                if h["labels"] == "real":
                    row |= a_rows[s, h["name"], n]
                rows.append(row)
        print(h["name"], "scored", flush=True)
    for n in names:
        if info[n][1] == "fullswap":
            continue
        for s in settings:
            q_up, q_down, _ = q_sets[n][s]
            table = set_table(q_up, q_down)
            sizes.append(dict(side="botryllus", name=n, alphabet=s[0], k=s[1], version="B", kmers=len(table),
                              bytes=write_table(table, out / "tables" / f"botryllus.{n}.{s[0]}.{s[1]}.tsv.gz")))

    schema = {"b_score": pl.Float64, "a_D": pl.Float64, "a_up_kept": pl.Int64, "a_down_kept": pl.Int64,
              "a_runs": pl.Int64, "a_runs_without_kmers": pl.Int64}
    res = pl.DataFrame(rows, schema_overrides=schema, infer_schema_length=None)
    res.write_csv(out / "scores.tsv", separator="\t")
    pl.DataFrame(sizes).write_csv(out / "sizes_b.tsv", separator="\t")

    # Exchanged labels: every score must change sign and keep its size
    key = ["botryllus_comparison", "alphabet", "k", "human_set"]
    real = res.filter(pl.col("botryllus_labels") == "real")
    full = res.filter(pl.col("botryllus_labels") == "fullswap").select(
        *key, pl.col("b_score").alias("b_score_fullswap"), pl.col("a_D").alias("a_D_fullswap"))
    chk = real.select(*key, "b_score", "a_D").join(full, on=key, how="left", validate="1:1")
    chk = chk.with_columns(b_sum=pl.col("b_score") + pl.col("b_score_fullswap"), a_sum=pl.col("a_D") + pl.col("a_D_fullswap"))
    chk.write_csv(out / "full_swap_check.tsv", separator="\t")
    bad = chk.filter((pl.col("b_sum").abs() > 1e-12) | (pl.col("a_sum").abs() > 1e-9)
                     | (pl.col("b_score").is_null() != pl.col("b_score_fullswap").is_null())
                     | (pl.col("a_D").is_null() != pl.col("a_D_fullswap").is_null()))
    print("exchanged-label check:", len(chk), "tests,", len(bad), "do not change sign and keep their size", flush=True)
    if len(bad):
        raise SystemExit("exchanged labels did not flip every score: see full_swap_check.tsv")

    # D here against D in PR 201, for the three settings and groupings scored there
    old_path = analysis / "exports" / "de2de" / "de2de_results.tsv"
    if old_path.exists():
        old = pl.read_csv(old_path, separator="\t", infer_schema_length=None).select(
            pl.col("botryllus").alias("botryllus_set"), "alphabet", "k", pl.col("human_contrast").alias("human_comparison"),
            pl.col("up_kept").alias("pr201_up_kept"), pl.col("down_kept").alias("pr201_down_kept"), pl.col("D").alias("pr201_D"))
        new = pl.DataFrame([dict(botryllus_set=n, alphabet=s[0], k=s[1], human_comparison=c, **v)
                            for (s, c, n), v in a_rows.items() if c in FIG4],
                           schema_overrides=schema, infer_schema_length=None)
        cmp = old.join(new, on=["botryllus_set", "alphabet", "k", "human_comparison"], how="left")
        cmp = cmp.with_columns(D_difference=pl.col("a_D") - pl.col("pr201_D"))
        cmp.write_csv(out / "pr201_check.tsv", separator="\t")
        print("PR 201 check:", len(cmp), "rows, largest |D difference|", cmp["D_difference"].abs().max(),
              ", kept counts differ in", len(cmp.filter((pl.col("a_up_kept") != pl.col("pr201_up_kept"))
                                                         | (pl.col("a_down_kept") != pl.col("pr201_down_kept")))), "rows")
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=250):
        print(res.filter((pl.col("botryllus_labels") == "real") & (pl.col("human_labels") == "real"))
              .select("botryllus_set", "alphabet", "k", "human_set", "kind", "a_D", "b_same", "b_reversed", "b_denominator", "b_score"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
