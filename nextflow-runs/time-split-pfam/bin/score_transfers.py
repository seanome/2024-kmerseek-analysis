#!/usr/bin/env python3
"""Score family transfers for the time-split test (notebook 280).

Every tool's hits become *calls*: (setting, query, query interval, family, score). A hit
on a Swiss-Prot 2019_08 protein transfers family F when its target interval overlaps a
Pfam 33.1 domain of F on that protein (kmerseek rows arrive already joined, by
compact_kmerseek.awk); an hmmscan hit names its Pfam 33.1 family directly. Scores are
oriented so higher is better:
    kmerseek exact    region_poisson_score
    kmerseek extend   -log10(region_evalue)
    MMseqs2           -log10(evalue)
    phmmer, hmmscan   -log10(domain i-Evalue)
All coordinates are 1-based with the end included.

Rules, fixed before any result (notebook 280, first cell):
- found: a call of the same family whose residues lie at least half inside the t1 domain
  (landing fraction >= 0.5). Coverage of the t1 domain and same-clan credit are reported
  as secondary readouts.
- errors are counted on decoys only: a transfer of any family to a decoy query.
- matched cutoff: on the tune half, the lowest score at which decoys receive at most
  1 family transfer (distinct decoy-family pair) per 100 decoy queries. Recall is read on
  the test half at that cutoff.

Writes calls/<search>.parquet (one file per search output, so no step holds every call
at once: the pilot's 185 kmerseek searches are 421 million rows), instance_calls.parquet
(best call per instance and setting), setting_summary.csv, controls.csv and
parse_checks.csv.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import polars as pl

LANDING_MIN = 0.5
DECOY_RATE_MAX = 0.01  # family transfers per decoy query
CALL_SCHEMA = {
    "setting": pl.String,
    "tool": pl.String,
    "query": pl.String,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "pfam_acc": pl.String,
    "score": pl.Float64,
    "target": pl.String,
}


def neg_log10(col: str) -> pl.Expr:
    """-log10 of an E-value column; 0 becomes 400 (beyond any double), inf or null -inf."""
    e = pl.col(col).cast(pl.Float64, strict=False)
    return (
        pl.when(e <= 0)
        .then(400.0)
        .when(e.is_infinite() | e.is_null())
        .then(float("-inf"))
        .otherwise(-e.log10())
    )


KM_COLUMNS = [
    "query",
    "qstart",
    "qend",
    "pfam_acc",
    "target",
    "region_poisson_score",
    "region_evalue",
]


def read_kmerseek_file(f: Path) -> tuple[pl.DataFrame, dict]:
    """One kmerseek search output (one setting, one query chunk) as calls. Only the columns
    scored are read: a pilot file holds up to ~27 million rows."""
    alphabet, k, arm = f.name.split(".")[0], f.name.split(".")[1], f.name.split(".")[2]
    setting = f"kmerseek {alphabet} {k} {arm}"
    with gzip.open(f, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    cols = KM_COLUMNS if set(KM_COLUMNS) <= set(header) else None
    df = pl.read_csv(f, separator="\t", infer_schema_length=0, columns=cols)
    check = {"file": f.name, "rows": df.height, "columns": ",".join(header)}
    if df.height == 0 or "region_evalue" not in header:
        return pl.DataFrame(schema=CALL_SCHEMA), check
    score = (
        pl.col("region_poisson_score").cast(pl.Float64)
        if arm == "exact"
        else neg_log10("region_evalue")
    )
    return (
        df.select(
            pl.lit(setting).alias("setting"),
            pl.lit("kmerseek").alias("tool"),
            "query",
            pl.col("qstart").cast(pl.Int64),
            pl.col("qend").cast(pl.Int64),
            "pfam_acc",
            score.alias("score"),
            "target",
        ),
        check,
    )


def transfer_by_overlap(hits: pl.DataFrame, domains: pl.DataFrame) -> pl.DataFrame:
    """Hits with tstart/tend on a Swiss-Prot protein -> one row per overlapped domain."""
    j = hits.join(domains, left_on="target", right_on="sp_acc", how="inner")
    return j.filter(
        (pl.col("tstart") <= pl.col("sp_end")) & (pl.col("sp_start") <= pl.col("tend"))
    )


def read_mmseqs(dirpath: Path, domains: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "query",
        "target",
        "qstart",
        "qend",
        "tstart",
        "tend",
        "pident",
        "bits",
        "evalue",
    ]
    frames = []
    for f in sorted(dirpath.glob("*.mmseqs2.tsv.gz")):
        with gzip.open(f, "rt") as fh:
            if not fh.read(1):
                continue  # no hit at all in this chunk
        frames.append(
            pl.read_csv(
                f,
                separator="\t",
                has_header=False,
                new_columns=cols,
                schema_overrides={"query": pl.String, "target": pl.String},
            )
        )
    hits = pl.concat(frames) if frames else None
    if hits is None or hits.height == 0:
        return pl.DataFrame(schema=CALL_SCHEMA)
    t = transfer_by_overlap(hits, domains)
    return t.select(
        pl.lit("MMseqs2").alias("setting"),
        pl.lit("MMseqs2").alias("tool"),
        "query",
        pl.col("qstart").cast(pl.Int64),
        pl.col("qend").cast(pl.Int64),
        "pfam_acc",
        neg_log10("evalue").alias("score"),
        "target",
    )


def read_domtbl(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                rows.append(line.split(None, 22))
    return rows


def read_phmmer(dirpath: Path, domains: pl.DataFrame) -> pl.DataFrame:
    # domtblout: 1 target, 4 query, 13 i-Evalue, 16-17 hmm from/to (= query positions,
    # the query is the profile), 18-19 ali from/to (= target positions).
    rows = [r for f in sorted(dirpath.glob("*.domtbl.gz")) for r in read_domtbl(f)]
    if not rows:
        return pl.DataFrame(schema=CALL_SCHEMA)
    hits = pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "query": [r[3] for r in rows],
            "ievalue": [float(r[12]) for r in rows],
            "qstart": [int(r[15]) for r in rows],
            "qend": [int(r[16]) for r in rows],
            "tstart": [int(r[17]) for r in rows],
            "tend": [int(r[18]) for r in rows],
        }
    )
    t = transfer_by_overlap(hits, domains)
    return t.select(
        pl.lit("phmmer").alias("setting"),
        pl.lit("phmmer").alias("tool"),
        "query",
        "qstart",
        "qend",
        "pfam_acc",
        neg_log10("ievalue").alias("score"),
        "target",
    )


def read_hmmscan(dirpath: Path) -> pl.DataFrame:
    # domtblout: 2 Pfam accession (PF00001.21), 4 query, 13 i-Evalue, 18-19 ali from/to
    # (query positions; for hmmscan the query is the sequence).
    rows = [r for f in sorted(dirpath.glob("*.domtbl.gz")) for r in read_domtbl(f)]
    if not rows:
        return pl.DataFrame(schema=CALL_SCHEMA)
    return pl.DataFrame(
        {
            "setting": "hmmscan Pfam 33.1",
            "tool": "hmmscan",
            "query": [r[3] for r in rows],
            "qstart": [int(r[17]) for r in rows],
            "qend": [int(r[18]) for r in rows],
            "pfam_acc": [r[1].split(".")[0] for r in rows],
            "score": [
                400.0 if float(r[12]) <= 0 else -math.log10(float(r[12])) for r in rows
            ],
            "target": None,
        }
    ).with_columns(pl.col("target").cast(pl.String))


def matched_cutoff(dec: pl.DataFrame, n_decoys: int) -> float:
    """Lowest score at which distinct decoy-family transfers <= DECOY_RATE_MAX x n_decoys.
    `dec` holds one row per (decoy, family) with its best score."""
    allowed = math.floor(DECOY_RATE_MAX * n_decoys)
    s = sorted(dec["score"].to_list(), reverse=True)
    if len(s) <= allowed:
        return float("-inf")
    # Keep the top `allowed` pairs: the cutoff sits just above the next one.
    return math.nextafter(s[allowed], math.inf)


INSTANCE_KEY_COLS = ["setting", "query", "pfam_acc", "t1_start", "t1_end"]


def best_instance_calls(real: pl.DataFrame, inst: pl.DataFrame) -> pl.DataFrame:
    """Best call per (setting, instance) of the same family that lands at least half inside
    the t1 domain, with landing and coverage. Ties on score go to the earlier call."""
    return (
        inst.join(
            real.select("setting", "query", "qstart", "qend", "pfam_acc", "score"),
            on=["query", "pfam_acc"],
            how="inner",
        )
        .with_columns(
            ov=(
                pl.min_horizontal("qend", "t1_end")
                - pl.max_horizontal("qstart", "t1_start")
                + 1
            ).clip(0)
        )
        .with_columns(
            landing=pl.col("ov") / (pl.col("qend") - pl.col("qstart") + 1),
            coverage=pl.col("ov") / pl.col("length"),
        )
        .filter(pl.col("landing") >= LANDING_MIN)
        .sort(
            ["score", "qstart", "qend"],
            descending=[True, False, False],
            maintain_order=True,
        )
        .unique(INSTANCE_KEY_COLS, keep="first", maintain_order=True)
    )


def reduce_calls(c: pl.DataFrame, inst: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Everything the scoring needs from one batch of calls (already joined to queries.tsv):
    parse checks, call counts, best call per instance, and best score per decoy-family pair.
    A query sits in one chunk, so these add up across chunks of the same setting."""
    return {
        "pc": c.group_by("setting").agg(
            n_calls=pl.len(),
            unknown_query=pl.col("is_decoy").is_null().sum(),
            start_below_1=(pl.col("qstart") < 1).sum(),
            end_past_query=(pl.col("qend") > pl.col("query_length")).sum(),
            start_after_end=(pl.col("qstart") > pl.col("qend")).sum(),
        ),
        "counts": c.group_by("setting").agg(
            n_calls=pl.len(),
            real_queries_with_call=pl.col("query")
            .filter(~pl.col("is_decoy"))
            .n_unique(),
            decoy_queries_with_call=pl.col("query")
            .filter(pl.col("is_decoy"))
            .n_unique(),
        ),
        "ic": best_instance_calls(c.filter(~pl.col("is_decoy")), inst),
        "dec": (
            c.filter(pl.col("is_decoy") & (pl.col("role") == "sample"))
            .group_by("setting", "query", "pfam_acc", "half")
            .agg(pl.col("score").max())
        ),
        "unknown": c.filter(pl.col("is_decoy").is_null()).select("query").unique(),
    }


P4_TYPES = ("Motif", "Disordered", "Repeat")
SHORT_AA = 60
INSTANCE_KEY = ["query", "pfam_acc", "t1_start", "t1_end"]
N_BOOT = 10_000
BOOT_SEED = 280


def is_p4(inst: pl.DataFrame) -> pl.DataFrame:
    """An instance counts toward P4 when its t1 type is Motif, Disordered or Repeat, or its
    t1 region is under 60 aa (notebook 280, first cell)."""
    return inst.with_columns(
        p4=pl.col("pfam_type").is_in(P4_TYPES) | (pl.col("length") < SHORT_AA)
    )


def found_at_cutoff(ic: pl.DataFrame, cutoffs: dict[str, float]) -> pl.DataFrame:
    """(setting, instance) pairs found at each setting's matched cutoff."""
    cut = pl.DataFrame({"setting": list(cutoffs), "cutoff": list(cutoffs.values())})
    return (
        ic.join(cut, on="setting")
        .filter(pl.col("score") >= pl.col("cutoff"))
        .select(["setting"] + INSTANCE_KEY)
        .unique()
    )


def unique_per_query(
    found: pl.DataFrame,
    p4_inst: pl.DataFrame,
    kmerseek: list[str],
    comparators: list[str],
) -> pl.DataFrame:
    """For each kmerseek setting and each query carrying a P4 instance: how many of that
    query's P4 instances the setting finds and no comparator finds. Zero rows included, so
    the table can be resampled query by query."""
    by_comp = (
        found.filter(pl.col("setting").is_in(comparators)).select(INSTANCE_KEY).unique()
    )
    uniq = (
        found.filter(pl.col("setting").is_in(kmerseek))
        .join(p4_inst.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
        .join(by_comp, on=INSTANCE_KEY, how="anti")
        .group_by("setting", "query")
        .agg(n_unique=pl.len())
    )
    grid = pl.DataFrame({"setting": kmerseek}).join(
        p4_inst.select("query").unique(), how="cross"
    )
    return grid.join(uniq, on=["setting", "query"], how="left").with_columns(
        pl.col("n_unique").fill_null(0).cast(pl.Int64)
    )


def decoy_transfers_per_query(
    dec_pairs: dict[str, pl.DataFrame],
    cutoffs: dict[str, float],
    decoys: list[str],
    half: str,
) -> pl.DataFrame:
    """Family transfers (distinct decoy-family pairs) at the matched cutoff, per setting and
    per decoy query of one half, zeros included."""
    frames = []
    for s, d in dec_pairs.items():
        c = (
            d.filter((pl.col("half") == half) & (pl.col("score") >= cutoffs[s]))
            .group_by("query")
            .agg(n_transfers=pl.len())
        )
        frames.append(
            pl.DataFrame({"query": decoys})
            .join(c, on="query", how="left")
            .with_columns(
                pl.col("n_transfers").fill_null(0).cast(pl.Int64), setting=pl.lit(s)
            )
        )
    return (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={"query": pl.String, "n_transfers": pl.Int64, "setting": pl.String}
        )
    )


def unique_vs_expected(uq: pl.DataFrame, dt: pl.DataFrame) -> pl.DataFrame:
    """Per kmerseek setting: unique finds, decoy transfer rate, and the number expected by
    chance = rate x number of real queries carrying a P4 instance."""
    u = uq.group_by("setting").agg(
        unique_finds=pl.col("n_unique").sum(), p4_queries=pl.len()
    )
    r = dt.group_by("setting").agg(decoy_rate=pl.col("n_transfers").mean())
    return (
        u.join(r, on="setting", how="left")
        .with_columns(expected=pl.col("decoy_rate") * pl.col("p4_queries"))
        .with_columns(unique_minus_expected=pl.col("unique_finds") - pl.col("expected"))
    )


def choose_p4_setting(tune: pl.DataFrame) -> str:
    """Largest unique finds minus expected; ties to the lower decoy rate, then by name."""
    return tune.sort(
        ["unique_minus_expected", "decoy_rate", "setting"],
        descending=[True, False, False],
    )["setting"][0]


def bootstrap_interval(
    n_unique: list[int],
    n_transfers: list[int],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Unique finds minus expected, with a 95% interval from resampling the real P4 queries
    and, independently, the decoys. Returns (point, low, high)."""
    import numpy as np

    u = np.asarray(n_unique, dtype=float)
    t = np.asarray(n_transfers, dtype=float)
    point = u.sum() - t.mean() * len(u)
    rng = np.random.default_rng(seed)
    ub = u[rng.integers(0, len(u), size=(n_boot, len(u)))].sum(axis=1)
    tb = t[rng.integers(0, len(t), size=(n_boot, len(t)))].mean(axis=1) * len(u)
    low, high = np.percentile(ub - tb, [2.5, 97.5])
    return float(point), float(low), float(high)


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in (
        "kmerseek",
        "stats",
        "mmseqs2",
        "phmmer",
        "hmmscan",
        "inputs",
        "reference",
        "settings",
        "out",
    ):
        ap.add_argument(f"--{a}", type=Path, required=True)
    ap.add_argument(
        "--read-test-half",
        action="store_true",
        help="compute test-half recall and decoy rate; off for the pilot, whose "
        "queries are part of the approved sample",
    )
    a = ap.parse_args()
    q = pl.read_csv(a.inputs / "queries.tsv", separator="\t")
    inst = pl.read_csv(a.inputs / "instances.tsv", separator="\t")
    dom = pl.read_csv(a.reference / "sprot_domains.tsv", separator="\t")

    if not a.read_test_half:
        # The pilot's sample queries belong to the approved sample, so their test half is
        # removed here, before any call is joined to it: nothing downstream can read it.
        hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))[
            "query"
        ]
        q = q.filter(~pl.col("query").is_in(hidden.implode()))
        inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
        print(f"test half left unread: {hidden.len()} queries (real and decoy) dropped")

    # Every setting in settings.tsv x {exact, extend}, with what arrived for it: a stats
    # file says fitted yes/no; no file means every chunk of that search failed.
    st = (
        pl.read_csv(a.settings, separator="\t") if a.settings.suffix == ".tsv" else None
    )
    if st is not None and {"alphabet", "ksize"} <= set(st.columns):
        stats = [
            pl.read_csv(f, separator="\t", infer_schema_length=0)
            for f in sorted(a.stats.glob("*.stats.tsv"))
        ]
        got = (
            pl.concat(stats).select("alphabet", "ksize", "arm", "chunk", "fitted")
            if stats
            else pl.DataFrame(
                schema={
                    c: pl.String
                    for c in ("alphabet", "ksize", "arm", "chunk", "fitted")
                }
            )
        )
        want = st.select("alphabet", pl.col("ksize").cast(pl.String)).join(
            pl.DataFrame({"arm": ["exact", "extend"]}), how="cross"
        )
        cover = want.join(
            got.group_by("alphabet", "ksize", "arm").agg(
                chunks=pl.len(), unfitted_chunks=(pl.col("fitted") == "no").sum()
            ),
            on=["alphabet", "ksize", "arm"],
            how="left",
        ).with_columns(
            pl.col("chunks").fill_null(0), pl.col("unfitted_chunks").fill_null(0)
        )
        cover.write_csv(a.out / "setting_coverage.csv")
        print(
            f"settings with no search output: {cover.filter(pl.col('chunks') == 0).height}; "
            f"with an unfitted extend search: {cover.filter(pl.col('unfitted_chunks') > 0).height}"
        )

    # One search output at a time: read, join to queries.tsv, save its calls, keep only the
    # small reductions. Comparators go through the same path.
    qcols = q.select(
        "query", "is_decoy", "half", "role", pl.col("length").alias("query_length")
    )
    calls_dir = a.out / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pl.DataFrame]] = {
        k: [] for k in ("pc", "counts", "ic", "dec", "unknown")
    }
    checks = []

    def add(c: pl.DataFrame, name: str) -> None:
        c = c.select([pl.col(k).cast(t) for k, t in CALL_SCHEMA.items()])
        if not a.read_test_half:
            c = c.filter(~pl.col("query").is_in(hidden.implode()))
        c = c.join(qcols, on="query", how="left")
        c.write_parquet(calls_dir / f"{name}.parquet")
        for k, v in reduce_calls(c, inst).items():
            parts[k].append(v)

    for f in sorted(a.kmerseek.glob("*.transfers.tsv.gz")):
        c, check = read_kmerseek_file(f)
        checks.append(check)
        add(c, f.name.removesuffix(".transfers.tsv.gz"))
        print(f"{f.name}: {check['rows']} rows", flush=True)
    add(read_mmseqs(a.mmseqs2, dom), "MMseqs2")
    add(read_phmmer(a.phmmer, dom), "phmmer")
    add(read_hmmscan(a.hmmscan), "hmmscan")

    pc = pl.concat(parts["pc"]).group_by("setting").sum().sort("setting")
    pc.write_csv(a.out / "parse_checks.csv")
    pl.DataFrame(
        checks, schema={"file": pl.String, "rows": pl.Int64, "columns": pl.String}
    ).write_csv(a.out / "kmerseek_files.csv")
    unknown = pl.concat(parts["unknown"]).unique()["query"].to_list()
    if unknown:
        print(
            f"WARNING: {len(unknown)} query names in calls are not in queries.tsv, e.g. {unknown[:5]}"
        )

    ic = pl.concat(parts["ic"])
    ic.write_parquet(a.out / "instance_calls.parquet")
    dec_all = pl.concat(parts["dec"])
    counts = pl.concat(parts["counts"]).group_by("setting").sum()

    settings = counts["setting"].sort().to_list()
    n_dec = {
        h: q.filter(
            pl.col("is_decoy") & (pl.col("half") == h) & (pl.col("role") == "sample")
        ).height
        for h in ("tune", "test")
    }
    inst_s = inst.filter(pl.col("role") == "sample").join(
        q.select("query", "half"), on="query"
    )
    rows, cutoffs, dec_by_setting = [], {}, {}
    for s in settings:
        cnt = counts.filter(pl.col("setting") == s).row(0, named=True)
        dec_pairs = dec_all.filter(pl.col("setting") == s).drop("setting")
        cut = matched_cutoff(dec_pairs.filter(pl.col("half") == "tune"), n_dec["tune"])
        cutoffs[s], dec_by_setting[s] = cut, dec_pairs
        dec_test = dec_pairs.filter(
            (pl.col("half") == "test") & (pl.col("score") >= cut)
        ).height
        f = ic.filter((pl.col("setting") == s) & (pl.col("score") >= cut))
        found_test = (
            inst_s.filter(pl.col("half") == "test")
            .join(
                f.select("query", "pfam_acc", "t1_start", "t1_end"),
                on=["query", "pfam_acc", "t1_start", "t1_end"],
                how="semi",
            )
            .height
        )
        ctrl = inst.filter(pl.col("role") == "control")
        ctrl_any = ctrl.join(
            ic.filter(pl.col("setting") == s).select(
                "query", "pfam_acc", "t1_start", "t1_end"
            ),
            on=["query", "pfam_acc", "t1_start", "t1_end"],
            how="semi",
        ).height
        rows.append(
            {
                "setting": s,
                "n_calls": cnt["n_calls"],
                "real_queries_with_call": cnt["real_queries_with_call"],
                "decoy_queries_with_call": cnt["decoy_queries_with_call"],
                "controls_found_any_score": ctrl_any,
                "controls": ctrl.height,
                "cutoff_tune": cut,
                "decoy_transfers_test": dec_test,
                "decoys_test": n_dec["test"],
                "instances_test": inst_s.filter(pl.col("half") == "test").height,
                "found_test": found_test,
            }
        )
    summ = pl.DataFrame(rows)
    if not a.read_test_half:
        summ = summ.drop("decoy_transfers_test", "found_test", strict=False)
    else:
        summ = summ.with_columns(
            recall_test=pl.col("found_test") / pl.col("instances_test"),
            decoy_rate_test=pl.col("decoy_transfers_test") / pl.col("decoys_test"),
        )
    summ.write_csv(a.out / "setting_summary.csv")
    ctrl_tab = inst.filter(pl.col("role") == "control").join(
        ic.select(
            "setting",
            "query",
            "pfam_acc",
            "t1_start",
            "t1_end",
            "landing",
            "coverage",
            "score",
        ),
        on=["query", "pfam_acc", "t1_start", "t1_end"],
        how="left",
    )
    ctrl_tab.write_csv(a.out / "controls.csv")

    # Unique finds (P4). The setting is chosen on the tune half; the test half is read only
    # for that setting, and only with --read-test-half.
    kmer_settings = [s for s in settings if s.startswith("kmerseek ")]
    comparators = [s for s in settings if not s.startswith("kmerseek ")]
    found = found_at_cutoff(ic, cutoffs)
    p4 = is_p4(inst_s).filter("p4")
    halves = ("tune", "test") if a.read_test_half else ("tune",)
    tables = {}
    for h in halves:
        decoys = q.filter(
            pl.col("is_decoy") & (pl.col("half") == h) & (pl.col("role") == "sample")
        )["query"].to_list()
        uq = unique_per_query(
            found, p4.filter(pl.col("half") == h), kmer_settings, comparators
        )
        dt = decoy_transfers_per_query(
            {s: dec_by_setting[s] for s in kmer_settings}, cutoffs, decoys, h
        )
        tables[h] = (uq, dt)
    tune = unique_vs_expected(*tables["tune"]).sort(
        "unique_minus_expected", descending=True
    )
    tune.write_csv(a.out / "unique_finds_tune.csv")
    p4_rows = {"comparators": comparators, "kmerseek_settings": len(kmer_settings)}
    if tune.height:
        chosen = choose_p4_setting(tune)
        p4_rows.update(
            chosen=chosen,
            **{
                f"tune_{k}": v
                for k, v in tune.filter(pl.col("setting") == chosen)
                .row(0, named=True)
                .items()
                if k != "setting"
            },
        )
        if a.read_test_half:
            uq, dt = tables["test"]
            u = uq.filter(pl.col("setting") == chosen)["n_unique"].to_list()
            t = dt.filter(pl.col("setting") == chosen)["n_transfers"].to_list()
            point, low, high = bootstrap_interval(u, t)
            p4_rows.update(
                test_unique_finds=sum(u),
                test_p4_queries=len(u),
                test_decoy_rate=sum(t) / len(t) if t else None,
                test_unique_minus_expected=point,
                test_ci95_low=low,
                test_ci95_high=high,
                p4_holds=low > 0,
            )
            (
                found.filter(pl.col("setting") == chosen)
                .join(p4.filter(pl.col("half") == "test"), on=INSTANCE_KEY)
                .join(
                    found.filter(pl.col("setting").is_in(comparators))
                    .select(INSTANCE_KEY)
                    .unique(),
                    on=INSTANCE_KEY,
                    how="anti",
                )
                .write_csv(a.out / "unique_finds_test_instances.csv")
            )
    (a.out / "p4.json").write_text(json.dumps(p4_rows, indent=1, default=str))
    print(tune.head(10))
    print(json.dumps(p4_rows, indent=1, default=str))
    with pl.Config(tbl_rows=400, tbl_cols=20, tbl_width_chars=250):
        print(pc)
        print(summ)


if __name__ == "__main__":
    main()
