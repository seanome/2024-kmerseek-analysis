#!/usr/bin/env python3
"""Score the midi-plus calls against the target proteins' exons (notebook 356).

Inputs:
  --calls-dir   out/ of scripts/extract_356_ortholog_calls.py (calls/*.parquet, plddt.parquet)
  notebooks/356_data/exons.parquet and orthologs_1to1.tsv (scripts/build_356_exon_inputs.py)
  the QfO target FASTAs, for protein lengths and the Kyte-Doolittle scan

Every interval is 0-based and end-exclusive. kmerseek's target coordinates already are; the
comparator tables are 1-based inclusive, so their starts move down by one. The Kyte-Doolittle
scan (swissprot_control_utils.kd_transmem_calls, window 19, threshold 1.6, as nb 231 and nb
244) is run on the target protein and scored like a tool.

Per (pair, exon, tool, setting), from the calls between that query and its ortholog:
  inside = overlap / call length, cover = overlap / exon length, iou = overlap / union
  lands  = some call has inside >= 0.8 and cover >= 0.3 (nb 244's rule)
  outcome: lands; spills (no call lands and the best-IoU call overlapping the exon has
           less than half of itself inside it); partial (overlaps, at least half inside,
           does not land); no call (nothing overlaps the exon)
  p_place = smallest placement p-value among the landing calls: the share of start
           positions on the target at which a window of the call's length would also
           land (pre-registration, "Placement null"); place_ok = p_place < 0.05

kmerseek is scored per setting and as the best of all settings: it lands on an exon when
any setting does. Its random-protein null moves each exon's interval, at the same
coordinates, onto the target of a randomly drawn other pair of the same species that is
long enough, and asks whether any kmerseek setting's call between that pair lands on it.

The verdict set is internal exons of 60 residues or fewer on pairs where both proteins
have an AlphaFold model in the run (amendment 6). The rules are in
notebooks/356_data/preregistered.md.

Writes into --out-dir (default tables/356, or tables/356/pilot with --pilot):
  exon_tool.parquet       one row per (pair, exon, tool, setting) for the comparators,
                          kmerseek's best of all, and kmerseek's best single setting
  kmerseek_settings.parquet  landing rate of every kmerseek setting on the verdict set
  summary_by_length.csv   landing / spill / no-call shares per tool and exon-length bin
  summary_by_disorder.csv the same per pLDDT class, for exons of 60 residues or fewer
  null_kmerseek_best.csv  the random-protein null draws
  verdict.json            the three rules, with the numbers behind them
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import swissprot_control_utils as su  # noqa: E402

DATA = REPO / "notebooks" / "356_data"
QFO = Path.home() / (
    "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota"
)
TARGET_FASTA = {"chicken": "UP000000539_9031", "zebrafish": "UP000000437_7955"}

INSIDE_MIN = 0.8  # nb 244, criterion 2
COVER_MIN = 0.3
SPILL_INSIDE = 0.5  # nb 244's "spills": less than half the call inside
P_PLACE_MAX = 0.05  # amendment 1
SHORT_MAX = 60  # decision rule: internal exons of 60 residues or fewer
PLDDT_ORDERED = 70.0
LENGTH_BINS = [
    (1, 20, "<=20"),
    (21, 40, "21-40"),
    (41, 60, "41-60"),
    (61, 120, "61-120"),
    (121, 10**9, ">120"),
]
#: Tools the rules name. jackhmmer and HHblits are scored and reported, not in the rules.
STRUCTURE_TOOLS = ("foldseek", "reseek", "prostt5")
SEQUENCE_TOOLS = ("hmmer3_phmmer", "mmseqs2_seqseq", "mmseqs2_iterative")
EXTRA_TOOLS = ("hmmer3_jackhmmer", "hhblits")
KD = "kd_scan"
KM_BEST = "kmerseek_best_of_all"
KM_SINGLE = "kmerseek_best_single"
PASS_KM = 0.50
PASS_STRUCT = 0.15
PASS_KD_RATIO = 2.0
EMPTY_QUESTION = 0.10  # Stop 0: below this, stop
N_NULL = 1_000
SEED = 356


def read_fasta(path: Path) -> dict[str, str]:
    seqs, acc = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split("|")[1]
                seqs[acc] = []
            else:
                seqs[acc].append(line.strip())
    return {a: "".join(s) for a, s in seqs.items()}


def pilot_pairs(pairs: pl.DataFrame, n: int) -> pl.DataFrame:
    """n pairs in a fixed order: the SHA-1 of 'query|species|target'."""
    return (
        pairs.with_columns(
            _h=pl.concat_str(
                ["query_acc", "species", "target_acc"], separator="|"
            ).map_elements(
                lambda s: hashlib.sha1(s.encode()).hexdigest(), return_dtype=pl.String
            )
        )
        .sort("_h")
        .head(n)
        .drop("_h")
    )


def length_bin(col: str = "exon_len") -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(LENGTH_BINS):
        e = pl.when(pl.col(col).is_between(lo, hi)).then(pl.lit(name)).otherwise(e)
    return e


def load_exons(pairs: pl.DataFrame, plddt: pl.DataFrame) -> pl.DataFrame:
    ex = pl.read_parquet(DATA / "exons.parquet").join(
        pairs, on=["species", "target_acc"], how="inner"
    )
    tm = (
        plddt.filter(pl.col("side") == "target")
        .group_by("species", "accession")
        .agg(target_model=pl.col("has_model").any())
    )
    qm = (
        plddt.filter(pl.col("side") == "query")
        .group_by("accession")
        .agg(query_model=pl.col("has_model").any())
    )
    res = plddt.filter((pl.col("side") == "target") & pl.col("has_model")).select(
        "species", pl.col("accession").alias("target_acc"), "residue", "plddt"
    )
    ex_pl = (
        ex.select("species", "target_acc", "prot_start", "prot_end")
        .join(res, on=["species", "target_acc"])
        .filter(
            (pl.col("residue") > pl.col("prot_start"))
            & (pl.col("residue") <= pl.col("prot_end"))
        )
        .group_by("species", "target_acc", "prot_start", "prot_end")
        .agg(exon_plddt=pl.col("plddt").mean())
    )
    return (
        ex.join(
            tm.rename({"accession": "target_acc"}),
            on=["species", "target_acc"],
            how="left",
        )
        .join(qm.rename({"accession": "query_acc"}), on="query_acc", how="left")
        .join(ex_pl, on=["species", "target_acc", "prot_start", "prot_end"], how="left")
        .with_columns(
            pl.col("target_model").fill_null(False),
            pl.col("query_model").fill_null(False),
            length_bin=length_bin(),
            short_internal=pl.col("internal") & (pl.col("exon_len") <= SHORT_MAX),
        )
        .with_columns(
            both_models=pl.col("target_model") & pl.col("query_model"),
            disorder=pl.when(pl.col("exon_plddt").is_null())
            .then(pl.lit("no model"))
            .when(pl.col("exon_plddt") >= PLDDT_ORDERED)
            .then(pl.lit("ordered"))
            .otherwise(pl.lit("disordered")),
        )
        .with_columns(verdict_set=pl.col("short_internal") & pl.col("both_models"))
    )


def reseek_f_accession_check(calls: pl.DataFrame, lengths: pl.DataFrame) -> dict:
    """Rows whose target end is past the protein's end, by accession shape (memory note:
    normalize_reseek.awk read F<digit> TrEMBL accessions as fragment numbers)."""
    c = calls.join(lengths, on=["species", "target_acc"])
    past = c.filter(pl.col("tend") > pl.col("target_len"))
    return {
        "n_rows": c.height,
        "n_past_end": past.height,
        "n_past_end_F_digit_accession": past.filter(
            pl.col("target_acc").str.contains(r"^F[0-9]")
        ).height,
    }


def fix_reseek(calls: pl.DataFrame, lengths: pl.DataFrame) -> pl.DataFrame:
    """Undo the (d - 1) * 200 shift on F<d>... accessions, only where the call runs past the
    end of the protein (the stored table may already be corrected for some rows)."""
    d = pl.col("target_acc").str.slice(1, 1).cast(pl.Int64, strict=False)
    shift = (d - 1) * 200
    c = calls.join(lengths, on=["species", "target_acc"], how="left")
    bad = pl.col("target_acc").str.contains(r"^F[2-9]") & (
        pl.col("tend") > pl.col("target_len")
    )
    return c.with_columns(
        tstart=pl.when(bad).then(pl.col("tstart") - shift).otherwise(pl.col("tstart")),
        tend=pl.when(bad).then(pl.col("tend") - shift).otherwise(pl.col("tend")),
    ).drop("target_len")


def load_calls(
    calls_dir: Path, pairs: pl.DataFrame, lengths: pl.DataFrame, seqs: dict
) -> tuple[pl.DataFrame, dict]:
    """All calls on the chosen pairs, 0-based end-exclusive on the target."""
    frames, notes = [], {}
    for p in sorted((calls_dir / "calls").glob("*.parquet")):
        tool, arm, species = p.stem.split(".")
        df = pl.read_parquet(p)
        if df.height == 0:
            continue
        df = df.select(
            "query_acc", "target_acc", "tstart", "tend", "score"
        ).with_columns(species=pl.lit(species), tool=pl.lit(tool), arm=pl.lit(arm))
        if tool != "kmerseek":
            df = df.with_columns(tstart=pl.col("tstart") - 1)
        if tool == "reseek":
            notes[f"reseek_{species}_before_fix"] = reseek_f_accession_check(
                df, lengths
            )
            df = fix_reseek(df, lengths)
            notes[f"reseek_{species}_after_fix"] = reseek_f_accession_check(df, lengths)
        frames.append(
            df.join(pairs, on=["species", "query_acc", "target_acc"], how="semi")
        )
    calls = pl.concat(frames, how="vertical_relaxed")

    # The no-search control: Kyte-Doolittle segments on each pair's target protein.
    kd_rows = []
    for r in pairs.iter_rows(named=True):
        s = seqs[r["species"]][r["target_acc"]]
        k = su.kd_transmem_calls({r["target_acc"]: s}, 19, 1.6)
        if k.height:
            kd_rows.append(
                k.select(
                    pl.lit(r["query_acc"]).alias("query_acc"),
                    pl.col("query_acc").alias("target_acc"),
                    (pl.col("qstart") - 1).alias("tstart"),
                    pl.col("qend").alias("tend"),
                    "score",
                    species=pl.lit(r["species"]),
                    tool=pl.lit(KD),
                    arm=pl.lit("w19_t1.6"),
                )
            )
    if kd_rows:
        calls = pl.concat([calls, pl.concat(kd_rows)], how="vertical_relaxed")
    return (
        calls.with_columns(
            pl.col("tstart").cast(pl.Int64), pl.col("tend").cast(pl.Int64)
        ),
        notes,
    )


def p_place_expr() -> pl.Expr:
    """Placement p-value of a landing call (pre-registration, "Placement null")."""
    L = pl.col("tend") - pl.col("tstart")
    E = pl.col("prot_end") - pl.col("prot_start")
    need = pl.max_horizontal((INSIDE_MIN * L).ceil(), (COVER_MIN * E).ceil()).cast(
        pl.Int64
    )
    lo = pl.max_horizontal(pl.col("prot_start") + need - L, pl.lit(0))
    hi = pl.min_horizontal(pl.col("prot_end") - need, pl.col("target_len") - L)
    count = (
        pl.when(need <= pl.min_horizontal(L, E))
        .then(pl.max_horizontal(hi - lo + 1, pl.lit(0)))
        .otherwise(0)
    )
    return count / (pl.col("target_len") - L + 1)


def score(calls: pl.DataFrame, exons: pl.DataFrame) -> pl.DataFrame:
    """One row per (pair, exon, tool, arm) that had any call on the pair."""
    key = ["species", "query_acc", "target_acc"]
    ex = exons.select(*key, "exon_id", "prot_start", "prot_end", "target_len")
    ov = (
        pl.min_horizontal("tend", "prot_end")
        - pl.max_horizontal("tstart", "prot_start")
    ).clip(lower_bound=0)
    m = (
        calls.join(ex, on=key)
        .with_columns(ov=ov)
        .with_columns(
            inside=pl.col("ov")
            / (pl.col("tend") - pl.col("tstart")).clip(lower_bound=1),
            cover=pl.col("ov") / (pl.col("prot_end") - pl.col("prot_start")),
            iou=pl.col("ov")
            / (
                pl.max_horizontal("tend", "prot_end")
                - pl.min_horizontal("tstart", "prot_start")
            ),
        )
        .with_columns(
            lands=(pl.col("inside") >= INSIDE_MIN) & (pl.col("cover") >= COVER_MIN)
        )
        .with_columns(p_place=pl.when("lands").then(p_place_expr()))
    )
    g = key + ["exon_id", "tool", "arm"]
    best = (
        m.filter(pl.col("ov") > 0)
        .sort(["iou", "score", "tstart"], descending=[True, True, False])
        .group_by(g, maintain_order=True)
        .agg(
            n_overlapping=pl.len(),
            lands=pl.col("lands").any(),
            p_place=pl.col("p_place").min(),
            best_iou=pl.col("iou").first(),
            best_inside=pl.col("inside").first(),
            best_cover=pl.col("cover").first(),
            best_tstart=pl.col("tstart").first(),
            best_tend=pl.col("tend").first(),
            land_iou=pl.col("iou").filter("lands").max(),
        )
    )
    # Pairs where the tool made a call somewhere, but none on this exon: "no call".
    has_call = calls.select(*key, "tool", "arm").unique()
    full = (
        has_call.join(ex.select(*key, "exon_id"), on=key)
        .join(best, on=g, how="left")
        .with_columns(
            pl.col("n_overlapping").fill_null(0), pl.col("lands").fill_null(False)
        )
    )
    return full


def with_outcome(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        outcome=pl.when(pl.col("lands"))
        .then(pl.lit("lands"))
        .when(pl.col("n_overlapping") == 0)
        .then(pl.lit("no call"))
        .when(pl.col("best_inside") < SPILL_INSIDE)
        .then(pl.lit("spills"))
        .otherwise(pl.lit("partial")),
        place_ok=pl.col("lands") & (pl.col("p_place") < P_PLACE_MAX),
    )


def complete(rows: pl.DataFrame, exons: pl.DataFrame, tools: list[str]) -> pl.DataFrame:
    """Every exon for every tool: a tool with no call at all on a pair scores 'no call'."""
    key = ["species", "query_acc", "target_acc", "exon_id"]
    grid = exons.select(key).join(pl.DataFrame({"tool": tools}), how="cross")
    out = grid.join(rows, on=key + ["tool"], how="left").with_columns(
        pl.col("n_overlapping").fill_null(0), pl.col("lands").fill_null(False)
    )
    return with_outcome(out).join(exons, on=key, how="left")


def kmerseek_best_of_all(sc: pl.DataFrame) -> pl.DataFrame:
    km = sc.filter(pl.col("tool") == "kmerseek")
    km = km.with_columns(
        place_ok_arm=pl.col("lands") & (pl.col("p_place") < P_PLACE_MAX)
    )
    key = ["species", "query_acc", "target_acc", "exon_id"]
    return (
        km.sort(["lands", "place_ok_arm", "best_iou"], descending=True, nulls_last=True)
        .group_by(key, maintain_order=True)
        .agg(
            n_overlapping=pl.col("n_overlapping").sum(),
            lands=pl.col("lands").any(),
            n_settings_land=pl.col("lands").sum(),
            p_place=pl.col("p_place").filter("lands").min(),
            best_iou=pl.col("best_iou").first(),
            best_inside=pl.col("best_inside").first(),
            best_cover=pl.col("best_cover").first(),
            best_tstart=pl.col("best_tstart").first(),
            best_tend=pl.col("best_tend").first(),
            land_iou=pl.col("land_iou").max(),
            arm=pl.col("arm").first(),
        )
        .with_columns(tool=pl.lit(KM_BEST))
    )


def random_protein_null(
    calls: pl.DataFrame,
    exons: pl.DataFrame,
    n_draws: int,
    seed: int,
    arm: str | None = None,
    require_place: bool = False,
) -> np.ndarray:
    """kmerseek landing rate on decoy exons, one value per draw.

    lands[e, j] is whether a kmerseek call of pair j lands on exon e's interval placed at
    the same coordinates on pair j's target. Each draw picks, per exon, one other pair of the
    same species whose target is long enough. With arm=None every setting's calls count
    (best of all settings); otherwise only that setting's. With require_place, a landing
    call also needs p_place < 0.05, computed on the decoy protein's length. The same seed
    gives the same decoy choices for every variant."""
    ev = exons.filter("verdict_set")
    km = calls.filter(pl.col("tool") == "kmerseek")
    if arm is not None:
        km = km.filter(pl.col("arm") == arm)
    km = km.select("species", "query_acc", "target_acc", "tstart", "tend").unique()
    rng = np.random.default_rng(seed)
    total = 0  # exons with at least one long-enough other pair to move onto
    landed = np.zeros(n_draws)
    for species in ev["species"].unique().sort():
        e = ev.filter(pl.col("species") == species)
        a = e["prot_start"].to_numpy()
        b = e["prot_end"].to_numpy()
        own = (e["query_acc"] + "|" + e["target_acc"]).to_list()
        pairs = (
            exons.filter(pl.col("species") == species)
            .select("query_acc", "target_acc", "target_len")
            .unique()
            .sort("query_acc")
        )
        names = (pairs["query_acc"] + "|" + pairs["target_acc"]).to_list()
        tlen = pairs["target_len"].to_numpy()
        kmp = km.filter(pl.col("species") == species)
        by_pair = {
            (r[0] + "|" + r[1]): r
            for r in kmp.group_by("query_acc", "target_acc")
            .agg("tstart", "tend")
            .iter_rows()
        }
        lands = np.zeros((len(a), len(names)), dtype=bool)
        for j, nm in enumerate(names):
            if nm not in by_pair:
                continue
            s = np.asarray(by_pair[nm][2])[None, :]
            t = np.asarray(by_pair[nm][3])[None, :]
            ov = np.clip(np.minimum(t, b[:, None]) - np.maximum(s, a[:, None]), 0, None)
            L = np.maximum(t - s, 1)
            E = (b - a)[:, None]
            hit = (ov >= INSIDE_MIN * L) & (ov >= COVER_MIN * E)
            if require_place:
                need = np.maximum(np.ceil(INSIDE_MIN * L), np.ceil(COVER_MIN * E))
                lo = np.maximum(a[:, None] + need - L, 0)
                hi = np.minimum(b[:, None] - need, tlen[j] - L)
                count = np.where(
                    need <= np.minimum(L, E), np.maximum(hi - lo + 1, 0), 0
                )
                hit &= count / (tlen[j] - L + 1) < P_PLACE_MAX
            lands[:, j] = hit.any(axis=1)
        eligible = (tlen[None, :] >= b[:, None]) & (
            np.array(names)[None, :] != np.array(own)[:, None]
        )
        total += int(eligible.any(axis=1).sum())
        for d in range(n_draws):
            u = rng.random(eligible.shape) * eligible
            j = u.argmax(axis=1)
            ok = eligible[np.arange(len(a)), j]
            landed[d] += (lands[np.arange(len(a)), j] & ok).sum()
    return landed / max(total, 1)


def rate(df: pl.DataFrame, col: str = "lands") -> tuple[int, int, float | None]:
    n = df.height
    k = int(df[col].sum()) if n else 0
    return k, n, (k / n if n else None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--calls-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--pilot", type=int, default=None, help="score only this many pairs"
    )
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    out = args.out_dir or (REPO / "tables" / "356" / ("pilot" if args.pilot else ""))
    out.mkdir(parents=True, exist_ok=True)

    pairs = pl.read_csv(DATA / "orthologs_1to1.tsv", separator="\t")
    if args.pilot:
        pairs = pilot_pairs(pairs, args.pilot)
    seqs = {sp: read_fasta(QFO / f"{stem}.fasta") for sp, stem in TARGET_FASTA.items()}
    lengths = pl.DataFrame(
        [
            {"species": sp, "target_acc": a, "target_len": len(s)}
            for sp, d in seqs.items()
            for a, s in d.items()
        ]
    )
    plddt = pl.read_parquet(args.calls_dir / "plddt.parquet")
    exons = load_exons(pairs, plddt)
    calls, notes = load_calls(args.calls_dir, pairs, lengths, seqs)

    sc = score(calls, exons)
    single_tools = [
        t
        for t in (*STRUCTURE_TOOLS, *SEQUENCE_TOOLS, *EXTRA_TOOLS, KD)
        if t in set(calls["tool"].unique())
    ]
    comp = sc.filter(pl.col("tool").is_in(single_tools))
    best = kmerseek_best_of_all(sc)

    # Every kmerseek setting's landing rate on the verdict set, and the best single one.
    vs_key = exons.filter("verdict_set").select(
        "species", "query_acc", "target_acc", "exon_id"
    )
    n_vs = vs_key.height
    settings = (
        sc.filter(pl.col("tool") == "kmerseek")
        .join(vs_key, on=vs_key.columns)
        .group_by("arm")
        .agg(n_land=pl.col("lands").sum())
        .with_columns(n_exons=pl.lit(n_vs), rate=pl.col("n_land") / max(n_vs, 1))
        .sort(["rate", "arm"], descending=[True, False])
    )
    settings.write_parquet(out / "kmerseek_settings.parquet")
    best_arm = settings["arm"][0] if settings.height else None
    single = sc.filter(
        (pl.col("tool") == "kmerseek") & (pl.col("arm") == best_arm)
    ).with_columns(tool=pl.lit(KM_SINGLE))

    tools = [KM_BEST, KM_SINGLE, *single_tools]
    rows = pl.concat(
        [comp, best.select(comp.columns), single.select(comp.columns)],
        how="vertical_relaxed",
    )
    table = complete(rows, exons, tools)
    table.write_parquet(out / "exon_tool.parquet")

    def summarize(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        return (
            df.group_by(by + ["tool"])
            .agg(
                n_exons=pl.len(),
                n_lands=pl.col("lands").sum(),
                n_spills=(pl.col("outcome") == "spills").sum(),
                n_partial=(pl.col("outcome") == "partial").sum(),
                n_no_call=(pl.col("outcome") == "no call").sum(),
                n_place_ok=pl.col("place_ok").sum(),
                median_land_iou=pl.col("land_iou").median(),
            )
            .with_columns(land_rate=pl.col("n_lands") / pl.col("n_exons"))
            .sort(by + ["tool"])
        )

    internal_models = table.filter(pl.col("internal") & pl.col("both_models"))
    summarize(internal_models, ["length_bin"]).write_csv(out / "summary_by_length.csv")
    summarize(table.filter("verdict_set"), ["disorder"]).write_csv(
        out / "summary_by_disorder.csv"
    )
    summarize(table.filter(pl.col("short_internal")), []).write_csv(
        out / "summary_short_internal_all_pairs.csv"
    )

    null = random_protein_null(calls, exons, args.n_null, SEED)
    # Added after Stop 0 (not in the pre-registration): the same null for the two other
    # kmerseek numbers the verdict quotes, so each has its own chance level.
    null_place = random_protein_null(
        calls, exons, args.n_null, SEED, require_place=True
    )
    null_single = random_protein_null(calls, exons, args.n_null, SEED, arm=best_arm)
    pl.DataFrame(
        {
            "draw": np.arange(len(null)),
            "best_of_all": null,
            "best_of_all_p_place_lt_0.05": null_place,
            f"single_{best_arm}": null_single,
        }
    ).write_csv(out / "null_kmerseek_variants.csv")
    pl.DataFrame({"draw": np.arange(len(null)), "land_rate": null}).write_csv(
        out / "null_kmerseek_best.csv"
    )

    v = table.filter("verdict_set")
    r = {t: rate(v.filter(pl.col("tool") == t)) for t in tools}
    rp = {t: rate(v.filter(pl.col("tool") == t), "place_ok") for t in tools}
    km_k, km_n, km_rate = r[KM_BEST]
    kd_rate = r[KD][2] if KD in r else None
    km_place_rate = rp[KM_BEST][2]
    struct = {t: r[t] for t in STRUCTURE_TOOLS if t in r}
    null_999 = float(np.quantile(null, 0.999)) if len(null) else None
    verdict = {
        "stage": f"pilot, {args.pilot} pairs" if args.pilot else "full",
        "n_pairs": pairs.height,
        "n_verdict_exons": km_n,
        "verdict_set": "internal coding exons of <= 60 residues, both proteins with an "
        "AlphaFold model in the run",
        "rule_1": {
            "kmerseek_best_of_all_lands": [km_k, km_n, km_rate],
            "threshold": PASS_KM,
            "pass": km_rate is not None and km_rate >= PASS_KM,
        },
        "rule_2": {
            "structure_tools_land": {t: list(x) for t, x in struct.items()},
            "threshold": PASS_STRUCT,
            "missing_tools": [t for t in STRUCTURE_TOOLS if t not in struct],
            "pass": len(struct) == len(STRUCTURE_TOOLS)
            and all(x[2] is not None and x[2] < PASS_STRUCT for x in struct.values()),
        },
        "rule_3": {
            "kmerseek_lands_with_p_place_lt_0.05": list(rp[KM_BEST]),
            "kd_scan_lands": list(r.get(KD, (0, 0, None))),
            "ratio": (km_place_rate / kd_rate) if kd_rate else None,
            "threshold_ratio": PASS_KD_RATIO,
            "pass": km_place_rate is not None
            and (
                kd_rate == 0
                and km_place_rate > 0
                or kd_rate
                and km_place_rate / kd_rate >= PASS_KD_RATIO
            ),
        },
        "random_protein_null": {
            "n_draws": len(null),
            "median": float(np.median(null)),
            "quantile_0.999": null_999,
            "observed_above_0.999": km_rate is not None
            and null_999 is not None
            and km_rate > null_999,
        },
        "random_protein_null_added_after_stop0": {
            "best_of_all_p_place_lt_0.05": {
                "observed": km_place_rate,
                "median": float(np.median(null_place)),
                "quantile_0.999": float(np.quantile(null_place, 0.999)),
            },
            "best_single_setting": {
                "observed": r[KM_SINGLE][2],
                "median": float(np.median(null_single)),
                "quantile_0.999": float(np.quantile(null_single, 0.999)),
            },
        },
        "kmerseek_best_single_setting": {"arm": best_arm, "lands": list(r[KM_SINGLE])},
        "other_tools_land": {
            t: list(r[t]) for t in tools if t not in (KM_BEST, *STRUCTURE_TOOLS)
        },
        "kill_condition_kd_at_kmerseek_rate": kd_rate is not None
        and km_rate is not None
        and kd_rate >= km_rate,
        "empty_question_kmerseek_below_0.10": km_rate is not None
        and km_rate < EMPTY_QUESTION,
        "notes": notes,
    }
    verdict["pass"] = all(verdict[k]["pass"] for k in ("rule_1", "rule_2", "rule_3"))
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
