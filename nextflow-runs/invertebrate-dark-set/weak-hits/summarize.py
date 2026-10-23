#!/usr/bin/env python3
"""Weak-hit accuracy: tables, checks and figures from score-bands' output.

    python weak-hits/summarize.py            (from nextflow-runs/invertebrate-dark-set)

Reads weak-hits/data/<species>_weak_hits.{tsv,json} (copied from
$SCRATCH/invertebrate-dark-set/results-0.4/accuracy/weak-hits/ on Sherlock) and writes:

  weak-hits/summary.txt                       every number the PR and the cards quote
  weak-hits/weak_hits_all_sources.parquet     every source, set, split and band, with precision
  weak-hits/figures/*.png|pdf|svg             the two figures
  ../../results/leaderboard/weak_hits/weak_hits.parquet   the site's table
  ../../results/leaderboard/weak_hits/provenance.json     sha256 of inputs and outputs

The rules (bands, the two calls, the tune/test split, how the card's kmerseek setting is
chosen) are in weak-hits/README.md and were written before any of these numbers existed.
Exits 1 when the recount of PR 133's four rows does not match.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "notebooks"))
SPECIES = ["human", "worm", "yeast", "borrelia"]
SPECIES_NAME = {"human": "human", "worm": "C. elegans", "yeast": "S. cerevisiae",
                "borrelia": "B. burgdorferi"}
BANDS = ["le_0.001", "0.001_1", "1_10"]
BAND_WORDS = {"le_0.001": "E ≤ 0.001", "0.001_1": "0.001 < E ≤ 1", "1_10": "1 < E ≤ 10"}
SEQ_TOOLS = {"phmmer": "hmmer3_phmmer", "jackhmmer": "hmmer3_jackhmmer",
             "mmseqs2": "mmseqs2_iterative"}
TOOL_WORDS = {"phmmer": "phmmer", "jackhmmer": "jackhmmer (3 rounds)",
              "mmseqs2": "MMseqs2 (-s 7, 3 rounds)"}
MIN_N = 30  # fewer judged hits than this: shown grey, never chosen

# PR 133's table (its body, 2026-10-13): dark at E <= 1, all proteins with a known family.
# kmerseek right / wrong among best regions at E <= 1; phmmer right / wrong at E 1-10.
PR133 = {
    "human": dict(n=1071, findable=184, setting="hp_lehninger2.k17.s1", k=(0, 270), p=(24, 716)),
    "worm": dict(n=1892, findable=421, setting="hp_lehninger_c_nonpolar2.k19.s1", k=(6, 397),
                 p=(59, 1371)),
    "yeast": dict(n=342, findable=89, setting="hp_lehninger2.k17.s1", k=(1, 46), p=(10, 263)),
    "borrelia": dict(n=165, findable=20, setting="hp_lehninger_c_nonpolar2.k19.s1", k=(0, 13),
                     p=(1, 131)),
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    inputs = {}
    frames, js = [], {}
    for sp in SPECIES:
        t = HERE / "data" / f"{sp}_weak_hits.tsv"
        j = HERE / "data" / f"{sp}_weak_hits.json"
        inputs[str(t.relative_to(REPO))] = sha256(t)
        inputs[str(j.relative_to(REPO))] = sha256(j)
        frames.append(pl.read_csv(t, separator="\t"))
        js[sp] = json.loads(j.read_text())
    long = pl.concat(frames)
    long = long.with_columns(
        (pl.col("family") + pl.col("clan") + pl.col("wrong")).alias("n_judged"),
        (pl.col("family") + pl.col("clan")).alias("right_loose"),
        pl.col("strict_right").alias("right_strict"),
    )

    # ---- 1. recount PR 133's four rows before reading anything else ----
    say("Recount of PR 133's table (dark at E <= 1, all proteins with a known Pfam family)")
    ok = True
    for sp, w in PR133.items():
        d = long.filter((pl.col("species") == sp) & (pl.col("query_set") == "dark_1")
                        & (pl.col("split") == "all"))
        ks = d.filter(pl.col("source") == "kmerseek:" + w["setting"])
        kb = ks.filter(pl.col("band").is_in(["le_0.001", "0.001_1"]))
        k = (int(kb["right_loose"].sum()), int(kb["wrong"].sum()))
        ph = d.filter(pl.col("source") == "phmmer")
        pb = ph.filter(pl.col("band") == "1_10")
        p = (int(pb["right_loose"].sum()), int(pb["wrong"].sum()))
        lower = int(ph.filter(pl.col("band") != "1_10")["n_best_hit_in_band"].sum())
        n, f = int(ks["n_proteins"][0]), int(ks["n_findable"][0])
        match = (n, f, k, p, lower) == (w["n"], w["findable"], w["k"], w["p"], 0)
        ok &= match
        say(f"  {sp:9s} dark {n:,} (PR 133 {w['n']:,}), findable {f} ({w['findable']}), "
            f"kmerseek {w['setting']} right/wrong at E <= 1 {k[0]}/{k[1]} "
            f"({w['k'][0]}/{w['k'][1]}), phmmer at E 1-10 {p[0]}/{p[1]} ({w['p'][0]}/{w['p'][1]}), "
            f"phmmer best hits at E <= 1 on dark proteins {lower} (0): "
            + ("MATCH" if match else "DIFFERENT"))
    if not ok:
        say("Recount does not match PR 133; stopping.")
        (HERE / "summary.txt").write_text("\n".join(out_lines) + "\n")
        sys.exit(1)

    # ---- 2. precision with 95% intervals ----
    rows = []
    for r in long.iter_rows(named=True):
        n = r["n_judged"]
        ls, hs = wilson(r["right_strict"], n)
        ll, hl = wilson(r["right_loose"], n)
        rows.append(dict(r, precision_strict=r["right_strict"] / n if n else None,
                         strict_lo=ls, strict_hi=hs,
                         precision_loose=r["right_loose"] / n if n else None,
                         loose_lo=ll, loose_hi=hl))
    allp = pl.DataFrame(rows, infer_schema_length=None)
    allp.write_parquet(HERE / "weak_hits_all_sources.parquet")

    say()
    say("Findable: dark proteins (E <= 1) whose Pfam family is anywhere in the "
        "clade-excluded reference")
    for sp in SPECIES:
        d = allp.filter((pl.col("species") == sp) & (pl.col("query_set") == "dark_1")
                        & (pl.col("split") == "all") & (pl.col("source") == "phmmer")
                        & (pl.col("band") == "1_10"))
        n, f = d["n_proteins"][0], d["n_findable"][0]
        say(f"  {sp:9s} {f:,} of {n:,} ({100 * f / n:.0f}%)")

    say()
    say("Sequence tools' query stretch from the phmmer re-alignment "
        "(pairs with best E <= 10)")
    for sp in SPECIES:
        for tool, st in js[sp]["realign"].items():
            say(f"  {sp:9s} {tool:9s} pairs {st['pairs']:,}: a realigned domain overlaps the "
                f"tool's target stretch {st['overlapping_target_range']:,}, same target stretch "
                f"{st['same_target_range']:,}, none {st['no_overlap']:,}")

    # ---- 3. kmerseek setting per species, chosen on the tune half ----
    say()
    say("kmerseek setting per species: highest strict precision at E 1-10 on dark tune "
        f"proteins, among settings with >= {MIN_N} judged hits there")
    chosen = {}
    for sp in SPECIES:
        tune = allp.filter((pl.col("species") == sp) & (pl.col("query_set") == "dark_1")
                           & (pl.col("split") == "tune") & (pl.col("band") == "1_10")
                           & pl.col("source").str.starts_with("kmerseek:"))
        n_set = tune.height
        elig = tune.filter(pl.col("n_judged") >= MIN_N)
        if elig.height == 0:
            say(f"  {sp:9s} no setting has {MIN_N} judged hits on the tune half "
                f"({n_set} settings); none chosen")
            continue
        top = elig.sort(["precision_strict", "n_judged", "source"],
                        descending=[True, True, False]).row(0, named=True)
        chosen[sp] = top["source"]
        say(f"  {sp:9s} {top['source'].removeprefix('kmerseek:')} (of {elig.height} eligible, "
            f"{n_set} settings): tune {top['right_strict']}/{top['n_judged']}")
    allp = allp.with_columns(
        pl.struct("species", "source").map_elements(
            lambda s: chosen.get(s["species"]) == s["source"], return_dtype=pl.Boolean
        ).alias("chosen_for_card"))

    def show(sp, src, qset, split, band):
        r = allp.filter((pl.col("species") == sp) & (pl.col("source") == src)
                        & (pl.col("query_set") == qset) & (pl.col("split") == split)
                        & (pl.col("band") == band))
        return r.row(0, named=True) if r.height else None

    def fmt(r, which="strict"):
        if r is None:
            return "not run"
        n = r["n_judged"]
        if n == 0:
            return "n = 0"
        k = r[f"right_{which}"]
        lo, hi = r[f"{which}_lo"], r[f"{which}_hi"]
        grey = " (grey: n < 30)" if n < MIN_N else ""
        return f"{100 * k / n:.1f}% ({k}/{n}; 95% CI {100 * lo:.1f}-{100 * hi:.1f}%){grey}"

    for qset, words in (("dark_1", "Dark proteins (no sequence-tool hit at E <= 1)"),
                        ("placed_1", "Placed proteins (a sequence-tool hit at E <= 1)")):
        say()
        say(f"{words}: precision of the best hit, strict call (loose call), all proteins")
        for sp in SPECIES:
            for src in list(SEQ_TOOLS) + ([chosen[sp]] if sp in chosen else []):
                for band in BANDS:
                    r = show(sp, src, qset, "all", band)
                    if r is None or r["n_judged"] == 0:
                        continue
                    name = TOOL_WORDS.get(src, src)
                    say(f"  {sp:9s} {name:42s} {BAND_WORDS[band]:15s} strict {fmt(r)}; "
                        f"loose {fmt(r, 'loose')}")
    say()
    say("Leaderboard column: strict precision at E 1-10, dark proteins, test half")
    for sp in SPECIES:
        for src in list(SEQ_TOOLS) + ([chosen[sp]] if sp in chosen else []):
            r = show(sp, src, "dark_1", "test", "1_10")
            say(f"  {sp:9s} {TOOL_WORDS.get(src, src):42s} {fmt(r)}")

    # Predictions in README.md, checked
    say()
    say("Predictions (weak-hits/README.md)")
    p1 = allp.filter((pl.col("n_judged") > 0) & (pl.col("split") == "all"))
    worse = p1.filter(pl.col("right_strict") > pl.col("right_loose")).height
    eq = p1.filter(pl.col("right_strict") == pl.col("right_loose")).height
    say(f"  1. strict right more often than loose in {worse} of {p1.height} rows "
        f"(equal in {eq})")
    for sp in SPECIES:
        for t in SEQ_TOOLS:
            r = show(sp, t, "dark_1", "all", "1_10")
            say(f"  2. {sp:9s} {t:9s} dark E 1-10 strict {fmt(r)}: "
                + ("holds" if r and r["n_judged"] and r["right_strict"] / r["n_judged"] < 0.05
                   else "fails"))
        if sp in chosen:
            r = show(sp, chosen[sp], "dark_1", "test", "1_10")
            say(f"  3. {sp:9s} chosen kmerseek, test half {fmt(r)}: "
                + ("holds" if r and r["n_judged"] and r["right_strict"] / r["n_judged"] < 0.10
                   else ("no judged hits" if r is None or not r["n_judged"] else "fails")))
        r = show(sp, "mmseqs2", "placed_1", "all", "le_0.001")
        say(f"  4. {sp:9s} MMseqs2 placed E <= 0.001 strict {fmt(r)}: "
            + ("holds" if r and r["right_strict"] / r["n_judged"] > 0.90 else "fails"))

    # Every kmerseek setting, dark E 1-10, all proteins
    say()
    say("Every kmerseek setting, dark proteins, E 1-10, all proteins: strict precision")
    for sp in SPECIES:
        k = allp.filter((pl.col("species") == sp) & (pl.col("query_set") == "dark_1")
                        & (pl.col("split") == "all") & (pl.col("band") == "1_10")
                        & pl.col("source").str.starts_with("kmerseek:"))
        big = k.filter(pl.col("n_judged") >= MIN_N)
        say(f"  {sp:9s} {k.height} settings, {big.height} with >= {MIN_N} judged: strict "
            f"precision median {100 * big['precision_strict'].median():.1f}%, max "
            f"{100 * big['precision_strict'].max():.1f}%; right {int(k['right_strict'].sum())} "
            f"of {int(k['n_judged'].sum())} judged best hits summed over settings")

    # Spot check: perilipin-4
    say()
    hum = js["human"]["strict_right_dark_1"]
    say("Spot check Q96Q06 (perilipin-4): strict-right in any human source? "
        + str(any(c["query"] == "Q96Q06" for v in hum.values() for c in v)))

    # ---- 4. the site's table ----
    site_rows = []
    for sp in SPECIES:
        for src in list(SEQ_TOOLS) + ([chosen[sp]] if sp in chosen else []):
            tool = SEQ_TOOLS.get(src, "kmerseek")
            for qset in ("dark_1", "placed_1"):
                for split in ("all", "test"):
                    for band in BANDS:
                        r = show(sp, src, qset, split, band)
                        site_rows.append(dict(
                            species=sp, tool=tool, source=src, query_set=qset.split("_")[0],
                            split=split, band=band, n_proteins=r["n_proteins"],
                            n_findable=r["n_findable"], n_best_hit_in_band=r["n_best_hit_in_band"],
                            n_judged=r["n_judged"], right_strict=r["right_strict"],
                            right_loose=r["right_loose"],
                            target_without_pfam=r["target_without_pfam"],
                            strict_query_stretch_unknown=r["strict_query_stretch_unknown"]))
    site = pl.DataFrame(site_rows)
    outdir = REPO / "results" / "leaderboard" / "weak_hits"
    outdir.mkdir(parents=True, exist_ok=True)
    site.write_parquet(outdir / "weak_hits.parquet")
    say()
    say(f"site table: {site.height} rows, {outdir.relative_to(REPO)}/weak_hits.parquet")

    # ---- 5. figures ----
    import figures_weak_hits
    figs = figures_weak_hits.draw(allp, chosen, HERE / "figures")
    (HERE / "summary.txt").write_text("\n".join(out_lines) + "\n")
    outputs = {"weak_hits.parquet": sha256(outdir / "weak_hits.parquet")}
    prov = {
        "made_by": "nextflow-runs/invertebrate-dark-set/weak-hits/summarize.py",
        "rules": "nextflow-runs/invertebrate-dark-set/weak-hits/README.md",
        "inputs": inputs,
        "outputs": outputs,
        "other_outputs": {str(p.relative_to(REPO)): sha256(p) for p in
                          [HERE / "summary.txt", HERE / "weak_hits_all_sources.parquet"] + figs},
        "columns": {
            "query_set": "dark: no phmmer, jackhmmer or MMseqs2 hit with E <= 1; placed: the rest",
            "split": "all proteins, or the test half (md5 of the accession odd)",
            "band": "E-value band of the best hit: le_0.001, 0.001_1, 1_10",
            "n_judged": "best hits in the band whose target carries a Pfam family",
            "right_strict": "target shares a Pfam family AND the hit's query stretch overlaps "
                            "a query domain of that family",
            "right_loose": "target shares a Pfam family or clan (PR 133's call)",
        },
        "chosen_kmerseek_setting": {sp: chosen.get(sp) for sp in SPECIES},
    }
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
