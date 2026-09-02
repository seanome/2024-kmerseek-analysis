#!/usr/bin/env python3
"""Cost per search, per tool, from Nextflow trace files.

The unit is one search of the human query set against ONE target proteome, because that is
the thing a user actually runs. It is not "per arm": kmerseek's 404 alphabet x k-size arms
are a parameter sweep and only one of them ships, so charging kmerseek for all 404 would
compare a sweep against a single configuration.

Two processes need care. folddiscoQuery is chunked, so its per-search cost is the sum over
that species' chunks, not one chunk. Database and index builds are separated out rather
than folded in: they are paid once per target and amortise over every later query, so
mixing them into a per-search number would flatter whichever tool happens to build last.
"""
import argparse, glob, re, sys
import polars as pl

SEARCH = {"kmerseekSearch": "kmerseek", "phmmerSearch": "hmmer3_phmmer",
          "jackhmmerSearch": "hmmer3_jackhmmer", "mmseqs2Search": "mmseqs2",
          "hhblitsSearch": "hhblits", "prostt5Search": "prostt5",
          "foldseekSearch": "foldseek", "reseekSearch": "reseek",
          "folddiscoQuery": "folddisco"}
BUILD = {"kmerseekIndex": "kmerseek", "mmseqsDb": "mmseqs2", "hhblitsBuildDB": "hhblits",
         "prostt5Db": "prostt5", "foldseekDb": "foldseek", "reseekConvert": "reseek",
         "folddiscoIndex": "folddisco", "mmseqsDomainDb": "mmseqs2"}

DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)")
MULT = {"ms": 1/1000, "s": 1, "m": 60, "h": 3600, "d": 86400}


def seconds(v):
    """Nextflow writes '2h 34m 1s', or a bare integer of milliseconds. Handle both."""
    if v is None:
        return None
    v = str(v).strip()
    if v in ("-", "", "null"):
        return None
    if re.fullmatch(r"\d+", v):
        return int(v) / 1000.0
    tot = sum(float(n) * MULT[u] for n, u in DUR.findall(v))
    return tot or None


def species_of(tag):
    m = re.search(r"human_vs_(\w+?)(?:\s|$|\.)", str(tag))
    if m:
        return m.group(1)
    m = re.match(r"([a-z]+)[_.]", str(tag))
    return m.group(1) if m else str(tag)


def load(paths):
    frames = []
    for p in paths:
        df = pl.read_csv(p, separator="\t", infer_schema_length=0)
        df = df.with_columns(trace=pl.lit(p))
        frames.append(df)
    t = pl.concat(frames, how="diagonal_relaxed")
    t = t.with_columns(
        proc=pl.col("process").str.split(":").list.last(),
        rt=pl.col("realtime").map_elements(seconds, return_dtype=pl.Float64),
        dur=pl.col("duration").map_elements(seconds, return_dtype=pl.Float64),
        ncpu=pl.col("cpus").cast(pl.Float64, strict=False),
    )
    return t.filter(pl.col("status") == "COMPLETED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="efficiency_per_search.tsv")
    a = ap.parse_args()

    paths = [p for g in a.traces for p in glob.glob(g)]
    t = load(paths)
    print(f"traces: {len(paths)}   COMPLETED tasks: {t.height:_}")
    print(t.group_by("proc").len().sort("len", descending=True).head(20))

    s = (t.filter(pl.col("proc").is_in(list(SEARCH)))
           .with_columns(tool=pl.col("proc").replace_strict(SEARCH),
                         species=pl.col("tag").map_elements(species_of, return_dtype=pl.String),
                         cpu_h=(pl.col("rt") * pl.col("ncpu") / 3600.0)))
    if s.height == 0:
        print("no search tasks found", file=sys.stderr); return

    # One search = one (tool, species, arm). folddisco chunks collapse into their species.
    s = s.with_columns(arm=pl.when(pl.col("tool") == "kmerseek")
                             .then(pl.col("tag")).otherwise(pl.col("species")))
    per = (s.group_by("tool", "species", "arm")
             .agg(wall_s=pl.col("rt").sum(), cpu_h=pl.col("cpu_h").sum(),
                  n_tasks=pl.len(), cpus=pl.col("ncpu").max()))

    summ = (per.group_by("tool").agg(
                n_searches=pl.len(),
                median_wall_min=(pl.col("wall_s").median() / 60),
                p25_wall_min=(pl.col("wall_s").quantile(0.25) / 60),
                p75_wall_min=(pl.col("wall_s").quantile(0.75) / 60),
                median_cpu_h=pl.col("cpu_h").median(),
                total_cpu_h=pl.col("cpu_h").sum(),
                cpus=pl.col("cpus").max())
            .sort("median_cpu_h"))
    print("\n=== cost of ONE search (human query set vs one target proteome) ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(summ.with_columns(pl.col("^.*(min|cpu_h)$").round(3)))

    b = (t.filter(pl.col("proc").is_in(list(BUILD)))
           .with_columns(tool=pl.col("proc").replace_strict(BUILD),
                         cpu_h=(pl.col("rt") * pl.col("ncpu") / 3600.0))
           .group_by("tool").agg(n_builds=pl.len(),
                                 median_build_min=(pl.col("rt").median() / 60),
                                 total_build_cpu_h=pl.col("cpu_h").sum()))
    print("\n=== one-off database / index builds, amortised over every later query ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(b.sort("total_build_cpu_h").with_columns(pl.col("^.*(min|cpu_h)$").round(3)))

    per.write_csv(a.out, separator="\t")
    print(f"\nwrote {a.out}: {per.height:_} searches")


if __name__ == "__main__":
    main()
