#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Print every (dash_encoding, display_encoding, ksize) combo that has a real (non-empty)
genome-wide results file on disk, as CSV on stdout, for Nextflow to split into per-combo
computeMetricLeaderboard tasks.

Derived directly from the filesystem -- human_vs_mouse.{encoding}.k{ksize}.results.{parquet,
csv.zst,csv.gz} under --data-dir and ortholog_analysis_utils.EXTRA_DATA_DIRS -- NOT from any
notebook-written CSV. Nextflow pipeline steps must never depend on a notebook's output; only the
reverse (notebooks read pipeline output). An earlier version of this script read notebook 200's
own 200_alphabet_ksize_matched_scope_comparison.csv / 200_hp_variants_full_sweep.csv, which meant
that notebook had to run to completion before this pipeline step could work at all -- backwards.
Reuses EXTRA_DATA_DIRS and the empty-stub size check from ortholog_analysis_utils.py so this
can't silently drift from what genome_wide_results_file/scan_genome_wide_results would actually
find at read time.

Usage:
    list_metric_leaderboard_combos.py --data-dir <dir> --output combos.csv
"""

import argparse
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "/Users/olga/code/2024-kmerseek-analysis/notebooks")
import ortholog_analysis_utils as ou  # noqa: E402

FILENAME_RE = re.compile(
    r"^human_vs_mouse\.(?P<encoding>[a-z0-9_-]+)\.k(?P<ksize>\d+)\.results\.(?P<ext>parquet|csv\.zst|csv\.gz)$"
)
EXT_PREFERENCE = {"parquet": 0, "csv.zst": 1, "csv.gz": 2}  # matches genome_wide_results_file


def find_combos(data_dir: Path) -> list[tuple[str, int]]:
    """One (dash_encoding, ksize) per file found, format-deduped by EXT_PREFERENCE.

    A handful of legacy HP-variant combos (e.g. hp_thomas_dill k25/26) have both kmerseek's
    real dash-named CLI output AND a stray older underscore-named duplicate left over from an
    early manual copy (both real data, different mtimes). Left alone, both would collide under
    the same display_encoding downstream (dash.replace('-','_') == the underscore name already),
    double-scoring one combo and silently confusing any groupby on (encoding, ksize). Deduped
    here by preferring the dash-named file whenever one exists for that (ksize, display name) --
    kmerseek's actual --encoding CLI convention -- over a bare-underscore alternate.
    """
    search_dirs = [data_dir] + [d for d in ou.EXTRA_DATA_DIRS if d != data_dir]
    found: list[tuple[str, int, Path, int]] = []  # (dash_encoding, ksize, path, ext_rank)
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = FILENAME_RE.match(f.name)
            if not m:
                continue
            if ou._is_empty_results_file(f):
                continue
            found.append((m.group("encoding"), int(m.group("ksize")), f, EXT_PREFERENCE[m.group("ext")]))

    # Within one dash_encoding, keep only the best-format file (parquet > csv.zst > csv.gz).
    best_by_dash: dict[tuple[str, int], tuple[Path, int]] = {}
    for enc, k, f, rank in found:
        key = (enc, k)
        if key not in best_by_dash or rank < best_by_dash[key][1]:
            best_by_dash[key] = (f, rank)

    # Across dash_encodings that normalize to the same display name, prefer the one with a
    # literal dash (kmerseek's real CLI convention) over a bare-underscore duplicate.
    best_by_display: dict[tuple[str, int], str] = {}
    for enc, k in best_by_dash:
        display_key = (enc.replace("-", "_"), k)
        incumbent = best_by_display.get(display_key)
        if incumbent is None or ("-" in enc and "-" not in incumbent):
            best_by_display[display_key] = enc

    return sorted((enc, k) for (_, k), enc in best_by_display.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = [
        {"dash_encoding": enc, "display_encoding": enc.replace("-", "_"), "ksize": k}
        for enc, k in find_combos(args.data_dir)
    ]
    combos = pl.DataFrame(
        rows, schema={"dash_encoding": pl.Utf8, "display_encoding": pl.Utf8, "ksize": pl.Int64}
    )
    combos.write_csv(args.output)
    print(f"{combos.height} combos written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
