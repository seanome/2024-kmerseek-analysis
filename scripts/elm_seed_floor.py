#!/usr/bin/env python3
"""Two seed lengths per alphabet for the ELM search, chosen by how many proteins a seed hits.

For every kmerseek alphabet, count how many human proteins contain a seed taken at a random
position of the human proteome, for each seed length k:

    proteins per seed = sum over seeds of n_seed^2 / sum over seeds of n_seed

where n_seed is the number of proteins that contain that seed. Squared because a seed found
in many proteins is also drawn more often. The count includes the protein the seed came
from, so it is at least 1.

Counts come from kmerseek itself, so the classes are exactly the ones the search uses:
`kmerseek index --stats-only --kmer-stats-out`, whose spectrum lists, per seed frequency, how
many distinct seeds occur in that many proteins (the header calls it seqs_per_kmer).

Chosen per alphabet:
  k_main   the shortest k with fewer than 100 proteins per seed
  k_small  the shortest k with fewer than 1_000: ten times noisier, short enough to fit
           closer around a short motif, but only if it fits in memory (below)

Memory. kmerseek search memory grows with this same number (the "spectrum load" in the
dark-set memory model). The limit is params.kmerseek_memory_max, 128 GB per search on the
first attempt. The midi-plus run searched the same nine QfO proteomes, so its trace says
which loads fit: the largest human-proteome load whose worst search peaked under 128 GB is
the cap (452, gbmr7 k10, max peak_rss 96 GB; gbmr7 k9 at 758 peaked at 191 GB). k_small is
raised until its load is at or under that cap. k=4 is never used for the 17-20 letter
alphabets: main.nf dropped protein20 k4 after it ran out of memory, and 20^4 = 160_000 keys
against ~11 million proteome k-mers fails the same way for 17 or 18 letters. When the
smallest k that fits equals k_main, the alphabet gets one k.

Inputs
  human proteome  QfO release 2020_04, Eukaryota/UP000005640_9606.fasta
  kmerseek        image kmerseek-region:0.4.0-rc5-arm64 (kmerseek 0.4.0), built from
                  nextflow-runs/qfo-pfam-region-benchmark/Dockerfile

Outputs, in --out-dir: human.<alphabet>.k<k>.csv (one spectrum per alphabet and k) and
scan.csv; and tables/250_two_k_per_alphabet.csv in the repo.

Usage:
  python scripts/elm_seed_floor.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QFO_HUMAN = Path(
    "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
    "/Eukaryota/UP000005640_9606.fasta"
)
IMAGE = "kmerseek-region:0.4.0-rc5-arm64"
MIDI_TRACE = Path(
    "/Users/olga/data/qfo-pfam-region-midi-plus/traces/midi-plus.trace.txt"
)
MEMORY_LIMIT_GB = 128
NO_K4 = {"protein20", "uniprot18", "hsdm17", "wass14"}
THRESHOLDS = {"k_main": 100, "k_small": 1_000}
#: (alphabet, pipeline's shortest k in qfo-pfam-region-benchmark main.nf ALL_ENCODINGS /
#: EXTRA_ENCODINGS). The scan starts 6 below it.
ALPHABETS = [
    ("protein20", 5),
    ("uniprot18", 5),
    ("hsdm17", 5),
    ("wass14", 5),
    ("mmseqs12", 5),
    ("sdm12", 6),
    ("dayhoff6", 8),
    ("wwmj5", 8),
    ("gbmr7", 9),
    ("gbmr4", 12),
    ("hp_lehninger_hpc3", 16),
    ("hp_lehninger2", 18),
    ("hp_lehninger_c_nonpolar2", 18),
    ("hp_pbotc_1st_ed2", 18),
    ("hp_thomas_dill2", 19),
    ("hp_thomas_dill_no_c2", 19),
    ("hp_kyte_doolittle2", 19),
    ("polarity4", 10),
    ("funcgroups8", 7),
]


def proteins_per_seed(out_dir: Path, human: Path, alphabet: str, k: int) -> float:
    spectrum = out_dir / f"human.{alphabet}.k{k}.csv"
    if not spectrum.exists():
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                "/Users/olga/data:/Users/olga/data",
                "--entrypoint",
                "kmerseek",
                IMAGE,
                "index",
                "--alphabet",
                alphabet,
                "--ksize",
                str(k),
                "--input",
                str(human),
                "--output",
                str(out_dir / f"tmp_{alphabet}_{k}"),
                "--stats-only",
                "--kmer-stats-out",
                str(spectrum),
            ],
            check=True,
            capture_output=True,
        )
    s1 = s2 = 0
    with open(spectrum) as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            o, n = int(r["occurrences"]), int(r["n_kmers"])
            s1 += o * n
            s2 += o * o * n
    return s2 / s1


def load_cap(scan: list[dict]) -> tuple[float, str]:
    """Largest load whose worst midi-plus kmerseekSearch peak stayed under MEMORY_LIMIT_GB."""
    import polars as pl

    t = pl.read_csv(
        MIDI_TRACE, separator="\t", infer_schema_length=0, truncate_ragged_lines=True
    )
    t = t.filter(pl.col("name").str.starts_with("kmerseekSearch")).with_columns(
        m=pl.col("name").str.extract_groups(r"\(([a-z]+)_(.+)_k(\d+)_lc(true|false)\)")
    )
    t = t.unnest("m").rename({"1": "species", "2": "alphabet", "3": "k", "4": "lc"})
    unit = {"GB": 1.0, "MB": 1 / 1024, "TB": 1024.0, "KB": 1 / 1024**2}
    t = t.with_columns(
        peak_gb=pl.col("peak_rss").map_elements(
            lambda v: (
                float(v.split()[0]) * unit[v.split()[1]] if v and " " in v else None
            ),
            return_dtype=pl.Float64,
        ),
        k=pl.col("k").cast(pl.Int64),
    )
    worst = t.group_by("alphabet", "k").agg(max_peak_gb=pl.col("peak_gb").max())
    loads = pl.DataFrame(scan).select("alphabet", "k", "proteins_per_seed")
    j = worst.join(loads, on=["alphabet", "k"], how="inner")
    ok = (
        j.filter(pl.col("max_peak_gb") < MEMORY_LIMIT_GB)
        .sort("proteins_per_seed", descending=True)
        .row(0, named=True)
    )
    return (
        ok["proteins_per_seed"],
        f"{ok['alphabet']} k{ok['k']}, max peak {ok['max_peak_gb']:.0f} GB",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/olga/data/elm-motif-transfer/seed_floor"),
    )
    ap.add_argument("--human", type=Path, default=QFO_HUMAN)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scan, chosen = [], []
    for alphabet, kmin in ALPHABETS:
        k = max(3, kmin - 6)
        found: dict[str, tuple[int, float]] = {}
        first = True
        while True:
            v = proteins_per_seed(args.out_dir, args.human, alphabet, k)
            scan.append(
                {
                    "alphabet": alphabet,
                    "pipeline_kmin": kmin,
                    "k": k,
                    "proteins_per_seed": round(v, 1),
                }
            )
            for name, t in THRESHOLDS.items():
                if name not in found and v < t:
                    if first:
                        raise SystemExit(
                            f"{alphabet}: already under {t} at the first k scanned ({k}); start lower"
                        )
                    found[name] = (k, v)
            first = False
            if len(found) == len(THRESHOLDS) or k > kmin + 12:
                break
            k += 1
        chosen.append(
            {
                "alphabet": alphabet,
                "pipeline_kmin": kmin,
                **{n: found[n][0] for n in THRESHOLDS},
                **{f"proteins_per_seed_{n}": round(found[n][1], 1) for n in THRESHOLDS},
            }
        )
        print(chosen[-1], flush=True)

    cap, cap_from = load_cap(scan)
    print(f"memory: load cap {cap} ({cap_from}), from {MIDI_TRACE}")
    by = {(r["alphabet"], r["k"]): r["proteins_per_seed"] for r in scan}
    for c in chosen:
        k = c["k_small"]
        while k < c["k_main"] and (
            (c["alphabet"] in NO_K4 and k <= 4) or by[(c["alphabet"], k)] > cap
        ):
            k += 1
        c["k_small_fits_memory"] = k if k < c["k_main"] else None
        c["proteins_per_seed_k_small_fits_memory"] = (
            by[(c["alphabet"], k)] if k < c["k_main"] else None
        )
        c["load_cap"] = cap

    with open(args.out_dir / "scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scan[0]))
        w.writeheader()
        w.writerows(scan)
    table = REPO / "tables" / "250_two_k_per_alphabet.csv"
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chosen[0]))
        w.writeheader()
        w.writerows(chosen)
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
