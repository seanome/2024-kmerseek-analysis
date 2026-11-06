"""Write the sample sheets and comparisons for the young-colony PEC run.

Domen et al. 2026 (PNAS) treated Botryllus colonies with pulsatile electrical current (PEC)
and sequenced treated and control colonies at times from 2 hours to 132 days after. This
script takes the 52 libraries from young colonies (370-510 days old; the old colony
944aXBYd196-4 is 7,600+ days old) and puts each one in one of six time windows. Every
library is kept, however few reads it has.

Inputs (all in inputs/):
  ena_sample_metadata.tsv        ENA run table with the BioSample attributes, 64 runs
  published_vial_hours.tsv       hours after treatment for each vial, from the paper's
                                 Dataset S1 sample table
  trimmed-inputs.validated.json  Trim Galore output for the 64 runs on Sherlock, with read
                                 pair counts (from PR 203)

Outputs:
  samplesheet.csv   nf-core/rnaseq input: trimmed read pairs, one row per library
  samples.csv       nf-core/differentialabundance sample table
  contrasts.csv     one treated-vs-control comparison per window, colony as blocking factor
  splash/<window>/input.txt, cjs.csv
                    SPLASH input (one read file per library) and sample weights
                    (control -1, treated +1)

Usage: python make_design.py --strandedness <unstranded|forward|reverse>
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Hours after treatment -> window. Chosen 2026-10-27 so that every window has both control
# and treated libraries; 9 of the 16 exact collection times have only one of the two.
WINDOWS = [
    ("young_2h", 2, 2),
    ("young_24h", 24, 24),
    ("young_13_17d", 13 * 24, 17 * 24),
    ("young_21_28d", 21 * 24, 28 * 24),
    ("young_49_52d", 49 * 24, 52 * 24),
    ("young_104_132d", 104 * 24, 132 * 24),
]
YOUNG_MAX_AGE_DAYS = 1000

# SPLASH takes one read file per library. splash_prep.sbatch writes read 1 and read 2 of the
# same Trim Galore pairs that nf-core/rnaseq uses into one file per library here. (The
# fastp files from notebook 361, PR 201, lack the deepest library, SRR32233964.)
SPLASH_READS = "/scratch/users/olgabot/botryllus-pec-young/splash-reads/{sample}.fq.gz"


def window_of(hours):
    for name, lo, hi in WINDOWS:
        if lo <= hours <= hi:
            return name
    raise ValueError(f"{hours} h is in no window")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--strandedness", required=True, choices=["unstranded", "forward", "reverse"]
    )
    args = ap.parse_args()

    ena = {
        r["run_accession"]: r
        for r in csv.DictReader(
            open(HERE / "inputs/ena_sample_metadata.tsv"), delimiter="\t"
        )
    }
    hours = {
        r["vial"]: int(r["hours"])
        for r in csv.DictReader(
            open(HERE / "inputs/published_vial_hours.tsv"), delimiter="\t"
        )
    }
    trimmed = {
        x["run"]: x
        for x in json.load(open(HERE / "inputs/trimmed-inputs.validated.json"))
    }
    assert set(ena) == set(trimmed), "ENA runs and trimmed runs differ"

    rows = []
    for run in sorted(ena):
        m = ena[run]
        age = int(m["Age In Days"].rstrip("d"))
        if age >= YOUNG_MAX_AGE_DAYS:
            continue
        vial = m["Vial Number"]
        treatment = {"PaceMakerCon": "control", "PaceMakerTre": "treated"}[
            m["Experimental Group"]
        ]
        window = window_of(hours[vial])
        r1, r2 = (f["path"] for f in trimmed[run]["files"])
        rows.append(
            dict(
                sample=run,
                vial=vial,
                treatment=treatment,
                window=window,
                group=f"{window}_{treatment}",
                colony=m["Colony Id"],
                hours=hours[vial],
                age_days=age,
                trimmed_pairs=trimmed[run]["pairs"],
                fastq_1=r1,
                fastq_2=r2,
            )
        )
    rows.sort(key=lambda r: (r["hours"], r["treatment"], r["sample"]))

    n = Counter((r["window"], r["treatment"]) for r in rows)
    for name, _, _ in WINDOWS:
        assert (
            n[(name, "control")] and n[(name, "treated")]
        ), f"{name} lacks one side: {n}"
    assert len(rows) == 52, len(rows)

    with open(HERE / "samplesheet.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sample", "fastq_1", "fastq_2", "strandedness"])
        for r in rows:
            w.writerow([r["sample"], r["fastq_1"], r["fastq_2"], args.strandedness])

    cols = [
        "sample",
        "vial",
        "treatment",
        "window",
        "group",
        "colony",
        "hours",
        "age_days",
        "trimmed_pairs",
    ]
    with open(HERE / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    with open(HERE / "contrasts.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "variable", "reference", "target", "blocking"])
        for name, _, _ in WINDOWS:
            w.writerow(
                [
                    f"{name}_treated_vs_control",
                    "group",
                    f"{name}_control",
                    f"{name}_treated",
                    "colony",
                ]
            )

    for name, _, _ in WINDOWS:
        d = HERE / "splash" / name
        d.mkdir(parents=True, exist_ok=True)
        mine = [r for r in rows if r["window"] == name]
        (d / "input.txt").write_text(
            "".join(
                f"{r['sample']} {SPLASH_READS.format(sample=r['sample'])}\n"
                for r in mine
            )
        )
        (d / "cjs.csv").write_text(
            "".join(
                f"{r['sample']},{1 if r['treatment'] == 'treated' else -1}\n"
                for r in mine
            )
        )

    for name, _, _ in WINDOWS:
        print(
            f"{name:16s} control {n[(name, 'control')]}  treated {n[(name, 'treated')]}"
        )


if __name__ == "__main__":
    main()
