"""Cost of protein-sketching three sequence collections under four translation choices.

Reads the measured rates in measurements/ (from bench.sbatch) plus the measured rates and sizes
written in RATES below, and writes measurements/estimate.csv and a figure.
Usage: python estimate.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "measurements"

# Sizes of the three collections, in gigabases (Gbp).
COLLECTIONS = {
    # Public SRA, December 2024: 66.9 Pbp (arXiv 2505.06395). Logan's December 2023 freeze
    # was 50.3 Pbp, so the archive grew 33% in a year; at that rate it is ~113 Pbp in
    # October 2026. Both are shown; the table uses the measured 2024 number.
    "1 SRA (public, Dec 2024)": 66.9e6,
    # Same reads, run through SPLASH and DE-kupl per BioProject; only their output is sketched.
    "2 SRA after SPLASH + DE-kupl": 66.9e6,
    # Logan contigs: 0.90 Pbp, 315 TB zstd (Logan paper, PMC12424806). Unitigs are 4.59 Pbp.
    "3 Logan contigs": 0.90e6,
}
SRA_2026_FACTOR = 113 / 66.9

# Oyster mantle run (oyster-mvp-20261023, 6 libraries, 145 M read pairs), measured with
# /usr/bin/time on Sherlock hns nodes. CPU-hours per Gbp of trimmed reads.
SPLASH_CPU_H = (7109.81 + 545.61) / 3600  # 16 min wall on 8 cores, 24.9 GB peak
DEKUPL_CPU_H = (125118.70 + 3304.88) / 3600  # 3 h 05 min wall on 32 cores, 21.8 GB peak


def load():
    frames = json.load(open(R / "frames_summary.json"))
    lens = json.load(open(R / "splash_dekupl_lengths.json"))
    t = {}
    for row in csv.reader(open(R / "timings.tsv"), delimiter="\t"):
        t[row[0]] = {"cpu_s": float(row[1]), "wall": row[2], "rss_gb": float(row[3])}
    sizes = {}
    for row in csv.reader(open(R / "sizes.tsv"), delimiter="\t"):
        sizes[Path(row[1]).name] = int(row[0])
    return frames, lens, t, sizes


def main():
    frames, lens, t, sizes = load()
    gbp_bench = frames["bases"] / 1e9
    dw1_bytes = int(open(R / "DW1_1.bytes").read()) if (R / "DW1_1.bytes").exists() else None
    oyster_gbp = lens["oyster_full_trimmed_bases_upper"] / 1e9
    kept = (lens["dekupl_contigs"]["bases"] + lens["splash_compactors"]["bases"]) / 1e9
    kept_frac = kept / oyster_gbp

    # CPU-hours per Gbp of DNA going in, per choice.
    seqkit = t["seqkit_translate6"]["cpu_s"] / 3600 / gbp_bench
    per_gbp = {}
    for x in "abcd":
        sketch = t[f"manysketch_{x}_s100"]["cpu_s"] / 3600 / gbp_bench
        extra = 0.0
        if x == "d":
            # Looking up every 10-mer of every stop-free stretch against Swiss-Prot costs about
            # what hashing them all costs: proxy is manysketch on all of (b)'s stretches... (a).
            extra = t["manysketch_a_s100"]["cpu_s"] / 3600 / gbp_bench
        per_gbp[x] = seqkit + extra + sketch
    sourmash_py = t["sourmash_translate_s100"]["cpu_s"] / 3600 / gbp_bench

    rows = []
    for name, gbp in COLLECTIONS.items():
        for x in "abcd":
            if name.startswith("2"):
                pre = gbp * (SPLASH_CPU_H + DEKUPL_CPU_H) / oyster_gbp
                sk = gbp * kept_frac * per_gbp[x]
                res_frac = kept_frac
            else:
                pre = 0.0
                sk = gbp * per_gbp[x]
                res_frac = 1.0
            sig = sizes[f"{x}_s1000.sig.zip"] / gbp_bench  # bytes per Gbp at scaled 1000
            rows.append({
                "collection": name, "choice": x,
                "preprocess_cpu_h": round(pre),
                "translate_sketch_cpu_h": round(sk),
                "total_cpu_h": round(pre + sk),
                "residues_per_input_base": frames["residues_per_base"][x] * res_frac,
                "sketch_bytes_s1000_upper": sig * gbp * res_frac,
            })
    with open(R / "estimate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"seqkit_cpu_h_per_gbp": seqkit, "per_gbp": per_gbp,
               "sourmash_python_translate_cpu_h_per_gbp": sourmash_py,
               "splash_dekupl_kept_fraction": kept_frac, "oyster_gbp": oyster_gbp,
               "dw1_fastq_bytes": dw1_bytes, "sra_2026_factor": SRA_2026_FACTOR},
              open(R / "rates.json", "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
