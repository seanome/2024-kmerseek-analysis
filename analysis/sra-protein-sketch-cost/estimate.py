"""Cost of protein-sketching three sequence collections under four translation choices.

Reads the rates measured by bench.sbatch and bench_full.sbatch (measurements/), combines them
with the collection sizes and SPLASH / DE-kupl rates below, and writes
measurements/estimate.csv, measurements/rates.json and
figures/sra_protein_sketch_cost_sherlock_oyster.*.
Run from the repo root: python analysis/sra-protein-sketch-cost/estimate.py
"""

import csv
import json
import sys
from pathlib import Path

from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
M = HERE / "measurements"
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))

CHOICES = {
    "a": "all 6 frames",
    "b": "stretches of 30+ aa",
    "c": "longest stretch per read or contig",
    "d": "frame with most Swiss-Prot 10-mers",
}

# Collection sizes in gigabases (Gbp) and bytes as stored.
# Public SRA, Dec 2024: 66.9 Pbp, ~27 PB (arXiv 2505.06395). Logan's Dec 2023 freeze was
# 50.3 Pbp, so the archive grew 33% that year; at that rate it is ~113 Pbp in Oct 2026.
SRA_GBP, SRA_BYTES = 66.9e6, 27e15
SRA_2026_FACTOR = 113 / 66.9
# Logan (PMC12424806): contigs 0.90 Pbp, 315 TB zstd; unitigs 4.59 Pbp, 2.18 PB zstd.
LOGAN = {"3 Logan contigs": (0.90e6, 315e12), "3' Logan unitigs": (4.59e6, 2.18e15)}

# Logan contig files measured by bench_logan.sbatch. The two ~90 Mbp files give the CPU and
# storage rates (the 4.5 Mbp E. coli file is mostly start-up time); all three give the
# amino-acid fractions, since Logan holds RNA-seq, metagenomes and genomes.
LOGAN_RUNS = {"SRR9089190": "oyster RNA-seq", "SRR5650070": "human gut metagenome",
              "SRR2584863": "E. coli genome"}
LOGAN_RATE_RUNS = ("SRR9089190", "SRR5650070")

# Oyster mantle run, oyster-mvp-20261023 on Sherlock hns (AMD EPYC), 6 libraries,
# 307 M trimmed reads; /usr/bin/time user + system seconds of the whole run.
SPLASH_CPU_S, SPLASH_RSS_GB = 7109.81 + 545.61, 24.9  # 16 min wall on 8 cores
DEKUPL_CPU_S, DEKUPL_RSS_GB = 125118.70 + 3304.88, 21.8  # 3 h 05 min wall on 32 cores


def read_tsv(path):
    return {r[0]: r[1:] for r in csv.reader(open(path), delimiter="\t")}


def main():
    frames = json.load(open(M / "frames_summary.json"))
    lens = json.load(open(M / "splash_dekupl_lengths.json"))
    t = {k: float(v[0]) for k, v in read_tsv(M / "timings.tsv").items()}
    tf = {k: float(v[0]) for k, v in read_tsv(M / "timings_full.tsv").items()}
    sizes = {Path(v[0]).name: int(k) for k, v in read_tsv(M / "sizes.tsv").items()}
    depth = read_tsv(M / "depth_sizes.tsv")

    bench_gbp = frames["bases"] / 1e9
    dw1_gbp = 24137563 * 150 / 1e9  # DW1 read 1, untrimmed (validated.json pairs_in)
    oyster_gbp = lens["oyster_full_trimmed_bases_upper"] / 1e9

    # CPU-seconds per Gbp of DNA, one core.
    read_s = t["zcat_full_DW1_1"] / dw1_gbp
    translate_s = t["seqkit_translate6"] / bench_gbp
    sketch_s = {x: t[f"manysketch_{x}_s100"] / bench_gbp for x in CHOICES}
    # (d): looking up each 10-mer of the six frames in a Swiss-Prot table costs about what
    # hashing each one does, so the lookup is charged as one more sketch of (a).
    per_gbp = {
        x: read_s + translate_s + sketch_s[x] + (sketch_s["a"] if x == "d" else 0)
        for x in CHOICES
    }
    python_a = t["sourmash_translate_s100"] / bench_gbp
    python_a_full = tf["full_translate_s100"] / dw1_gbp

    # Sketch bytes per Gbp at full library depth: the 1 M-read sketch of each choice,
    # scaled by how much smaller per Gbp the whole-library six-frame sketch is.
    def per_gbp_bytes(s):
        sub = int(depth[f"a_translate_s{s}"][1]) / bench_gbp
        full = int(depth[f"full_a_s{s}"][1]) / dw1_gbp
        idx = int(depth[f"full_a_s{s}"][2]) / int(depth[f"full_a_s{s}"][1])
        return {x: sizes[f"{x}_s{s}.sig.zip"] / bench_gbp * full / sub for x in CHOICES}, idx

    sig100, idx100 = per_gbp_bytes(100)
    sig1000, idx1000 = per_gbp_bytes(1000)
    # Logan contigs and unitigs are already non-redundant within an accession, so depth
    # does not shrink them: count one hash per `scaled` amino acids (an upper bound), at
    # the bytes per hash of the measured whole-library .sig.zip.
    bytes_per_hash = {s: int(depth[f"full_a_s{s}"][1]) / int(depth[f"full_a_s{s}"][0])
                      for s in (100, 1000)}
    # `sourmash scripts index` CPU-seconds per byte of .sig.zip (whole library, scaled 100).
    index_s_per_byte = tf["index_full_a_s100"] / int(depth["full_a_s100"][1])

    # Logan contigs, measured directly.
    L = M / "logan"
    lt = {k: float(v[0]) for k, v in read_tsv(L / "logan_timings.tsv").items()}
    lsz = {(r[0], r[1]): (int(r[2]), int(r[3]))
           for r in csv.reader(open(L / "logan_sizes.tsv"), delimiter="\t")}
    lf = {a: json.load(open(L / f"{a}.frames_summary.json")) for a in LOGAN_RUNS}
    logan_frac = {x: sum(lf[a]["residues_per_base"][x] for a in LOGAN_RUNS) / len(LOGAN_RUNS)
                  for x in CHOICES}

    def mean_rate(key):
        return sum(lt[key.format(a=a)] / (lf[a]["bases"] / 1e9) for a in LOGAN_RATE_RUNS) / 2

    l_translate = mean_rate("{a}_translate6")
    l_sketch = {x: mean_rate(f"{{a}}_manysketch_{x}_s1000") for x in CHOICES}
    # Decompressing .zst was not timed; charged at the FASTQ.GZ rate, which is slower.
    logan_per_gbp = {x: read_s + l_translate + l_sketch[x] + (l_sketch["a"] if x == "d" else 0)
                     for x in CHOICES}
    logan_hashes_per_gbp = {x: sum(lsz[(a, x)][0] / (lf[a]["bases"] / 1e9)
                                   for a in LOGAN_RATE_RUNS) / 2 for x in CHOICES}
    logan_bytes_per_gbp = {x: sum(lsz[(a, x)][1] / (lf[a]["bases"] / 1e9)
                                  for a in LOGAN_RATE_RUNS) / 2 for x in CHOICES}

    splash_s, dekupl_s = SPLASH_CPU_S / oyster_gbp, DEKUPL_CPU_S / oyster_gbp
    kept = {
        x: (lens["dekupl_contigs"]["bases"] + lens["splash_compactors"]["bases"])
        if x != "b"
        else (lens["dekupl_contigs"]["bases_ge_90bp"] + lens["splash_compactors"]["bases_ge_90bp"])
        for x in CHOICES
    }
    kept_frac = {x: kept[x] / 1e9 / oyster_gbp for x in CHOICES}

    collections = {"1 SRA (public, Dec 2024)": (SRA_GBP, SRA_BYTES),
                   "2 SRA after SPLASH + DE-kupl": (SRA_GBP, SRA_BYTES), **LOGAN}
    rows = []
    for name, (gbp, in_bytes) in collections.items():
        for x in CHOICES:
            frac = kept_frac[x] if name.startswith("2") else 1.0
            pre = name.startswith("2")
            splash_h = gbp * splash_s / 3600 if pre else 0.0
            dekupl_h = gbp * dekupl_s / 3600 if pre else 0.0
            pre_h = splash_h + dekupl_h
            logan = "Logan" in name
            sk_h = gbp * frac * (logan_per_gbp if logan else per_gbp)[x] / 3600
            aa = gbp * 1e9 * frac * (logan_frac if logan else frames["residues_per_base"])[x]
            if logan:
                # Contig k-mers barely repeat within a sample, so ten times the hashes at
                # scaled 100, at the bytes per hash of the whole-library read sketch.
                sig = gbp * logan_bytes_per_gbp[x]
                sig_100 = gbp * logan_hashes_per_gbp[x] * 10 * bytes_per_hash[100]
            else:
                sig = gbp * frac * sig1000[x]
                sig_100 = gbp * frac * sig100[x]
            rows.append({
                "collection": name, "choice": x, "choice_label": CHOICES[x],
                "input_gbp": gbp, "input_bytes": in_bytes,
                "splash_cpu_h": round(splash_h), "dekupl_cpu_h": round(dekupl_h),
                "preprocess_cpu_h": round(pre_h), "translate_sketch_cpu_h": round(sk_h),
                "index_build_cpu_h_scaled100": round(sig_100 * index_s_per_byte / 3600),
                "total_cpu_h": round(pre_h + sk_h),
                "wall_days_1000_cores": round((pre_h + sk_h) / 1000 / 24, 2),
                "wall_days_10000_cores": round((pre_h + sk_h) / 10000 / 24, 2),
                "amino_acids_out": aa,
                "sketch_bytes_scaled1000": sig,
                "sketch_bytes_scaled100": sig_100,
                "rocksdb_bytes_scaled1000": sig * idx1000,
                "rocksdb_bytes_scaled100": sig_100 * idx100,
            })
    with open(M / "estimate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rates = {
        "cpu_s_per_gbp": {"read_fastq_gz": read_s, "translate_6_frames_seqkit": translate_s,
                          **{f"manysketch_{x}": v for x, v in sketch_s.items()},
                          **{f"total_{x}": v for x, v in per_gbp.items()},
                          "sourmash_python_translate_a_1M_reads": python_a,
                          "sourmash_python_translate_a_full_library": python_a_full,
                          "splash": splash_s, "dekupl": dekupl_s,
                          "rocksdb_index_per_sig_zip_byte": index_s_per_byte},
        "sig_zip_bytes_per_hash": bytes_per_hash,
        "logan_contigs": {"cpu_s_per_gbp": logan_per_gbp, "translate_6_frames": l_translate,
                          "manysketch_s1000": l_sketch,
                          "amino_acids_per_base_mean": logan_frac,
                          "amino_acids_per_base": {a: lf[a]["residues_per_base"] for a in LOGAN_RUNS},
                          "contigs_kept_d": {a: lf[a]["reads_kept_d"] for a in LOGAN_RUNS},
                          "sig_zip_bytes_per_gbp_s1000": logan_bytes_per_gbp},
        "splash_dekupl_kept_fraction_of_bases": kept_frac,
        "sketch_bytes_per_gbp_scaled1000": sig1000, "sketch_bytes_per_gbp_scaled100": sig100,
        "rocksdb_over_sig_zip": {"100": idx100, "1000": idx1000},
        "sra_2026_factor": SRA_2026_FACTOR, "oyster_gbp": oyster_gbp,
        "peak_rss_gb": {"splash": SPLASH_RSS_GB, "dekupl": DEKUPL_RSS_GB},
    }
    json.dump(rates, open(M / "rates.json", "w"), indent=1)
    figure(rows, frames, lf)
    for r in rows:
        print({k: r[k] for k in ("collection", "choice", "total_cpu_h", "wall_days_1000_cores",
                                 "sketch_bytes_scaled1000", "rocksdb_bytes_scaled1000")})


def figure(rows, frames, lf):
    import pubfig as pf

    pf.use_style()
    colours = list(pf.OKABE_ITO.values())
    fig, (ax1, ax2) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)
    names = list(dict.fromkeys(r["collection"] for r in rows))
    labels = ["SRA", "SRA after\nSPLASH + DE-kupl", "Logan\ncontigs", "Logan\nunitigs"]
    w = 0.2
    for i, x in enumerate(CHOICES):
        sub = [r for r in rows if r["choice"] == x]
        xs = [j + (i - 1.5) * w for j in range(len(names))]
        ax1.bar(xs, [r["preprocess_cpu_h"] for r in sub], w, color=pf.GREY,
                label="SPLASH + DE-kupl" if i == 0 else None)
        ax1.bar(xs, [r["translate_sketch_cpu_h"] for r in sub], w,
                bottom=[r["preprocess_cpu_h"] for r in sub], color=colours[i],
                label=f"({x}) {CHOICES[x]}")
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(names)), labels)
    ax1.set_ylabel("CPU-hours (one core each)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    sources = [("Oyster\nRNA-seq\n1 M reads", frames),
               ("Oyster\nRNA-seq\nLogan contigs", lf["SRR9089190"]),
               ("Human gut\nmetagenome\nLogan contigs", lf["SRR5650070"]),
               ("E. coli\ngenome\nLogan contigs", lf["SRR2584863"])]
    for i, x in enumerate(CHOICES):
        xs = [j + (i - 1.5) * w for j in range(len(sources))]
        ax2.bar(xs, [s["residues_per_base"][x] for _, s in sources], w, color=colours[i])
    ax2.set_xticks(range(len(sources)), [lab for lab, _ in sources])
    ax2.set_ylabel("Amino acids kept per DNA base")
    pf.shared_legend(fig, ncol=3)
    for ax, letter in zip((ax1, ax2), "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, str(HERE.parents[1] / "figures" / "sra_protein_sketch_cost_sherlock_oyster"))


if __name__ == "__main__":
    main()
