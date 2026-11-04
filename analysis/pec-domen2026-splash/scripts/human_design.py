#!/usr/bin/env python3
"""Build the human contrasts of notebook 361 from the GEO series matrices (added 2026-11-03).

Usage: human_design.py <geo_dir> <paper_accession_runs.tsv> <out_dir>

geo_dir holds the GEO series matrices (<GSE>[-<GPL>]_series_matrix.txt.gz), downloaded on
2026-11-03 from https://ftp.ncbi.nlm.nih.gov/geo/series/<GSEnnn>/<GSE>/matrix/.
paper_accession_runs.tsv is ENA's run table for the paper's accessions (exports/sra_search/).
GEO samples are joined to ENA runs by BioSample accession (SAMN...), one run per sample.

Writes, in out_dir:
  samples.tsv                  one row per run: series, run, GSM, GEO title, subject, the GEO
                               fields each contrast uses, read count and mean read length
  contrasts.tsv                one row per contrast: series, name, what it compares, sample and
                               pair counts, paired or not, the DESeq2 design used
  contrast_<name>.design.tsv   sample, condition, pair, genotype (the columns the Botryllus
                               scripts read), plus subject and group
  contrast_<name>.cjs          SPLASH sample sheet: reference group -1, other group +1

The shared scripts call the two groups "control" and "PEC". Here "control" is the reference
group of each study (before exercise, the rested leg, fit, young) and "PEC" the other group
(after, the exercised leg, frail, old); contrasts.tsv names them.
"""
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_matrix(path):
    """GEO series matrix -> list of dicts, one per sample (characteristics as ch_<key>)."""
    cols = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue
            row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            key, vals = row[0][len("!Sample_"):], row[1:]
            if key == "characteristics_ch1":
                key = "ch_" + vals[0].split(":", 1)[0].strip().replace(" ", "_")
                vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
            elif key == "relation":
                key = "rel_" + vals[0].split(":", 1)[0]
            while key in cols:
                key += "_2"
            cols[key] = vals
    n = len(cols["title"])
    out = []
    for i in range(n):
        d = {k: v[i] for k, v in cols.items()}
        d["biosample"] = d["rel_BioSample"].rsplit("/", 1)[-1]
        out.append(d)
    return out


def main():
    geo_dir, runs_tsv, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(list)
    for r in csv.DictReader(open(runs_tsv), delimiter="\t"):
        runs[(r["paper_accession"], r["sample_accession"])].append(r)

    samples = []
    for series in ("GSE99963", "GSE144304", "GSE120862"):
        geo = []
        for p in sorted(geo_dir.glob(f"{series}*_series_matrix.txt.gz")):
            geo += read_matrix(p)
        for g in geo:
            rr = runs[(series, g["biosample"])]
            if len(rr) != 1:
                raise SystemExit(f"{series} {g['geo_accession']} {g['biosample']}: {len(rr)} ENA runs")
            r = rr[0]
            s = dict(series=series, run=r["run_accession"], gsm=g["geo_accession"],
                     biosample=g["biosample"], title=g["title"],
                     reads=int(r["read_count"]),
                     mean_read_nt=round(int(r["base_count"]) / int(r["read_count"]), 1),
                     fastq_ftp=r["fastq_ftp"], fastq_md5=r["fastq_md5"])
            if series == "GSE99963":
                s["subject"] = g["ch_subject_id"]
                s["arm"] = g["ch_group"].split(";")[0].strip()  # Sedentary | Exercise
                s["time"] = g["ch_time_point"]  # Pre | Post
                s["sex"] = ""
            elif series == "GSE144304":
                s["subject"] = g["title"].split("_")[0]
                s["sex"] = g["ch_gender"]
                s["group"] = g["ch_group"]  # Young | Fit | Frail (GEO "group")
            else:
                s["subject"] = "P" + g["ch_individual"]
                s["training"] = g["ch_training_status"]  # untrained | trained
                s["time"] = {"baseline": "baseline", "+1 h after exercise": "1h",
                             "+4 h after exercise": "4h"}[g["ch_time_point"]]
                s["leg"] = "Ex" if g["ch_exercised/non-exercised_muscle"].startswith("exercised") else "NE"
                s["sex"] = g["ch_Sex"]
            samples.append(s)
        n_runs = sum(1 for (a, _), v in runs.items() if a == series for _ in v)
        got = sum(1 for s in samples if s["series"] == series)
        if got != n_runs:
            raise SystemExit(f"{series}: {got} GEO samples joined, {n_runs} ENA runs")

    # contrasts: (series, name, reference label, other label, meaning, select(s) -> 'control'|'PEC'|None, paired)
    C = []
    for arm, tag in (("Sedentary", "sed"), ("Exercise", "ex")):
        C.append(("GSE99963", f"gse99963_{tag}_post_vs_pre", "Pre", "Post",
                  f"{arm} arm (9 days of high-fat diet{' with resistance exercise' if arm == 'Exercise' else ' alone'}): after vs before, paired by subject",
                  lambda s, arm=arm: None if s["arm"] != arm else ("control" if s["time"] == "Pre" else "PEC"), True))
    for sex in ("Female", "Male"):
        C.append(("GSE144304", f"gse144304_{sex.lower()}_frail_vs_fit", "Fit", "Frail",
                  f"old {sex.lower()}s, (pre-)frail vs fit (GEO group, Fried criteria), not paired",
                  lambda s, sex=sex: None if s["sex"] != sex or s["group"] == "Young" else ("control" if s["group"] == "Fit" else "PEC"), False))
    for sex in ("Female", "Male"):
        C.append(("GSE144304", f"gse144304_{sex.lower()}_old_vs_young", "Young", "Old",
                  f"{sex.lower()}s, old (fit and frail) vs young, not paired",
                  lambda s, sex=sex: None if s["sex"] != sex else ("control" if s["group"] == "Young" else "PEC"), False))
    for tr in ("untrained", "trained"):
        for t in ("1h", "4h"):
            C.append(("GSE120862", f"gse120862_{tr}_{t}_ex_vs_ne", "NE", "Ex",
                      f"{tr}, {t[0]} h after one-legged exercise: exercised vs rested leg, paired by subject",
                      lambda s, tr=tr, t=t: None if s["training"] != tr or s["time"] != t else ("control" if s["leg"] == "NE" else "PEC"), True))
    C.append(("GSE120862", "gse120862_baseline_trained_vs_untrained", "untrained", "trained",
              "rest (baseline), rested (NE) leg: after vs before 2 months of aerobic training, paired by subject",
              lambda s: None if s["time"] != "baseline" or s["leg"] != "NE" else ("control" if s["training"] == "untrained" else "PEC"), True))
    # Domen et al. 2026 Fig. 4 groups, added 2026-11-04. These are the groups that reproduce the
    # paper's printed AUC and p (exports/exercise_score/README.md on PR 212), not the panel labels:
    # GSE99963 pools both arms, GSE144304 is young vs fit old adults (frail left out), and
    # GSE120862 is all untrained vs all trained samples, every time point and both legs.
    C.append(("GSE99963", "gse99963_fig4_post_vs_pre", "Pre", "Post",
              "Fig. 4D: after vs before, sedentary and exercise arms pooled, paired by subject",
              lambda s: "control" if s["time"] == "Pre" else "PEC", True))
    C.append(("GSE144304", "gse144304_fig4_fit_vs_young", "Young", "Fit",
              "Fig. 4D: fit old adults (panel: Ex) vs young adults (panel: Sed), frail left out, not paired",
              lambda s: {"Young": "control", "Fit": "PEC"}.get(s["group"]), False))
    C.append(("GSE120862", "gse120862_fig4_trained_vs_untrained", "untrained", "trained",
              "Fig. 4C: all trained vs all untrained samples, every time point and both legs, paired by subject",
              lambda s: "control" if s["training"] == "untrained" else "PEC", True))

    contrasts = []
    for series, name, ref, other, meaning, sel, paired in C:
        rows = []
        for s in samples:
            if s["series"] != series:
                continue
            c = sel(s)
            if c is not None:
                rows.append((s, c))
        rows.sort(key=lambda x: (x[0]["subject"], x[1]))
        n = Counter(c for _, c in rows)
        if paired:
            per = Counter((s["subject"], c) for s, c in rows)
            bad = {k: v for k, v in per.items() if per[(k[0], "control")] != per[(k[0], "PEC")]}
            if bad:
                raise SystemExit(f"{name}: subjects with unequal samples per group {bad}")
            if n["control"] != n["PEC"]:
                raise SystemExit(f"{name}: unbalanced pairs {n}")
        with open(out_dir / f"contrast_{name}.design.tsv", "w") as f:
            f.write("sample\tcondition\tpair\tgenotype\tsubject\tgroup\n")
            for s, c in rows:
                pair = f"{name}:{s['subject']}" if paired else f"{name}:{s['run']}"
                f.write(f"{s['run']}\t{c}\t{pair}\t{s['subject']}\t{s['subject']}\t{ref if c == 'control' else other}\n")
        with open(out_dir / f"contrast_{name}.cjs", "w") as f:
            for s, c in rows:
                f.write(f"{s['run']},{-1 if c == 'control' else 1}\n")
        lens = sorted({s["mean_read_nt"] for s, _ in rows})
        contrasts.append(dict(series=series, contrast=name, reference=ref, other=other, meaning=meaning,
                              n_reference=n["control"], n_other=n["PEC"], paired=paired,
                              deseq_design="pair_condition" if paired else "condition",
                              reads_million=round(sum(s["reads"] for s, _ in rows) / 1e6, 1),
                              mean_read_nt_range=f"{lens[0]}-{lens[-1]}"))

    keys = ["series", "run", "gsm", "biosample", "title", "subject", "sex", "arm", "time", "group",
            "training", "leg", "reads", "mean_read_nt", "fastq_ftp", "fastq_md5"]
    with open(out_dir / "samples.tsv", "w") as f:
        w = csv.DictWriter(f, keys, delimiter="\t", extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(samples)
    with open(out_dir / "contrasts.tsv", "w") as f:
        w = csv.DictWriter(f, list(contrasts[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(contrasts)
    for c in contrasts:
        print(c["contrast"], c["n_reference"], c["n_other"], c["reads_million"], c["mean_read_nt_range"])


if __name__ == "__main__":
    main()
