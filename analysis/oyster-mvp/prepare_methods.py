"""Turn SPLASH and DE-kupl output at one depth into standard tables.

SPLASH (unit: anchor). Significant: pval_opt_corrected < 0.05. Each
significant anchor keeps SPLASH's own statistics. Its direction is
descriptive only: per library, the anchor's counts summed over all targets
(from the sample-anchor-target count dump), divided by that library's
trimmed pairs (per million); log2 of mean breed over mean wild, with 0.5
added to each mean. This is never a test.

DE-kupl (unit: k-mer, then contig). merged-diff-counts holds contigs made
from the k-mers that passed DE-kupl's filter (BH-adjusted p <= 0.05 and
|log2FC| >= log2(1.5), set in the run config). The project rule is
padj < 0.05, so a contig whose padj equals 0.05 exactly is flagged.
log2FC is breed over wild (condition B over A).

Writes, in --out:
  splash_anchors.tsv, splash_sequences.tsv (one row per compactor of a
  significant anchor; group_id = the anchor),
  dekupl_contigs.tsv, dekupl_sequences.tsv, method_counts.json
"""
import argparse
import gzip
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

SAMPLES = ["DW1", "DW2", "DW3", "ZW1", "ZW2", "ZW3"]
WILD, BREED = SAMPLES[:3], SAMPLES[3:]


def count_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return sum(1 for _ in f)


def splash(splash_dir, pairs, out):
    d = Path(splash_dir)
    scores = pd.read_csv(d / "oyster.after_correction.scores.tsv", sep="\t")
    tested = count_lines(d / "oyster.after_correction.all_anchors.tsv") - 1 \
        if (d / "oyster.after_correction.all_anchors.tsv").exists() else None
    sig = scores[scores["pval_opt_corrected"] < 0.05].copy()
    keep = ["anchor", "pval_opt", "pval_opt_corrected", "effect_size_bin", "M", "anch_uniqTargs",
            "number_nonzero_samples", "target_entropy", "most_freq_target_1", "cnt_most_freq_target_1",
            "most_freq_target_2", "cnt_most_freq_target_2"]
    sig = sig[[c for c in keep if c in sig.columns]]
    # Per-library anchor counts from the dump, only for significant anchors.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(sig["anchor"]) + "\n")
        anchor_list = f.name
    dumps = sorted((d / "oyster_dumps").glob("*.satc.dump")) or sorted(d.glob("*dumps/*.satc.dump"))
    if not dumps:
        raise SystemExit(f"no SATC dumps under {d}")
    # Dump lines are "count anchor+target sample" (--satc_merge_dump_format
    # splash, the default) or "sample anchor target count" (satc format).
    awk = ("NR==FNR{s[$1]=1;next} "
           "NF==3{a=substr($2,1,31); if(a in s) c[a\"\\t\"$3]+=$1; next} "
           "NF==4{if($2 in s) c[$2\"\\t\"$1]+=$4} "
           "END{for(k in c) print k\"\\t\"c[k]}")
    res = subprocess.run(["awk", awk, anchor_list, *map(str, dumps)],
                         check=True, capture_output=True, text=True)
    rows = [line.split("\t") for line in res.stdout.splitlines()]
    counts = pd.DataFrame(rows, columns=["anchor", "sample", "count"]).astype({"count": "int64"})
    if counts.empty or not set(counts["sample"]) <= set(SAMPLES):
        raise SystemExit(f"anchor counts did not parse from {dumps[0]}")
    counts = counts.pivot_table(index="anchor", columns="sample", values="count", fill_value=0)
    counts = counts.reindex(columns=SAMPLES, fill_value=0)
    sig = sig.merge(counts.add_prefix("count_"), left_on="anchor", right_index=True, how="left")
    for s in SAMPLES:
        sig[f"count_{s}"] = sig[f"count_{s}"].fillna(0).astype("int64")
        sig[f"per_million_{s}"] = sig[f"count_{s}"] / pairs[s] * 1e6
    mb = sig[[f"per_million_{s}" for s in BREED]].mean(axis=1)
    mw = sig[[f"per_million_{s}" for s in WILD]].mean(axis=1)
    sig["log2_breed_over_wild_descriptive"] = ((mb + 0.5) / (mw + 0.5)).map(math.log2)
    sig["direction_descriptive"] = sig["log2_breed_over_wild_descriptive"].map(
        lambda x: "higher in breed" if x > 0 else "higher in wild" if x < 0 else "equal")
    sig = sig.sort_values(["pval_opt_corrected", "effect_size_bin", "anchor"], ascending=[True, False, True])
    sig["rank"] = range(1, len(sig) + 1)
    sig.to_csv(out / "splash_anchors.tsv", sep="\t", index=False)

    # SPLASH 2.11.9 runs Compactors (by default) on two subsets of the
    # significant anchors: the top 20,000 by effect_size_bin and the top
    # 10,000 by target entropy. The union of both files is kept.
    comp_paths = sorted((d / "oyster_compactors").glob("*.tsv"))
    comp = pd.concat([pd.read_csv(p, sep="\t") for p in comp_paths], ignore_index=True)
    comp = comp.drop_duplicates(["anchor", "compactor"]).copy()
    comp = comp[comp["anchor"].isin(set(sig["anchor"]))]
    comp["group_id"] = "splash:" + comp["anchor"]
    comp["seq_id"] = "splash:" + comp["anchor"] + ":c" + comp.groupby("anchor").cumcount().add(1).astype(str)
    comp["sequence_kind"] = "compactor"
    comp = comp.rename(columns={"compactor": "dna"})
    # Anchors without a compactor are still graded, from anchor + each of
    # its two most frequent targets (62 nt, gap 0).
    have = set(comp["anchor"])
    rest = []
    for row in sig[~sig["anchor"].isin(have)].itertuples(index=False):
        for j, t in enumerate([row.most_freq_target_1, row.most_freq_target_2], 1):
            if isinstance(t, str) and set(t) <= set("ACGT"):
                rest.append(dict(seq_id=f"splash:{row.anchor}:t{j}", group_id=f"splash:{row.anchor}",
                                 anchor=row.anchor, dna=row.anchor + t, support=None, exact_support=None,
                                 sequence_kind="anchor+target"))
    seqs = pd.concat([comp[["seq_id", "group_id", "anchor", "dna", "support", "exact_support", "sequence_kind"]],
                      pd.DataFrame(rest)], ignore_index=True)
    seqs.to_csv(out / "splash_sequences.tsv", sep="\t", index=False)
    sig["has_compactor"] = sig["anchor"].isin(have)
    sig.to_csv(out / "splash_anchors.tsv", sep="\t", index=False)
    return dict(splash_tested_anchors=tested, splash_scored_anchors=len(scores),
                splash_significant_anchors=len(sig), splash_compactor_files=[p.name for p in comp_paths],
                splash_compactors=len(comp), splash_anchors_with_compactor=len(have),
                splash_anchor_target_sequences=len(rest))


def dekupl(dekupl_dir, out):
    d = Path(dekupl_dir) / "out"
    kd = d / "wild_vs_breed_kmer_counts"
    merged = pd.read_csv(kd / "merged-diff-counts.tsv.gz", sep="\t")
    merged = merged.rename(columns={merged.columns[0]: "nb_merged_kmers"})
    merged["contig_id"] = [f"dekupl:{i + 1}" for i in range(len(merged))]
    merged["padj_equals_0.05"] = merged["pvalue"] == 0.05
    merged["direction"] = merged["log2FC"].map(lambda x: "higher in breed" if x > 0 else "higher in wild")
    merged["contig_length"] = merged["contig"].str.len()
    merged = merged.sort_values(["pvalue", "log2FC", "contig_id"], key=lambda c: c.abs() if c.name == "log2FC" else c,
                                ascending=[True, False, True])
    merged["rank"] = range(1, len(merged) + 1)
    merged.to_csv(out / "dekupl_contigs.tsv", sep="\t", index=False)
    merged.rename(columns={"contig_id": "seq_id", "contig": "dna"})[["seq_id", "dna"]].to_csv(
        out / "dekupl_sequences.tsv", sep="\t", index=False)
    raw = d / "kmer_counts" / "raw-counts.tsv.gz"
    return dict(dekupl_kmers_after_recurrence_filter=count_lines(raw) - 1 if raw.exists() else None,
                dekupl_kmers_tested=count_lines(kd / "raw_pvals.txt.gz"),
                dekupl_de_kmers=count_lines(kd / "diff-counts.tsv.gz") - 1,
                dekupl_contigs=len(merged), dekupl_contigs_padj_equal_0_05=int(merged["padj_equals_0.05"].sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-dekupl", action="store_true")
    p.add_argument("--skip-splash", action="store_true")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = {s: json.loads((root / "trimmed" / a.depth / f"{s}.validated.json").read_text())["pairs_kept"]
             for s in SAMPLES}
    summary = dict(depth=a.depth, trimmed_pairs=pairs)
    if not a.skip_splash:
        summary.update(splash(root / "splash" / a.depth, pairs, out))
    if not a.skip_dekupl:
        summary.update(dekupl(root / "dekupl" / a.depth, out))
    (out / "method_counts.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
