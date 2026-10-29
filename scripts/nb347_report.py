"""Markdown report for notebook 347 (called by run_nb347_invisible_mutations.py).

Every number in the report is read from the tables the run just wrote. The verdict for
each encoding comes from the stop rule in the config, not from the text here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

PRIMARY = ["hp_lehninger2", "dayhoff6"]
NAMES = {
    "hp_lehninger2": "hp_lehninger2",
    "dayhoff6": "dayhoff6",
    "composition": "amino-acid counts",
    "identity": "protein20 (identity)",
}


def name(enc):
    return NAMES.get(enc, enc)


def row(summary, enc, grouping="full", scope="all"):
    return summary.filter(
        (pl.col("encoding") == enc)
        & (pl.col("grouping") == grouping)
        & (pl.col("scope") == scope)
    ).row(0, named=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def pct(x):
    return f"{100 * x:.1f}%"


def verdict(r, rule) -> str:
    if r["n_groups_collided"] < 30 or r["n_variants_collided"] < 100:
        return "Inconclusive because coverage is inadequate."
    if r["noise_ratio_lo"] is not None and r["noise_ratio_lo"] > 1:
        return (
            "Detectable information loss, but practical importance is not established."
        )
    return (
        "No detectable information loss in this dataset; this does not validate the "
        "representation."
    )


def residues_example(variants, groups_df, enc):
    """Two members of the same-position group with the largest range, shown residue by
    residue with their encoded strings."""
    import invisible_mutations_utils as im

    g = (
        groups_df.filter((pl.col("encoding") == enc) & ~pl.col("is_reference_group"))
        .sort("range", descending=True)
        .row(0, named=True)
    )
    by = {r["hgvs_pro"]: r for r in variants.iter_rows(named=True)}
    members = sorted(g["members"].split(";"), key=lambda m: by[m]["y"])
    lo, hi = by[members[0]], by[members[-1]]
    p = lo["position"]
    a, b = max(0, p - 11), min(len(lo["sequence"]), p + 10)
    sa, sb = lo["sequence"][a:b], hi["sequence"][a:b]
    ea, eb = im.encode(sa, enc), im.encode(sb, enc)
    match = "".join("|" if x == y else " " for x, y in zip(sa, sb))
    ematch = "".join("|" if x == y else " " for x, y in zip(ea, eb))
    n_id = match.count("|")
    lab_a = f"{lo['hgvs_pro']} (score {lo['y']:.2f} ± {lo['se']:.2f} s.e., n={lo['n_replicates']})"
    lab_b = f"{hi['hgvs_pro']} (score {hi['y']:.2f} ± {hi['se']:.2f} s.e., n={hi['n_replicates']})"
    w = 15
    return (
        f"Residues {a + 1}-{b} of PTEN, the two ends of the {enc} group at position {p} "
        f"({g['n_variants']} variants, range {g['range']:.2f}):\n\n"
        "```text\n"
        f"{'protein20':<{w}}{sa}   {lab_a}\n"
        f"{'':<{w}}{match}   {n_id} of {len(sa)} identical\n"
        f"{'protein20':<{w}}{sb}   {lab_b}\n\n"
        f"{enc:<{w}}{ea}\n"
        f"{'':<{w}}{ematch}   {ematch.count('|')} of {len(ea)} identical\n"
        f"{enc:<{w}}{eb}\n"
        "```\n"
    )


def write(path: Path, *, cfg, summary, groups_df, variants, excl, disc, syn_check, sens, kmer,
          examples, score_check, n_boot, n_positions, kmer_floors):  # fmt: skip
    N = int(summary["N"][0])
    ref_len = len(variants["sequence"][0])
    rep_lo, rep_hi = int(variants["n_replicates"].min()), int(variants["n_replicates"].max())
    n_pooled = int((variants["n_rows_pooled"] > 1).sum())
    pooled_text = (
        "no sequence was measured twice"
        if n_pooled == 0
        else f"{n_pooled} sequences were measured in more than one row and pooled"
    )
    n_examples = examples.filter(pl.col("encoding").is_in(PRIMARY)).height
    sd_all = float(np.sqrt(summary["total_variance"][0]))
    syn_ratio = syn_check["ratio_var_to_mean_se2"]
    L = []
    w = L.append
    w("# Mutations a reduced alphabet cannot see: PTEN abundance (notebook 347)\n")
    w("Written by `scripts/run_nb347_invisible_mutations.py`. Rules were fixed in "
      "`scripts/nb347_invisible_mutations_config.json` before the run (git history shows "
      "the order).\n")  # fmt: skip

    w("## Question\n")
    w("If two PTEN variants have the same sequence after encoding with a reduced alphabet, "
      "any predictor that reads only the encoded sequence must give them the same value. "
      "Their reduced k-mer sets and sketches are then identical too. How much of the "
      "measured abundance variation sits inside such groups? This report answers that for "
      "one assay, one protein and the single substitutions that were measured. It says "
      "nothing about other proteins, other functions, or homology search.\n")  # fmt: skip

    w("## Data\n")
    d = cfg["dataset"]
    w(
        f"- Assay: {d['name']}, MaveDB score set `{d['mavedb_score_set']}` ({d['license']}). "
        f"{d['publication']}."
    )
    w(f"- What is measured: {cfg['assay']}")
    w(
        "- Outcome: `score`, a unitless abundance score: wild type = 1, mean of nonsense "
        "variants = 0 (min-max normalised by the authors). Higher means more PTEN protein in "
        "the cell."
    )
    w(
        f"- Reference: PTEN, UniProt P60484, {ref_len} aa. The MaveDB target DNA translates to "
        "exactly the UniProt sequence (checked by the script)."
    )
    w(
        f"- Replicates: {rep_lo} to {rep_hi} per variant (`score1`..`score8`). The script checked that "
        f"`score` is their mean (largest difference {score_check['max_abs_score_minus_replicate_mean']:.1e}), "
        "`sd` their sample standard deviation and `se` = `sd`/sqrt(n)."
    )
    w(f"- Files and SHA-256 checksums: see the config; downloaded {d['downloaded']}.\n")  # fmt: skip

    w("### Rows kept and left out\n")
    w(f"{N} single missense variants were kept, one per distinct protein sequence "
      f"({pooled_text}), at {n_positions} of {ref_len} positions.\n")  # fmt: skip
    w("| Reason left out | Rows |\n|---|---|")
    for r in excl.iter_rows(named=True):
        w(f"| {r['reason']} | {r['n_rows']} |")
    w("")

    w("### Alphabets (repository mappings)\n")
    w("Copied from `ALPHABET_AA_GROUPS` in `notebooks/ortholog_analysis_utils.py`, which "
      "transcribes sourmash's `encodings.rs`, the tables kmerseek calls for `dayhoff6` and "
      "`hp_lehninger2`. `tests/test_invisible_mutations_utils.py` checks every residue pair "
      "against sourmash's own hashes. Full table: "
      "`tables/347_invisible_mutations_alphabet_mappings.tsv`.\n")  # fmt: skip
    w("| Alphabet | Classes |\n|---|---|")
    w(
        "| hp_lehninger2 | hydrophobic `AFGILMPVWY`, polar `CDEHKNQRST` (G and P count as hydrophobic, C as polar) |"
    )
    w("| dayhoff6 | `C`, `AGPST`, `DENQ`, `HKR`, `ILMV`, `FWY` |")
    w("| protein20 (control) | the 20 residues themselves |")
    w("| amino-acid counts (control) | the count of each residue, order ignored |\n")
    w("kmerseek's other 16 alphabets (added 2026-10-20, see below) are copied from "
      "kmerseek's `src/rust/alphabets.rs` into `KMERSEEK_CLUSTERS` in "
      "`notebooks/invisible_mutations_utils.py`; a test compares the two.\n")  # fmt: skip

    w("## Where the variants fall\n")
    w("![Collision groups](../figures/347_SF1_collision_groups_pten_vamp_seq.png)\n")
    w("For single substitutions, two variants share an encoded sequence in only two ways: "
      "both change the same position to residues of the same class, or both swap a "
      "residue for another of its own class, which leaves the encoded sequence equal to "
      "wild type's. That second group spans many positions. With only two HP classes, "
      "almost every substitution shares its class with another substitution at the same "
      "position, so almost every variant is in some group.\n")  # fmt: skip
    w(
        "| Encoding | Collision groups | Variants in them | Share | In the wild-type-sequence group | Other groups |\n|---|---|---|---|---|---|"
    )
    for enc in ["hp_lehninger2", "dayhoff6", "composition", "identity"]:
        a, rg, og = (
            row(summary, enc),
            row(summary, enc, scope="reference_group"),
            row(summary, enc, scope="other_groups"),
        )
        w(f"| {NAMES[enc]} | {a['n_groups_collided']} | {int(a['n_variants_collided'])} | "
          f"{pct(a['fraction_collided'])} | {int(rg['n_variants_collided'])} | "
          f"{int(og['n_variants_collided'])} in {og['n_groups_collided']} groups |")  # fmt: skip
    w("\nAmino-acid counts put two substitutions together when they make the same swap "
      "(say L to V) at different positions, so this control groups variants that no "
      "reduced sequence groups, and the reverse. It is neither coarser nor finer.\n")  # fmt: skip

    w("## Error floors\n")
    w("![Error floors](../figures/347_SF2_error_floors_pten_vamp_seq.png)\n")
    w(
        "For a group g of n_g variants with mean scores y_i, the best single value is the "
        "group mean. Over all N variants, each counted once:\n"
    )
    w(
        "```text\nfloor_MSE = sum over groups, sum over members (y_i - mean_g)^2 / N\n```\n"
    )
    w("Variants alone in their group add 0. The collision-only numbers divide the same sum "
      "by the number of variants in groups of 2 or more instead of N. Intervals are 95% "
      f"percentile intervals from {n_boot} resamples of the {n_positions} positions "
      "(all variants at a position drawn together); they describe other positions of PTEN "
      "measured like these, not other proteins. A resample holds about 63% of the "
      "distinct positions (the rest are repeats), so groups that span positions, the "
      "wild-type-sequence group and the amino-acid-count groups, get fewer distinct "
      "members and their intervals sit low next to the full-data value.\n")  # fmt: skip
    w(f"All {N} variants have a spread (SD) of {sd_all:.3f} score units "
      f"(variance {sd_all**2:.4f}). That is the error of predicting the overall mean for every variant.\n")  # fmt: skip
    w(
        "| Encoding | Grouping | Floor RMSE (95% interval) | Floor MSE / total variance (95% interval) | Collision-only RMSE | Noise-only RMSE |\n|---|---|---|---|---|---|"
    )
    for enc in ["hp_lehninger2", "dayhoff6", "composition", "identity"]:
        for grouping in ["full", "same_position"]:
            r = row(summary, enc, grouping)
            gname = (
                "same encoded sequence"
                if grouping == "full"
                else "same sequence and position"
            )
            w(f"| {NAMES[enc]} | {gname} | {f3(r['floor_rmse'])} ({f3(r['floor_rmse_lo'])}-{f3(r['floor_rmse_hi'])}) | "
              f"{f2(r['floor_fraction'])} ({f2(r['floor_fraction_lo'])}-{f2(r['floor_fraction_hi'])}) | "
              f"{f3(r['collided_rmse'])} | {f3(np.sqrt(r['expected_noise_mse']))} |")  # fmt: skip
    w(
        "\nThe protein20 control has no collisions and a floor of exactly 0, as it must "
        "after one record per sequence.\n"
    )
    w("### Does the wild-type-sequence group dominate?\n")
    w(
        "| Encoding | Floor MSE, all groups | from the wild-type-sequence group | from the other groups |\n|---|---|---|---|"
    )
    for enc in PRIMARY + ["composition"]:
        a, rg, og = (
            row(summary, enc),
            row(summary, enc, scope="reference_group"),
            row(summary, enc, scope="other_groups"),
        )
        share = rg["floor_mse"] / a["floor_mse"] if a["floor_mse"] else float("nan")
        w(
            f"| {NAMES[enc]} | {a['floor_mse']:.4f} | {rg['floor_mse']:.4f} ({pct(share)}) | {og['floor_mse']:.4f} |"
        )
    w("\nSplitting groups by position as well (second row of each encoding in the table "
      "above) removes the cross-position part of the wild-type-sequence group.\n")  # fmt: skip

    w("## Measurement noise\n")
    w(
        "Each variant's score is a mean over n_i replicates, with estimated variance "
        "se_i^2 = sd_i^2 / n_i. If every member of a group had the same true score and "
        "errors were independent between variants, the expected within-group sum of squares "
        "around the fitted group mean is\n"
    )
    w(
        "```text\nexpected_noise_SS_g = sum_i se_i^2 - sum_i se_i^2 / n_g = (1 - 1/n_g) * sum_i se_i^2\n```\n"
    )
    w("The 1/n_g term is there because the group mean is fitted from the same noisy "
      "scores (each y_i pulls the mean toward itself). The noise ratio is the observed "
      "within-group sum of squares divided by this; 1 means the spread is what replicate "
      "noise alone predicts, more than 1 means more spread than that.\n")  # fmt: skip
    w(
        "| Encoding | Grouping | Noise ratio (95% interval) | Noise-adjusted floor MSE (estimate) |\n|---|---|---|---|"
    )
    for enc in PRIMARY + ["composition"]:
        for grouping in ["full", "same_position"]:
            r = row(summary, enc, grouping)
            if r["n_groups_collided"] == 0:
                continue
            gname = (
                "same encoded sequence"
                if grouping == "full"
                else "same sequence and position"
            )
            w(f"| {NAMES[enc]} | {gname} | {f2(r['noise_ratio'])} ({f2(r['noise_ratio_lo'])}-{f2(r['noise_ratio_hi'])}) | "
              f"{r['noise_adjusted_floor_mse']:.4f} |")  # fmt: skip
    w("")
    w("Assumptions behind the noise term:\n")
    w(
        "- Replicates of one variant are independent. Errors shared by all replicates of a "
        "variant (a wrong barcode-to-variant link in the shared library, for example) are not "
        "in se, so they would appear as spread between variants."
    )
    w(
        "- Errors of different variants are independent. MaveDB gives no covariance, so none "
        "is used. Each replicate was normalised to its own wild type and nonsense mean, which "
        "can shift all variants of that replicate together."
    )
    w("- se from 2 replicates is very uncertain; it is unbiased for the variance, which is "
      "what the sum uses.\n")  # fmt: skip
    w("### Check of the noise model with synonymous variants\n")
    w(f"The {syn_check['n_synonymous']} synonymous variants all make wild-type protein. "
      f"Their scores vary {syn_ratio:.2f} times more (variance {syn_check['var_scores_ddof1']:.4f}) "
      f"than their replicate se predicts (mean se^2 {syn_check['mean_se2']:.4f}). "
      "So either se misses part of the noise, or the DNA change itself alters abundance "
      "(for example through mRNA), or both. The data cannot tell these apart.\n")  # fmt: skip
    w(f"If se understates the noise by that whole factor of {syn_ratio:.2f} for every "
      "variant (an assumption, not a measurement; this check was planned, dividing by it was "
      "not), the noise ratios become:\n")  # fmt: skip
    w(
        "| Encoding | Grouping | Noise ratio / synonymous factor (lower end of interval / factor) |\n|---|---|---|"
    )
    for enc in PRIMARY:
        for grouping in ["full", "same_position"]:
            r = row(summary, enc, grouping)
            gname = (
                "same encoded sequence"
                if grouping == "full"
                else "same sequence and position"
            )
            w(
                f"| {NAMES[enc]} | {gname} | {r['noise_ratio'] / syn_ratio:.2f} ({r['noise_ratio_lo'] / syn_ratio:.2f}) |"
            )
    w("\nThe noise-adjusted floor is left out of the figure for this reason: it depends on "
      "se being right, and this check says it may not be.\n")  # fmt: skip

    w("## Do these differences matter in practice?\n")
    w("No tolerance for prediction error is established for this assay and none was "
      "supplied, so there is no pass/fail line. As a description only: the authors put each "
      "variant in an abundance class. Collision groups holding both a 'low' and a "
      "'wt-like' variant:\n")  # fmt: skip
    w(
        "| Encoding | Collision groups | With both 'low' and 'wt-like' | Variants in those groups |\n|---|---|---|---|"
    )
    for r in disc.iter_rows(named=True):
        w(
            f"| {name(r['encoding'])} | {r['n_collision_groups']} | {r['n_groups_low_and_wt_like']} | {r['n_variants_in_those_groups']} |"
        )
    w("")

    w("## Examples\n")
    w("![Example groups](../figures/347_SF3_example_groups_pten_vamp_seq.png)\n")
    w("Panel a shows the range of every collision group, so the three largest (selected "
      "extremes) can be seen against the rest. Panels b and c show those three and three "
      f"groups drawn at random, per encoding. All {n_examples} are in "
      "`tables/347_invisible_mutations_examples.tsv`.\n")  # fmt: skip
    for enc in PRIMARY:
        w(residues_example(variants, groups_df, enc))

    w("## K-mers and sketches\n")
    k_ok = (
        kmer["encoded_kmer_sets_equal"].all() and kmer["sourmash_hash_sets_equal"].all()
    )
    s_ok = kmer["sourmash_scaled1_sketches_equal"].all()
    p_diff = (~kmer["protein20_kmer_sets_equal"]).sum()
    w(f"For {kmer.height} variant pairs from the {n_examples} example groups at k = "
      f"{', '.join(str(k) for k in sorted(kmer['k'].unique()))}, the encoded k-mer sets were "
      f"{'identical in every case' if k_ok else 'NOT always identical'}, and sourmash "
      f"MinHash sketches (scaled=1, seed 42, `hp` or `dayhoff`, the hashing kmerseek uses for "
      f"these alphabets) were {'identical in every case' if s_ok else 'NOT always identical'}. "
      f"The 20-letter k-mer sets differed in {p_diff} of {kmer.height}. Every pair comes "
      "from a group whose members have the same encoded sequence, so equal k-mer sets and "
      "sketches are expected at any k. This checks that the sketching code gives equal "
      "sketches for equal encoded sequences. It is not a finding about the data. Table: "
      "`tables/347_invisible_mutations_kmer_check.tsv`.\n")  # fmt: skip

    w("## Sensitivity: adding wild type as a variant\n")
    w(
        "Adding one wild-type record (outcome = mean of the synonymous scores, "
        f"{syn_check['mean_score']:.3f}) to the wild-type-sequence group:\n"
    )
    w("| Encoding | Floor RMSE without | with |\n|---|---|---|")
    for enc in PRIMARY:
        s = sens.filter((pl.col("encoding") == enc) & (pl.col("scope") == "all")).row(
            0, named=True
        )
        w(
            f"| {NAMES[enc]} | {row(summary, enc)['floor_rmse']:.5f} | {s['floor_rmse']:.5f} |"
        )
    w("")

    w(all_alphabets_section(cfg, summary, kmer_floors, syn_ratio))
    w("## Conclusion\n")
    w("Rule (from the config): inconclusive if fewer than 30 collision groups or 100 "
      "variants in them; detectable if the lower end of the noise-ratio interval is above "
      "1; no tolerance, so a detectable result cannot be called practically important.\n")  # fmt: skip
    for enc in PRIMARY:
        r = row(summary, enc)
        w(f"- **{enc}**, PTEN VAMP-seq abundance, {N} single missense variants at "
          f"{n_positions} positions: {verdict(r, cfg['stop_rule'])} Noise ratio "
          f"{f2(r['noise_ratio'])} ({f2(r['noise_ratio_lo'])}-{f2(r['noise_ratio_hi'])}); "
          f"floor RMSE {f3(r['floor_rmse'])} against an SD of {sd_all:.3f}.")  # fmt: skip
    w("")
    for enc in PRIMARY:
        r = row(summary, enc)
        rp = row(summary, enc, "same_position")
        w(
            f"- {enc}, if replicate se understates all noise by the synonymous factor "
            f"{syn_ratio:.2f} (not planned; see the noise section): lower end of the noise-ratio "
            f"interval {r['noise_ratio_lo'] / syn_ratio:.2f} for groups by encoded sequence, "
            f"{rp['noise_ratio_lo'] / syn_ratio:.2f} for groups by sequence and position."
        )
    w(
        "\nSo the dayhoff6 result, and the same-position part of the hp_lehninger2 result, "
        "rest on trusting the replicate se; the hp_lehninger2 result with groups spanning "
        "positions still holds under that assumption.\n"
        if all(
            row(summary, e, g)["noise_ratio_lo"] / syn_ratio > 1
            for e, g in [("hp_lehninger2", "full")]
        )
        and not all(
            row(summary, e, g)["noise_ratio_lo"] / syn_ratio > 1
            for e, g in [("dayhoff6", "full"), ("hp_lehninger2", "same_position")]
        )
        else ""
    )
    sec = [e for e in summary["encoding"].unique(maintain_order=True).to_list()
           if e not in PRIMARY + ["composition", "identity"]]  # fmt: skip
    n_det = sum(verdict(row(summary, e), None).startswith("Detectable") for e in sec)
    w(f"- Not planned (added 2026-10-20): {n_det} of kmerseek's {len(sec)} other reduced "
      "alphabets show detectable information loss by the same rule; see the table in "
      "\"All kmerseek alphabets and k-mer sizes\".\n")  # fmt: skip
    w("What this does not show:\n")
    w(
        "- The floor uses the measured scores themselves to set each group mean. It is an "
        "exact floor on squared error for these variants and this encoding, not a trained "
        "predictor, and not a bound for unmeasured variants or other proteins."
    )
    w(
        "- The floor includes measurement noise. The synonymous check above says the "
        "replicate se may understate it."
    )
    w(
        "- The variants are the single substitutions this library happened to measure. "
        "Sparser sampling leaves more variants alone in their group, which makes the floor "
        "look smaller."
    )
    w("- Reduced k-mer sets and sketches keep less than the full reduced sequence, so a "
      "small floor here would not have validated them, and a large one does not show that "
      "reduced-alphabet search fails. It also does not bear on the convergence searches "
      "that were stopped earlier.\n")  # fmt: skip

    w("## Reproduce\n")
    w(
        "```bash\n/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/run_nb347_invisible_mutations.py\n```\n"
    )
    w("Tests:\n")
    w(
        "```bash\n/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 -m pytest -q tests/test_invisible_mutations_utils.py\n```\n"
    )
    w("Outputs: `tables/347_invisible_mutations_*.tsv` (variants, collision groups, summary "
      "with intervals, exclusions, class counts, examples, k-mer check, sensitivity), "
      "`tables/347_invisible_mutations_provenance.json` (config, checksums, versions, git "
      "commit), `tables/347_invisible_mutations_kmer_set_floors.tsv` (every alphabet and "
      "k), `figures/347_SF1-4_*`.\n")  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def all_alphabets_section(cfg, summary, kmer_floors, syn_ratio) -> str:
    """Every kmerseek alphabet, whole sequence and k-mer sets (added 2026-10-20)."""
    extra = cfg["more_alphabets_and_ksizes"]
    L = []
    w = L.append
    w("## All kmerseek alphabets and k-mer sizes (not planned)\n")
    w(f"Added {extra['_added']}\n")
    w(
        "![All alphabets and k-mer sizes](../figures/347_SF4_all_alphabets_ksizes_pten_vamp_seq.png)\n"
    )
    w("### Whole encoded sequence\n")
    w("Same statistics as above, groups = same encoded sequence, coarsest alphabet first. "
      "The verdict uses the planned rule, but was not planned for these alphabets.\n")  # fmt: skip
    w(f"The last column divides the lower end of the noise-ratio interval by the synonymous "
      f"factor {syn_ratio:.2f} (see \"Check of the noise model\"); below 1 means the loss "
      "is no longer detectable if replicate se understates the noise that much.\n")  # fmt: skip
    w("| Alphabet | Letters | Variants in a group of 2+ | Floor RMSE (95% interval) | "
      "Floor / total variance (95% interval) | Noise ratio (95% interval) | Verdict | "
      "Lower end / synonymous factor |\n|---|---|---|---|---|---|---|---|")  # fmt: skip
    sizes = dict(kmer_floors.select("encoding", "alphabet_size").unique().iter_rows())
    for enc in sorted(sizes, key=lambda e: (sizes[e], e)):
        r = row(summary, enc)
        v = (
            "no collisions"
            if r["n_groups_collided"] == 0
            else verdict(r, None).split(",")[0].rstrip(".")
        )
        w(f"| {name(enc)} | {sizes[enc]} | {pct(r['fraction_collided'])} | "
          f"{f3(r['floor_rmse'])} ({f3(r['floor_rmse_lo'])}-{f3(r['floor_rmse_hi'])}) | "
          f"{f2(r['floor_fraction'])} ({f2(r['floor_fraction_lo'])}-{f2(r['floor_fraction_hi'])}) | "
          f"{f2(r['noise_ratio'])} ({f2(r['noise_ratio_lo'])}-{f2(r['noise_ratio_hi'])}) | {v} | "
          f"{f2(r['noise_ratio_lo'] / syn_ratio) if r['noise_ratio_lo'] is not None else 'n/a'} |")  # fmt: skip
    w("")
    w("### Only the set of encoded k-mers\n")
    w(f"{extra['kmer_set_grouping']} Two single substitutions can have the same k-mer set "
      "without the same sequence when every k-mer they change also occurs elsewhere in "
      "PTEN, which happens at small k and with few letters. Value: floor / total "
      "variance (share of score variance hidden). Every alphabet and k, with intervals: "
      "`tables/347_invisible_mutations_kmer_set_floors.tsv`.\n")  # fmt: skip
    ks = sorted(kmer_floors["k"].unique().to_list())
    shown = [k for k in (3, 5, 10, 20, 30) if k in ks]
    w("| Alphabet | Whole sequence | " + " | ".join(f"k={k}" for k in shown)
      + " | From this k on, exactly the whole-sequence value |\n|---|---|"
      + "---|" * len(shown) + "---|")  # fmt: skip
    kf = kmer_floors.filter(pl.col("scope") == "all")
    for enc in sorted(sizes, key=lambda e: (sizes[e], e)):
        seq_floor = row(summary, enc)["floor_mse"]
        sub = kf.filter(pl.col("encoding") == enc).sort("k")
        vals = dict(zip(sub["k"], sub["floor_fraction"]))
        same = [
            k for k, m in zip(sub["k"], sub["floor_mse"]) if abs(m - seq_floor) < 1e-12
        ]
        first = next((k for k in ks if all(kk in same for kk in ks if kk >= k)), None)
        w(f"| {name(enc)} | {f2(row(summary, enc)['floor_fraction'])} | "
          + " | ".join(f2(vals[k]) for k in shown)
          + f" | {first if first is not None else f'above {ks[-1]}'} |")  # fmt: skip
    w("")
    return "\n".join(L) + "\n"
