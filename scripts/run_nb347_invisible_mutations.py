"""Notebook 347: how much measured PTEN abundance variation a reduced alphabet cannot see.

Reads the rules in scripts/nb347_invisible_mutations_config.json (written before the run),
checks the data against them, and writes tables/347_invisible_mutations_*,
figures/347_SF*_*, and reports/347_invisible_mutations_pten_vamp_seq.md.

Run from the repository root (about one minute on a laptop):

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
        scripts/run_nb347_invisible_mutations.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import invisible_mutations_utils as im  # noqa: E402

ENCODINGS = ["hp_lehninger2", "dayhoff6", "composition", "identity"]
GROUPINGS = ["full", "same_position"]
SCOPES = ["all", "reference_group", "other_groups"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fasta(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines()[1:])


def check_inputs(cfg: dict) -> dict:
    files = cfg["dataset"]["files"]
    for f in files.values():
        got = sha256(ROOT / f["path"])
        if got != f["sha256"]:
            raise SystemExit(f"checksum mismatch for {f['path']}: {got}")
    meta = json.loads((ROOT / files["metadata"]["path"]).read_text())
    dna = meta["targetGenes"][0]["targetSequence"]["sequence"]
    reference = im.translate(dna)
    uniprot = read_fasta(ROOT / files["reference"]["path"])
    if reference != uniprot:
        raise SystemExit("translated MaveDB target does not equal UniProt P60484")
    return {"reference": reference, "metadata": meta}


def check_score_definition(scores: pl.DataFrame, reps: list[str]) -> dict:
    """score = mean of replicates, expts = their count, sd (ddof 1), se = sd/sqrt(n)."""
    m = scores.select(reps).to_numpy().astype(float)
    n = (~np.isnan(m)).sum(1)
    mean = np.nanmean(m, 1)
    sd = np.nanstd(m, 1, ddof=1)
    out = {
        "max_abs_score_minus_replicate_mean": float(
            np.nanmax(np.abs(scores["score"] - mean))
        ),
        "rows_expts_not_replicate_count": int((scores["expts"].to_numpy() != n).sum()),
        "max_abs_sd_minus_ddof1": float(
            np.nanmax(np.abs(scores["sd"].to_numpy() - sd))
        ),
        "max_abs_se_minus_sd_over_sqrt_n": float(
            np.nanmax(np.abs(scores["se"].to_numpy() - sd / np.sqrt(n)))
        ),
        "min_replicates": int(n.min()),
    }
    if (
        out["max_abs_score_minus_replicate_mean"] > 1e-6
        or out["rows_expts_not_replicate_count"]
        or out["max_abs_sd_minus_ddof1"] > 1e-6
        or out["max_abs_se_minus_sd_over_sqrt_n"] > 1e-6
    ):
        raise SystemExit(f"score columns do not match their stated definition: {out}")
    return out


def grouping_keys(variants: pl.DataFrame, encoding: str, grouping: str):
    enc = [im.encode(s, encoding) for s in variants["sequence"]]
    base = list(
        zip(variants["assay"], variants["condition"], variants["background"], enc)
    )
    if grouping == "same_position":
        base = [k + (p,) for k, p in zip(base, variants["position"])]
    return enc, base


def group_masks(enc: list[str], groups: np.ndarray, ref_enc: str) -> np.ndarray:
    """Boolean per group: does the group hold the reference's encoded sequence?"""
    is_ref = np.zeros(int(groups.max()) + 1, dtype=bool)
    for e, g in zip(enc, groups):
        if e == ref_enc:
            is_ref[g] = True
    return is_ref


def summaries(y, v, groups, is_ref, w=None) -> dict[str, dict]:
    return {
        "all": im.floor_summary(y, groups, v, w),
        "reference_group": im.floor_summary(y, groups, v, w, subset=is_ref),
        "other_groups": im.floor_summary(y, groups, v, w, subset=~is_ref),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", default=str(ROOT / "scripts/nb347_invisible_mutations_config.json")
    )
    ap.add_argument(
        "--n-boot", type=int, default=None, help="override the config (tests)"
    )
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    tables, figures, reports = ROOT / "tables", ROOT / "figures", ROOT / "reports"
    T = "347_invisible_mutations"

    inputs = check_inputs(cfg)
    reference = inputs["reference"]
    reps = cfg["uncertainty_fields"]["replicates"]
    scores = pl.read_csv(
        ROOT / cfg["dataset"]["files"]["scores"]["path"],
        null_values="NA",
        infer_schema_length=None,
    )
    score_check = check_score_definition(scores, reps)

    assay, condition, background = (
        "PTEN VAMP-seq",
        "single condition",
        "PTEN_HUMAN P60484",
    )
    variants, excluded, synonymous = im.build_variants(
        scores, reference, assay=assay, condition=condition, background=background,
        replicate_columns=reps,
    )  # fmt: skip
    y = variants["y"].to_numpy()
    v = variants["se"].to_numpy() ** 2
    if np.isnan(v).any():
        raise SystemExit("a variant has no replicate-based se")
    # reported score must equal the pooled mean when nothing was pooled
    single = variants.filter(pl.col("n_rows_pooled") == 1)
    assert np.allclose(single["y"], single["reported_score"])

    # ---- mappings -------------------------------------------------------------------
    map_rows = []
    for name in ["identity", "dayhoff6", "hp_lehninger2"]:
        m = im.alphabet_mapping(name)
        for aa in im.STANDARD_AA:
            map_rows.append({"encoding": name, "residue": aa, "symbol": m[aa]})
    pl.DataFrame(map_rows).write_csv(
        tables / f"{T}_alphabet_mappings.tsv", separator="\t"
    )

    # ---- exclusions -----------------------------------------------------------------
    excl_counts = (
        excluded.with_columns(pl.col("reason").str.replace(r" \(.*", ""))
        .group_by("reason")
        .len()
        .sort("reason")
    )
    syn_row = pl.DataFrame({"reason": ["synonymous (summarised separately)"],
                            "len": [synonymous.height]}, schema_overrides={"len": pl.UInt32})  # fmt: skip
    pl.concat([excl_counts, syn_row]).rename({"len": "n_rows"}).write_csv(
        tables / f"{T}_exclusion_counts.tsv", separator="\t"
    )
    excluded.write_csv(tables / f"{T}_excluded_rows.tsv", separator="\t")

    # ---- groups per encoding ----------------------------------------------------------
    positions = np.unique(variants["position"].to_numpy())
    pos_index = np.searchsorted(positions, variants["position"].to_numpy())
    rows, var_cols, group_rows, setup = [], {}, [], {}
    for enc_name in ENCODINGS:
        ref_enc = im.encode(reference, enc_name)
        for grouping in GROUPINGS:
            enc, keys = grouping_keys(variants, enc_name, grouping)
            g = im.group_ids(keys)
            is_ref = group_masks(enc, g, ref_enc)
            setup[(enc_name, grouping)] = (g, is_ref)
            for scope, s in summaries(y, v, g, is_ref).items():
                rows.append(
                    {"encoding": enc_name, "grouping": grouping, "scope": scope, **s}
                )
            if grouping == "full":
                var_cols[f"group_{enc_name}"] = g
                var_cols[f"in_reference_group_{enc_name}"] = is_ref[g]
                sizes = np.bincount(g)
                var_cols[f"group_size_{enc_name}"] = sizes[g]
                df = variants.with_columns(pl.Series("g", g)).filter(
                    pl.col("g").is_in(np.nonzero(sizes >= 2)[0].tolist())
                )
                for (gid,), sub in df.group_by(["g"], maintain_order=True):
                    cls = set(sub["abundance_class"].drop_nulls().to_list())
                    yy = sub["y"].to_numpy()
                    group_rows.append(
                        {
                            "encoding": enc_name,
                            "group": gid,
                            "is_reference_group": bool(is_ref[gid]),
                            "n_variants": sub.height,
                            "n_positions": sub["position"].n_unique(),
                            "mean": float(yy.mean()),
                            "sd_ddof0": float(yy.std()),
                            "min": float(yy.min()),
                            "max": float(yy.max()),
                            "range": float(yy.max() - yy.min()),
                            "sse": float(((yy - yy.mean()) ** 2).sum()),
                            "mean_se": float(sub["se"].mean()),
                            "has_low_and_wt_like": bool({1.0, 4.0} <= cls),
                            "members": ";".join(sub["hgvs_pro"].to_list()),
                        }
                    )
    summary = pl.DataFrame(rows)
    groups_df = pl.DataFrame(group_rows)
    groups_df.write_csv(tables / f"{T}_collision_groups.tsv", separator="\t")

    # identity control
    ident = summary.filter(
        (pl.col("encoding") == "identity") & (pl.col("scope") == "all")
    )
    assert (ident["floor_mse"] == 0).all() and (ident["n_groups_collided"] == 0).all()

    # ---- bootstrap over positions -------------------------------------------------------
    bcfg = cfg["bootstrap"]
    n_boot = args.n_boot or bcfg["replicates"]
    rng = np.random.default_rng(bcfg["seed"])
    boot = []
    for b in range(n_boot):
        counts = np.bincount(
            rng.integers(0, len(positions), len(positions)), minlength=len(positions)
        )
        w = counts[pos_index].astype(float)
        for (enc_name, grouping), (g, is_ref) in setup.items():
            if enc_name == "identity":
                continue
            for scope, s in summaries(y, v, g, is_ref, w).items():
                boot.append({"rep": b, "encoding": enc_name, "grouping": grouping, "scope": scope,
                             **{k: s[k] for k in ("floor_mse", "floor_rmse", "collided_rmse",
                                                  "floor_fraction", "noise_ratio",
                                                  "noise_adjusted_floor_mse",
                                                  "fraction_collided")}})  # fmt: skip
    boot = pl.DataFrame(boot)
    metrics = ["floor_mse", "floor_rmse", "collided_rmse", "floor_fraction", "noise_ratio",
               "noise_adjusted_floor_mse", "fraction_collided"]  # fmt: skip
    ci = boot.group_by(["encoding", "grouping", "scope"]).agg(
        [pl.col(m).quantile(0.025, "linear").alias(f"{m}_lo") for m in metrics]
        + [pl.col(m).quantile(0.975, "linear").alias(f"{m}_hi") for m in metrics]
    )
    summary = summary.join(ci, on=["encoding", "grouping", "scope"], how="left")
    summary.write_csv(tables / f"{T}_summary.tsv", separator="\t")

    # ---- class discordance, synonymous noise check, sensitivity -------------------------
    disc = (
        groups_df.group_by("encoding")
        .agg(
            pl.len().alias("n_collision_groups"),
            pl.col("has_low_and_wt_like").sum().alias("n_groups_low_and_wt_like"),
            pl.col("n_variants")
            .filter(pl.col("has_low_and_wt_like"))
            .sum()
            .alias("n_variants_in_those_groups"),
        )
        .sort("encoding")
    )
    disc.write_csv(tables / f"{T}_class_discordance.tsv", separator="\t")

    syn_scores = np.array(synonymous["reported_score"].to_list(), dtype=float)
    syn_se2 = np.array(
        [np.var(r, ddof=1) / len(r) for r in synonymous["replicates"].to_list()],
        dtype=float,
    )
    syn_check = {
        "n_synonymous": int(len(syn_scores)),
        "mean_score": float(syn_scores.mean()),
        "var_scores_ddof1": float(syn_scores.var(ddof=1)),
        "mean_se2": float(syn_se2.mean()),
        "ratio_var_to_mean_se2": float(syn_scores.var(ddof=1) / syn_se2.mean()),
    }

    wt = {
        "assay": assay, "condition": condition, "background": background,
        "variant_id": "wild type (mean of synonymous variants)", "hgvs_pro": "p.=",
        "position": 0, "ref": None, "alt": None, "sequence": reference,
        "y": syn_check["mean_score"], "reported_score": syn_check["mean_score"],
        "n_replicates": syn_check["n_synonymous"], "sd": float(np.sqrt(syn_check["var_scores_ddof1"])),
        "se": float(np.sqrt(syn_check["var_scores_ddof1"] / syn_check["n_synonymous"])),
        "replicates": [], "abundance_class": None, "n_rows_pooled": syn_check["n_synonymous"],
    }  # fmt: skip
    with_wt = pl.concat([variants, pl.DataFrame([wt], schema=variants.schema)])
    sens = []
    yw, vw = with_wt["y"].to_numpy(), with_wt["se"].to_numpy() ** 2
    for enc_name in ENCODINGS:
        enc, keys = grouping_keys(with_wt, enc_name, "full")
        g = im.group_ids(keys)
        is_ref = group_masks(enc, g, im.encode(reference, enc_name))
        for scope, s in summaries(yw, vw, g, is_ref).items():
            sens.append({"encoding": enc_name, "scope": scope, **s})
    sens = pl.DataFrame(sens)
    sens.write_csv(tables / f"{T}_sensitivity_with_wild_type.tsv", separator="\t")

    # ---- variant table ----------------------------------------------------------------
    replicate_text = [";".join(f"{x:.6g}" for x in r) for r in variants["replicates"]]
    variants.drop("replicates").with_columns(
        pl.Series("replicate_scores", replicate_text),
        *[pl.Series(k, val) for k, val in var_cols.items()],
    ).write_csv(tables / f"{T}_variants.tsv", separator="\t")

    # ---- examples and k-mer check -------------------------------------------------------
    kcfg = cfg["kmer_check"]
    import sourmash

    ex_rng = np.random.default_rng(347)
    examples, kmer_rows = [], []
    seq_by_hgvs = dict(zip(variants["hgvs_pro"], variants["sequence"]))
    for enc_name in kcfg["encodings"]:
        gd = groups_df.filter(pl.col("encoding") == enc_name)
        extremes = gd.sort("range", descending=True).head(3)
        rest = gd.filter(~pl.col("group").is_in(extremes["group"].to_list()))
        random = rest[
            ex_rng.choice(rest.height, size=min(3, rest.height), replace=False).tolist()
        ]
        for kind, part in [
            ("selected extreme (largest range)", extremes),
            ("random", random),
        ]:
            for row in part.iter_rows(named=True):
                members = row["members"].split(";")
                examples.append({"encoding": enc_name, "selection": kind, **row})
                shown = members[:6]
                a = seq_by_hgvs[shown[0]]
                for other in shown[1:]:
                    b = seq_by_hgvs[other]
                    ea, eb = im.encode(a, enc_name), im.encode(b, enc_name)
                    flag = {"hp_lehninger2": "hp", "dayhoff6": "dayhoff"}[enc_name]
                    for k in kcfg["ksizes"]:
                        mh = sourmash.MinHash(
                            n=0, ksize=k, scaled=1, seed=42, **{flag: True}
                        )
                        ma, mb = mh.copy_and_clear(), mh.copy_and_clear()
                        ma.add_protein(a)
                        mb.add_protein(b)
                        kmer_rows.append({
                            "encoding": enc_name, "group": row["group"], "variant_a": shown[0],
                            "variant_b": other, "k": k,
                            "encoded_sequences_equal": ea == eb,
                            "encoded_kmer_sets_equal": im.kmer_set(ea, k) == im.kmer_set(eb, k),
                            "sourmash_hash_sets_equal": set(mh.seq_to_hashes(a, is_protein=True))
                            == set(mh.seq_to_hashes(b, is_protein=True)),
                            "sourmash_scaled1_sketches_equal": ma == mb,
                            "n_hashes": len(ma),
                            "protein20_kmer_sets_equal": im.kmer_set(a, k) == im.kmer_set(b, k),
                        })  # fmt: skip
    examples = pl.DataFrame(examples)
    examples.write_csv(tables / f"{T}_examples.tsv", separator="\t")
    kmer = pl.DataFrame(kmer_rows)
    kmer.write_csv(tables / f"{T}_kmer_check.tsv", separator="\t")
    assert (
        kmer["encoded_kmer_sets_equal"].all()
        and kmer["sourmash_scaled1_sketches_equal"].all()
    )

    # ---- provenance -------------------------------------------------------------------
    code_paths = ["scripts/run_nb347_invisible_mutations.py", "scripts/nb347_figures.py",
                  "scripts/nb347_report.py", "scripts/nb347_invisible_mutations_config.json",
                  "notebooks/invisible_mutations_utils.py", "data/347_pten_vamp_seq"]  # fmt: skip
    # The last commit that changed the code or data, not HEAD: committing the outputs
    # must not change this file on the next run.
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", *code_paths],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    code_dirty = bool(
        subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *code_paths],
                       capture_output=True, text=True).stdout.strip()
    )  # fmt: skip
    prov = {
        "config": cfg,
        "checksums_verified": {
            k: f["sha256"] for k, f in cfg["dataset"]["files"].items()
        },
        "reference_length": len(reference),
        "reference_matches_uniprot": True,
        "score_definition_check": score_check,
        "synonymous_noise_check": syn_check,
        "n_bootstrap": n_boot,
        "last_commit_changing_code_or_data": commit,
        "code_had_uncommitted_changes": code_dirty,
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ("numpy", "polars", "matplotlib", "sourmash")
        },
        "command": "/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/run_nb347_invisible_mutations.py",
    }
    (tables / f"{T}_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    import nb347_figures

    nb347_figures.make_all(summary, groups_df, variants, examples, figures, T_FIG="347")
    import nb347_report

    nb347_report.write(
        reports / "347_invisible_mutations_pten_vamp_seq.md",
        cfg=cfg, summary=summary, groups_df=groups_df, variants=variants, excl=pl.read_csv(
            tables / f"{T}_exclusion_counts.tsv", separator="\t"),
        disc=disc, syn_check=syn_check, sens=sens, kmer=kmer, examples=examples,
        score_check=score_check, n_boot=n_boot, n_positions=len(positions),
    )  # fmt: skip
    print("done")


if __name__ == "__main__":
    main()
