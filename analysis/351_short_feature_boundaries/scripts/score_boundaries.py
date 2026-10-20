"""Score how close each method puts the start and end of each short feature.

Reads the kmerseek, hmmscan and blastp outputs, turns every reported match into a
placement on the query protein (1-based, inclusive), and scores each ELM instance
by the rules in preregistered.yaml. Writes:

  results/placements.csv.gz      every placement that passed the E-value cutoff
  results/boundary_errors.csv    one row per instance per method and setting
  results/uniform_random_errors.csv  the random-placement control, kept apart
  results/metrics.csv            the four pass/fail numbers per method and setting
  results/verdict.json           PASS, FAIL or DROP THE IDEA, with each condition

Usage: python -I score_boundaries.py <analysis dir> <run dir>
  <run dir> holds kmerseek/<setting>/*.csv.gz, hmmscan/*.domtbl, blastp/*.tsv
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EVALUE_CUTOFF = 0.01
WITHIN = 10
SEED = 351
INF = math.inf
SHUFFLE_TAG = "__shuf"
EXTEND_SUFFIX = "_extend2"

KMERSEEK_COLS = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "region_subseq",
    "target_start",
    "target_end",
    "target_subseq",
    "region_evalue",
    "region_mean_idf",
]
BLAST_COLS = "qseqid sseqid pident length qstart qend sstart send evalue bitscore qseq sseq".split()


def assert_cpu_only():
    """This analysis uses no GPU. Fail if one was handed to the job."""
    for var in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS"):
        assert not os.environ.get(var), f"{var} is set: a GPU was allocated"
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    ), "CUDA_VISIBLE_DEVICES is set"
    assert (
        "torch" not in sys.modules and "cupy" not in sys.modules
    ), "a GPU library is loaded"


# --- reading each tool's output -------------------------------------------------


def read_kmerseek(path, setting):
    """kmerseek regions are 0-based, end-exclusive on the query: start + 1, end as is."""
    df = pd.read_csv(path, usecols=lambda c: c in KMERSEEK_COLS)
    if df.empty:
        return pd.DataFrame()
    ev = pd.to_numeric(df["region_evalue"], errors="coerce").fillna(INF)
    return pd.DataFrame(
        {
            "method": "kmerseek",
            "setting": setting,
            "query": df["query_name"].astype(str).str.split().str[0],
            "start": df["region_start"].astype(int) + 1,
            "end": df["region_end"].astype(int),
            "evalue": ev,
            "target": df["target_name"].astype(str).str.split().str[0],
            "target_start": df["target_start"].astype(int) + 1,
            "target_end": df["target_end"].astype(int),
            "query_residues": df["region_subseq"],
            "target_residues": df["target_subseq"],
        }
    )


def read_domtbl(path):
    """hmmscan --domtblout: the protein is the 'query'; envelope from/to are 1-based."""
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        rows.append(
            {
                "method": "profile",
                "setting": "pfam_hmmscan",
                "query": c[3],
                "start": int(c[19]),
                "end": int(c[20]),
                "evalue": float(c[12]),
                "target": c[1] if c[1] != "-" else c[0],
                "target_start": int(c[15]),
                "target_end": int(c[16]),
                "query_residues": "",
                "target_residues": "",
            }
        )
    return pd.DataFrame(rows)


def read_blast(path):
    """blastp -outfmt 6: keep the best-scoring HSP of each query-target pair."""
    df = pd.read_csv(path, sep="\t", names=BLAST_COLS, header=None)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(
        ["qseqid", "sseqid", "evalue", "bitscore"], ascending=[True, True, True, False]
    )
    df = df.drop_duplicates(["qseqid", "sseqid"], keep="first")
    return pd.DataFrame(
        {
            "method": "alignment",
            "setting": "blastp",
            "query": df["qseqid"].astype(str),
            "start": df["qstart"].astype(int),
            "end": df["qend"].astype(int),
            "evalue": df["evalue"].astype(float),
            "target": df["sseqid"].astype(str),
            "target_start": df["sstart"].astype(int),
            "target_end": df["send"].astype(int),
            "query_residues": df["qseq"],
            "target_residues": df["sseq"],
        }
    )


def load_placements(run_dir, settings):
    run_dir = Path(run_dir)
    parts, ran = [], {}
    for s in settings:
        files = sorted((run_dir / "kmerseek" / s["name"]).glob("*.csv.gz"))
        ran[s["name"]] = (
            bool(files) and (run_dir / "kmerseek" / s["name"] / "DONE").exists()
        )
        parts += [read_kmerseek(f, s["name"]) for f in files]
    files = sorted((run_dir / "hmmscan").glob("*.domtbl"))
    ran["pfam_hmmscan"] = bool(files)
    parts += [read_domtbl(f) for f in files]
    files = sorted((run_dir / "blastp").glob("*.tsv"))
    ran["blastp"] = bool(files)
    parts += [read_blast(f) for f in files]
    parts = [p for p in parts if not p.empty]
    pl = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=["method", "setting", "query", "start", "end", "evalue"]
        )
    )
    pl = pl[pl["evalue"] <= EVALUE_CUTOFF].copy()
    pl["is_shuffle"] = pl["query"].str.contains(SHUFFLE_TAG, regex=False)
    pl["protein"] = pl["query"].str.split(SHUFFLE_TAG, regex=False).str[0]
    return pl, ran


# --- choosing the scored placement ------------------------------------------------


def nearest(cands, start, end):
    """Placement whose midpoint is closest to the true midpoint; ties: lower E, lower start."""
    mid = (start + end) / 2
    d = ((cands["start"] + cands["end"]) / 2 - mid).abs()
    order = np.lexsort(
        (cands["start"].to_numpy(), cands["evalue"].to_numpy(), d.to_numpy())
    )
    return cands.iloc[order[0]]


def best_evalue(cands):
    order = np.lexsort((cands["start"].to_numpy(), cands["evalue"].to_numpy()))
    return cands.iloc[order[0]]


def score_method(inst, pl_real, method, setting):
    by_protein = dict(tuple(pl_real.groupby("protein"))) if not pl_real.empty else {}
    rows = []
    for r in inst.itertuples():
        c = by_protein.get(r.protein)
        row = {
            "instance_id": r.instance_id,
            "protein": r.protein,
            "elm_class": r.elm_class,
            "true_start": r.start,
            "true_end": r.end,
            "true_length": r.length,
            "method": method,
            "setting": setting,
            "n_placements_on_protein": 0 if c is None else len(c),
        }
        if c is None:
            row.update(
                placed_start=None,
                placed_end=None,
                evalue=None,
                target=None,
                start_error=INF,
                end_error=INF,
                blind_start=None,
                blind_end=None,
                blind_start_error=INF,
                blind_end_error=INF,
            )
        else:
            p = nearest(c, r.start, r.end)
            b = best_evalue(c)
            row.update(
                placed_start=int(p["start"]),
                placed_end=int(p["end"]),
                evalue=float(p["evalue"]),
                target=p["target"],
                start_error=abs(int(p["start"]) - r.start),
                end_error=abs(int(p["end"]) - r.end),
                blind_start=int(b["start"]),
                blind_end=int(b["end"]),
                blind_start_error=abs(int(b["start"]) - r.start),
                blind_end_error=abs(int(b["end"]) - r.end),
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --- numbers ------------------------------------------------------------------------


def median_error(df, s="start_error", e="end_error"):
    return float(
        np.median(np.concatenate([df[s].to_numpy(float), df[e].to_numpy(float)]))
    )


def within(df, s="start_error", e="end_error", w=WITHIN):
    return float(((df[s] <= w) & (df[e] <= w)).mean())


def shuffle_rates(inst, pl_shuf, shuffle_names):
    """Primary: share of shuffled proteins with any placement. Secondary: per instance and
    shuffle, the nearest placement has both ends within 10 of the true feature."""
    hit = set(pl_shuf["query"]) if not pl_shuf.empty else set()
    primary = len(hit & shuffle_names) / len(shuffle_names)
    by_q = dict(tuple(pl_shuf.groupby("query"))) if not pl_shuf.empty else {}
    n = close = 0
    for r in inst.itertuples():
        for j in range(1, 11):
            n += 1
            c = by_q.get(f"{r.protein}{SHUFFLE_TAG}{j:02d}")
            if c is not None:
                p = nearest(c, r.start, r.end)
                close += (
                    abs(p["start"] - r.start) <= WITHIN
                    and abs(p["end"] - r.end) <= WITHIN
                )
    return primary, close / n


def uniform_random(inst, lengths, n_draws=None, seed=SEED):
    """One random placement per instance inside its protein, length drawn from the true
    feature lengths. With n_draws (dict protein -> n), the nearest of n draws is kept.
    """
    rng = np.random.default_rng(seed)
    pool = inst["length"].to_numpy()
    rows = []
    for r in inst.itertuples():
        L = lengths[r.protein]
        n = 1 if n_draws is None else max(1, n_draws.get(r.protein, 0))
        best = None
        for _ in range(n):
            ln = int(min(rng.choice(pool), L))
            s = int(rng.integers(1, L - ln + 2))
            cand = (s, s + ln - 1)
            dist = abs((cand[0] + cand[1]) / 2 - (r.start + r.end) / 2)
            if best is None or dist < best[0]:
                best = (dist, cand)
        s, e = best[1]
        rows.append(
            {
                "instance_id": r.instance_id,
                "protein": r.protein,
                "true_start": r.start,
                "true_end": r.end,
                "true_length": r.length,
                "placed_start": s,
                "placed_end": e,
                "start_error": abs(s - r.start),
                "end_error": abs(e - r.end),
                "n_draws": n,
            }
        )
    return pd.DataFrame(rows)


def read_fasta_lengths(path):
    lengths, name = {}, None
    for line in open(path):
        if line.startswith(">"):
            name = line[1:].split()[0]
            lengths[name] = 0
        else:
            lengths[name] += len(line.strip())
    return lengths


def extended(settings):
    """The not-preregistered runs with extension on (report.md, change 2)."""
    return [{**s, "name": s["name"] + EXTEND_SUFFIX} for s in settings]


def verdict(metrics, prereg, ran, settings=None):
    """Apply the preregistered rule to `settings` (default: the preregistered ones)."""
    if settings is None:
        settings = prereg["methods"]["kmerseek"]["settings"]
    k_rows = metrics[metrics["method"] == "kmerseek"]
    prof = metrics[metrics["setting"] == "pfam_hmmscan"]
    prof_median = float(prof["median_error"].iloc[0]) if len(prof) else None
    pf = prereg["pass_fail"]["PASS_requires_all"]
    out = {"cutoffs": pf, "profile_median_error": prof_median, "settings": {}}
    tried = []
    for s in settings:
        name = s["name"]
        if not ran.get(name):
            out["settings"][name] = {"ran": False}
            continue
        m = k_rows[k_rows["setting"] == name].iloc[0]
        a = m["median_error"] <= 5
        b = m["within_10"] >= 0.80
        c = m["shuffle_any_placement_rate"] <= 0.05
        d = prof_median is not None and prof_median >= 3 * m["median_error"]
        out["settings"][name] = {
            "ran": True,
            "ksize": s["ksize"],
            "a_median_error": m["median_error"],
            "a_pass": bool(a),
            "b_within_10": m["within_10"],
            "b_pass": bool(b),
            "c_shuffle_rate": m["shuffle_any_placement_rate"],
            "c_pass": bool(c),
            "d_profile_over_kmerseek": (
                (prof_median / m["median_error"]) if m["median_error"] > 0 else INF
            ),
            "d_pass": bool(d),
            "all_four": bool(a and b and c and d),
            "median_above_k": bool(m["median_error"] > s["ksize"]),
        }
        tried.append(out["settings"][name])
    if not tried:
        out["answer"] = "NOT RUN"
    elif all(t["median_above_k"] for t in tried):
        out["answer"] = "DROP THE IDEA"
    elif any(t["all_four"] for t in tried):
        out["answer"] = "PASS"
    else:
        out["answer"] = "FAIL"
    return out


def main(analysis_dir, run_dir):
    assert_cpu_only()
    a = Path(analysis_dir)
    prereg = yaml.safe_load(open(a / "preregistered.yaml"))
    settings = prereg["methods"]["kmerseek"]["settings"]
    inst = pd.read_csv(a / "data/filtered/instances_kept.tsv", sep="\t")
    lengths = read_fasta_lengths(a / "data/filtered/query_proteins.fasta")
    shuffle_names = {f"{p}{SHUFFLE_TAG}{j:02d}" for p in lengths for j in range(1, 11)}

    pl, ran = load_placements(run_dir, settings + extended(settings))
    res = a / "results"
    res.mkdir(exist_ok=True)
    pl.to_csv(res / "placements.csv.gz", index=False)

    groups = [("kmerseek", s["name"]) for s in settings + extended(settings)] + [
        ("profile", "pfam_hmmscan"),
        ("alignment", "blastp"),
    ]
    errors, metrics = [], []
    for method, setting in groups:
        if not ran.get(setting):
            print(f"{setting}: no output, not scored")
            continue
        sub = pl[pl["setting"] == setting]
        e = score_method(inst, sub[~sub["is_shuffle"]], method, setting)
        primary, secondary = shuffle_rates(inst, sub[sub["is_shuffle"]], shuffle_names)
        n_draws = e.groupby("protein")["n_placements_on_protein"].first().to_dict()
        ur_n = uniform_random(inst, lengths, n_draws=n_draws)
        errors.append(e)
        metrics.append(
            {
                "method": method,
                "setting": setting,
                "n_instances": len(e),
                "n_placed": int((e["n_placements_on_protein"] > 0).sum()),
                "median_error": median_error(e),
                "within_10": within(e),
                "shuffle_any_placement_rate": primary,
                "shuffle_within_10_rate": secondary,
                "blind_median_error": median_error(
                    e, "blind_start_error", "blind_end_error"
                ),
                "blind_within_10": within(e, "blind_start_error", "blind_end_error"),
                "random_nearest_of_n_median_error": median_error(ur_n),
                "random_nearest_of_n_within_10": within(ur_n),
            }
        )
    errors = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame()
    errors.to_csv(res / "boundary_errors.csv", index=False)
    ur = uniform_random(inst, lengths)
    ur.to_csv(res / "uniform_random_errors.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(res / "metrics.csv", index=False)
    v = verdict(metrics, prereg, ran)
    v["extended_not_preregistered"] = verdict(metrics, prereg, ran, extended(settings))
    v["uniform_random"] = {"median_error": median_error(ur), "within_10": within(ur)}
    json.dump(v, open(res / "verdict.json", "w"), indent=2, default=float)
    print(metrics.to_string(index=False))
    print(
        f"uniform random: median {v['uniform_random']['median_error']}, within 10 {v['uniform_random']['within_10']:.3f}"
    )
    print(f"answer under the preregistered rule: {v['answer']}")
    print(
        f"same rule on the extended runs (not preregistered): {v['extended_not_preregistered']['answer']}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
