#!/usr/bin/env python3
"""Step 0 of notebook 300: the smallest E-value one exact window of L residues can reach.

Usage:
  evalue_floor.py --swissprot uniprot_sprot.fasta.gz --ka-fits ka_fits.tsv \
      --composition swissprot=a.json --composition uniref90=b.json --composition uhgp90=c.json \
      --out-dir results/step0

Two estimates, kept under different names because neither is a kmerseek measurement:

- `expected_hits_counting`: N_start x Pr(word). Pr(word) is the product of the class
  frequencies of the window's residues, measured on the database itself. N_start =
  n_residues - n_sequences x (L - 1) is the number of places a word of length L can start.
- `region_evalue_ka_fit`: K m n e^(-r_database lambda S), the formula kmerseek uses for a
  region (`KaParams::evalue`, kmerseek src/rust/search.rs). S = L, the score of L matches at
  +1 each. m = L. n = the database's k-mer count divided by scaled, as the index stores it.
  lambda is the positive root of u e^x + (1 - u) e^(-C x) = 1 (`karlin_altschul_lambda`),
  with u the chance a window position and a database position share a class: the window's
  own class composition against the database's. K and r_database are the medians over the
  dark-set 0.4 indexes that store a fit for this setting; those indexes were built on other
  proteomes, so carrying K and r_database to Swiss-Prot, UniRef90 and UHGP-90 is an
  assumption, not a measurement.

Both are taken on the same 1_000 windows of 60 residues drawn from Swiss-Prot, read as
their first L residues for L = 9..60, and summarised by the median over windows.

A window shorter than k cannot hold a seed, so L < k gives no value for that setting.

The pair estimate for the gate: two exact windows inside one 30-residue peptide, split by
one residue that does not match. `expected_hits_counting_pair` = N_start x Pr(first a) x
Pr(last 29 - a) x n_splits, a = 14, n_splits = the number of places the unmatched residue
could sit with both windows at least k long (a chance count for each split is added).
A pair can never beat the single 30-residue window it sits in, which is also reported.
"""

import argparse
import gzip
import json

from pathlib import Path

import numpy as np
import polars as pl

from alphabets import CANONICAL, CLUSTERS, KMERSEEK_ALPHABETS, class_of

L_MIN, L_MAX = 9, 60
GATE_L = 14
GATE_PEPTIDE = 30
GATE_E = 0.01
DISAGREE_FACTOR = 3.0
MASSSPEC_KSIZES = [5, 6, 7, 8, 9]


def read_fasta(path: str):
    """Yield (accession, sequence) from a gzipped UniProt FASTA."""
    opener = gzip.open if path.endswith(".gz") else open
    name, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


def sample_windows(path: str, n: int, length: int, seed: int) -> pl.DataFrame:
    """n windows of `length` canonical residues, start positions uniform over residues."""
    names, seqs = [], []
    for name, seq in read_fasta(path):
        if len(seq) >= length:
            names.append(name)
            seqs.append(seq)
    starts_per_seq = np.array([len(s) - length + 1 for s in seqs], dtype=np.int64)
    cumulative = np.cumsum(starts_per_seq)
    rng = np.random.default_rng(seed)
    canonical = set(CANONICAL)
    rows = []
    while len(rows) < n:
        flat = int(rng.integers(cumulative[-1]))
        i = int(np.searchsorted(cumulative, flat, side="right"))
        start = flat - (int(cumulative[i - 1]) if i else 0)
        window = seqs[i][start:start + length]
        if set(window) <= canonical:
            rows.append((names[i], start, window))
    return pl.DataFrame(rows, schema=["accession", "start_0based", "window"], orient="row")


def class_frequencies(composition: dict, alphabet: str) -> np.ndarray:
    """Class frequencies over the 20 canonical residues of one database."""
    classes = class_of(alphabet)
    freq = np.zeros(len(CLUSTERS[alphabet]))
    for aa in CANONICAL:
        freq[classes[aa]] += composition["letters"].get(aa, 0)
    return freq / freq.sum()


def karlin_altschul_lambda(u: np.ndarray, penalty: float) -> np.ndarray:
    """Positive root of u e^x + (1 - u) e^(-penalty x) = 1, elementwise; 0 where none exists.

    The same bisection as kmerseek's `karlin_altschul_lambda`, run on a whole array.
    """
    u = np.asarray(u, dtype=float)
    w = 1.0 - u
    has_root = (u > 0.0) & (u - penalty * w < 0.0)
    us, ws = np.where(has_root, u, 0.5), np.where(has_root, w, 0.5)

    def f(x):
        return us * np.exp(x) + ws * np.exp(-penalty * x)

    upper = np.ones_like(u)
    for _ in range(11):
        upper = np.where(f(upper) <= 1.0, upper * 2.0, upper)
    lower = np.zeros_like(u)
    for _ in range(80):
        mid = 0.5 * (lower + upper)
        above = f(mid) > 1.0
        upper = np.where(above, mid, upper)
        lower = np.where(above, lower, mid)
    root = 0.5 * (lower + upper)
    return np.where(has_root & (root >= 1e-6), root, 0.0)


def settings_table(ka_fits: pl.DataFrame) -> pl.DataFrame:
    """One row per kmerseek setting: median K and r_database over the indexes that fit it."""
    fits = ka_fits.with_columns(pl.col("fitted").cast(pl.Utf8) == "true")
    by = ["alphabet", "ksize", "scaled", "mismatch_penalty", "xdrop", "lowcomplexity"]
    table = (
        fits.group_by(by)
        .agg(
            n_indexes=pl.len(),
            n_indexes_fitted=pl.col("fitted").sum(),
            ka_k_median=pl.col("k").filter(pl.col("fitted")).median(),
            ka_r_database_median=pl.col("slope").filter(pl.col("fitted")).median(),
            ka_k_min=pl.col("k").filter(pl.col("fitted")).min(),
            ka_k_max=pl.col("k").filter(pl.col("fitted")).max(),
        )
        .with_columns(ka_fit_status=pl.when(pl.col("n_indexes_fitted") > 0)
                      .then(pl.lit("fitted")).otherwise(pl.lit("no fit stored")))
    )
    massspec = pl.DataFrame({
        "alphabet": ["massspec18"] * len(MASSSPEC_KSIZES),
        "ksize": MASSSPEC_KSIZES,
        "scaled": [1] * len(MASSSPEC_KSIZES),
        "mismatch_penalty": [None] * len(MASSSPEC_KSIZES),
        "xdrop": [None] * len(MASSSPEC_KSIZES),
        "lowcomplexity": ["false"] * len(MASSSPEC_KSIZES),
        "n_indexes": [0] * len(MASSSPEC_KSIZES),
        "n_indexes_fitted": [0] * len(MASSSPEC_KSIZES),
        "ka_fit_status": ["no index: alphabet not in kmerseek"] * len(MASSSPEC_KSIZES),
    })
    table = pl.concat([table, massspec], how="diagonal_relaxed")
    return table.sort(by, nulls_last=True).with_columns(
        setting=pl.format("{}.k{}.s{}.C{}", "alphabet", "ksize", "scaled",
                          pl.col("mismatch_penalty").cast(pl.Utf8).fill_null("-"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--swissprot", required=True)
    ap.add_argument("--ka-fits", required=True)
    ap.add_argument("--composition", action="append", required=True, help="name=path.json")
    ap.add_argument("--n-windows", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    compositions = {}
    for item in args.composition:
        name, path = item.split("=", 1)
        compositions[name] = json.loads(Path(path).read_text())

    windows = sample_windows(args.swissprot, args.n_windows, L_MAX, args.seed)
    windows.write_csv(out / "windows_swissprot.tsv", separator="\t")
    lengths = np.arange(L_MIN, L_MAX + 1)

    ka_fits = pl.read_csv(args.ka_fits, separator="\t", infer_schema_length=0).with_columns(
        pl.col("ksize", "scaled").cast(pl.Int64),
        pl.col("mismatch_penalty", "xdrop", "slope", "k").cast(pl.Float64),
    ).filter(pl.col("alphabet").is_in(KMERSEEK_ALPHABETS))
    settings = settings_table(ka_fits)
    settings.write_csv(out / "settings.tsv", separator="\t")

    rows = []
    for alphabet in CLUSTERS:
        classes = class_of(alphabet)
        encoded = np.array([[classes[aa] for aa in w] for w in windows["window"]])
        alphabet_settings = settings.filter(pl.col("alphabet") == alphabet)
        for db, comp in compositions.items():
            freq = class_frequencies(comp, alphabet)
            log_pr = np.cumsum(np.log(freq[encoded]), axis=1)  # [window, L-1]
            n_res, n_seq = comp["n_residues"], comp["n_sequences"]
            n_start = n_res - n_seq * (lengths - 1)
            counting = np.exp(log_pr[:, lengths - 1] + np.log(n_start))  # [window, L]
            # class composition of each window's first L residues, for u
            onehot = np.eye(len(freq))[encoded]  # [window, 60, class]
            comp_prefix = np.cumsum(onehot, axis=1)[:, lengths - 1, :] / lengths[None, :, None]
            u = comp_prefix @ freq  # [window, L]
            for s in alphabet_settings.iter_rows(named=True):
                k = s["ksize"]
                ok = lengths >= k
                ka = np.full_like(counting, np.nan)
                if s["ka_fit_status"] == "fitted":
                    n_db = (n_res - n_seq * (k - 1)) / s["scaled"]
                    lam = karlin_altschul_lambda(u, s["mismatch_penalty"])
                    ka = (s["ka_k_median"] * lengths[None, :] * n_db
                          * np.exp(-s["ka_r_database_median"] * lam * lengths[None, :]))
                for j, L in enumerate(lengths):
                    if not ok[j]:
                        continue
                    rows.append({
                        "setting": s["setting"], "alphabet": alphabet, "ksize": k,
                        "scaled": s["scaled"], "mismatch_penalty": s["mismatch_penalty"],
                        "database": db, "window_length": int(L),
                        "expected_hits_counting": float(np.median(counting[:, j])),
                        "expected_hits_counting_q25": float(np.quantile(counting[:, j], 0.25)),
                        "expected_hits_counting_q75": float(np.quantile(counting[:, j], 0.75)),
                        "region_evalue_ka_fit": (float(np.median(ka[:, j]))
                                                 if s["ka_fit_status"] == "fitted" else None),
                        "ka_fit_status": s["ka_fit_status"],
                    })
            # pair of windows inside one 30-residue peptide, per k of this alphabet
            for k in sorted(set(alphabet_settings["ksize"].to_list())):
                n_splits = GATE_PEPTIDE - 2 * k
                if n_splits < 1:
                    continue
                a = GATE_L
                pr_pair = (log_pr[:, a - 1]
                           + (log_pr[:, GATE_PEPTIDE - 1] - log_pr[:, a]))
                n_start = n_res - n_seq * (GATE_PEPTIDE - 1)
                pair = np.exp(pr_pair + np.log(n_start) + np.log(n_splits))
                rows.append({
                    "setting": f"{alphabet}.k{k}.pair", "alphabet": alphabet, "ksize": k,
                    "scaled": None, "mismatch_penalty": None, "database": db,
                    "window_length": GATE_PEPTIDE,
                    "expected_hits_counting_pair": float(np.median(pair)),
                    "ka_fit_status": "pair",
                })
    est = pl.DataFrame(rows, infer_schema_length=None)
    est = est.with_columns(
        counting_over_ka=pl.col("expected_hits_counting") / pl.col("region_evalue_ka_fit"),
    ).with_columns(
        estimates_disagree=(pl.col("counting_over_ka") > DISAGREE_FACTOR)
        | (pl.col("counting_over_ka") < 1 / DISAGREE_FACTOR),
    )
    est.write_csv(out / "floor_estimates.tsv", separator="\t")

    single = est.filter(pl.col("ka_fit_status") != "pair")
    min_l = single.group_by(["setting", "alphabet", "ksize", "scaled", "mismatch_penalty",
                             "database", "ka_fit_status"]).agg(
        min_length_counting=pl.col("window_length")
        .filter(pl.col("expected_hits_counting") < GATE_E).min(),
        min_length_ka_fit=pl.col("window_length")
        .filter(pl.col("region_evalue_ka_fit") < GATE_E).min(),
        expected_hits_counting_at_14=pl.col("expected_hits_counting")
        .filter(pl.col("window_length") == GATE_L).first(),
        region_evalue_ka_fit_at_14=pl.col("region_evalue_ka_fit")
        .filter(pl.col("window_length") == GATE_L).first(),
    ).sort(["alphabet", "ksize", "scaled", "mismatch_penalty", "database"], nulls_last=True)
    min_l.write_csv(out / "floor_min_length.tsv", separator="\t")

    gate_db = "uhgp90"
    gate = []
    for alphabet in CLUSTERS:
        rows_a = min_l.filter((pl.col("alphabet") == alphabet) & (pl.col("database") == gate_db))
        pair_a = est.filter((pl.col("alphabet") == alphabet) & (pl.col("database") == gate_db)
                            & (pl.col("ka_fit_status") == "pair"))
        single_counting = rows_a["expected_hits_counting_at_14"].drop_nulls()
        single_ka = rows_a["region_evalue_ka_fit_at_14"].drop_nulls()
        pair_vals = pair_a["expected_hits_counting_pair"].drop_nulls()
        best = {
            "alphabet": alphabet,
            "database": gate_db,
            "best_expected_hits_counting_at_14": single_counting.min() if len(single_counting) else None,
            "best_region_evalue_ka_fit_at_14": single_ka.min() if len(single_ka) else None,
            "best_expected_hits_counting_pair": pair_vals.min() if len(pair_vals) else None,
        }
        ksizes = settings.filter(pl.col("alphabet") == alphabet)["ksize"]
        best["smallest_ksize"] = int(ksizes.min())
        alive = any(v is not None and v < GATE_E for k, v in best.items() if k.startswith("best"))
        best["verdict"] = "alive" if alive else "dead"
        if alive:
            best["reason"] = f"a 14-residue window or a pair in 30 residues reaches E < {GATE_E}"
        elif best["smallest_ksize"] > GATE_L and 2 * best["smallest_ksize"] + 1 > GATE_PEPTIDE:
            best["reason"] = (f"smallest k {best['smallest_ksize']} is longer than {GATE_L}, and "
                              f"two seeds do not fit in {GATE_PEPTIDE} residues")
        else:
            best["reason"] = f"every estimate is >= {GATE_E}"
        gate.append(best)
    pl.DataFrame(gate).write_csv(out / "gate_uhgp90.tsv", separator="\t")
    print(pl.DataFrame(gate))


if __name__ == "__main__":
    main()
