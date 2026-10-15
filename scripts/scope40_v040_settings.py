#!/usr/bin/env python3
"""Write the SCOPe40 kmerseek 0.4 sweep's settings table: per alphabet, the k-sizes to test and
the mismatch penalties C of its extension settings.

Inputs, all committed:
  tables/274_ksizes_to_test_per_alphabet_human_swissprot.csv
      k_min, k_max and letter_classes per alphabet (notebook 274, PR 101, 6e4752c).
  nextflow-runs/scope40-alphabet-ksize-v040/assets/kappa_by_alphabet.tsv
      kappa, the copy rate of a letter's class in aligned Pfam pairs at 20-30% identity
      (notebook 230; copied from the dark-set branch, c2eaaaa).
  SWISSPROT_PERCENT below: Swiss-Prot 2026_03 composition, the same values as
      scripts/make_nb274.py.

The penalties, as in Figure 11 of the "Information content per alphabet" explainer:
  C_best(q) = -ln(1 - kappa) / ln(1 + kappa / q - kappa)   (Equation 7a)
      the log-odds mismatch penalty for a +1 / -C score when the matched class has share q.
  c_opt  = C_best at q = 1 / classes, every class given an equal share (the c_opt column of
           kappa_by_alphabet.tsv, recomputed here to 3 decimals).
  c_max  = the largest C_best over the alphabet's classes at their Swiss-Prot shares (the
           right end of the bar in Figure 11; classes under 0.5% share are left out, as there).
  c_min  = s2 / (1 - s2), s2 = sum of squared class shares, the chance that two unrelated
           residues share a class (Equation 7b). At C <= c_min a chance position scores zero
           or more on average, so there is no lambda and no E-value.
  c_min_x1.1 = 1.1 x c_min, just above that floor.
  c2     = 2, kmerseek's own setting and the one every earlier extension run used.
An alphabet without a measured kappa (funcgroups8) gets no c_opt and no c_max.

Run:  python scripts/scope40_v040_settings.py
"""

import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
K_TABLE = REPO / "tables" / "274_ksizes_to_test_per_alphabet_human_swissprot.csv"
PIPE = REPO / "nextflow-runs" / "scope40-alphabet-ksize-v040"
KAPPA = PIPE / "assets" / "kappa_by_alphabet.tsv"
OUT = PIPE / "assets" / "settings_per_alphabet.tsv"

# UniProtKB/Swiss-Prot release 2026_03, "6.1 Composition in percent for the complete
# database", https://web.expasy.org/docs/relnotes/relstat.html (as in scripts/make_nb274.py)
SWISSPROT_PERCENT = {
    "A": 8.25, "Q": 3.93, "L": 9.64, "S": 6.66,
    "R": 5.52, "E": 6.71, "K": 5.79, "T": 5.36,
    "N": 4.06, "G": 7.07, "M": 2.41, "W": 1.10,
    "D": 5.46, "H": 2.27, "F": 3.86, "Y": 2.92,
    "C": 1.38, "I": 5.90, "P": 4.75, "V": 6.85,
}  # fmt: skip
MIN_SHARE = 0.005  # Figure 11 leaves out classes below this share
C_KMERSEEK = 2.0

# Checked against Figure 11 of the explainer (artifact H4cCWq5uhBHDk4rDj4nEz8, read
# 2026-10-02): c_min, c_opt, c_max for three alphabets that span the range. The
# explainer rounds c_opt to two decimals, hence the 0.005 tolerance.
EXPECTED = {
    "protein20": (0.064, 0.140, 0.211),
    "gbmr7": (0.728, 0.320, 2.110),
    "hp_kyte_doolittle2": (1.115, 1.570, 2.372),
}


def c_best(kappa: float, q: float) -> float:
    return -math.log(1 - kappa) / math.log(1 + kappa / q - kappa)


def fmt(c: float | None) -> str:
    return "" if c is None else f"{c:.3f}"


def main() -> None:
    total = sum(SWISSPROT_PERCENT.values())
    share = {aa: p / total for aa, p in SWISSPROT_PERCENT.items()}
    kappa = {}
    with open(KAPPA) as fh:
        rows = (line for line in fh if not line.startswith("#"))
        for r in csv.DictReader(rows, delimiter="\t"):
            kappa[r["alphabet"]] = float(r["kappa"])

    out = []
    with open(K_TABLE) as fh:
        for r in csv.DictReader(fh):
            classes = r["letter_classes"].split()
            shares = [sum(share[aa] for aa in cls) for cls in classes]
            s2 = sum(q * q for q in shares)
            c_min = s2 / (1 - s2)
            k = kappa.get(r["alphabet"])
            c_opt = None if k is None else c_best(k, 1 / len(classes))
            c_max = (
                None
                if k is None
                else max(c_best(k, q) for q in shares if q > MIN_SHARE)
            )
            out.append(
                {
                    "alphabet": r["alphabet"],
                    "k_min": int(r["k_min"]),
                    "k_max": int(r["k_max"]),
                    "n_ksizes": int(r["n_ksizes"]),
                    "kappa": "" if k is None else k,
                    "s2": f"{s2:.4f}",
                    "c_min": fmt(c_min),
                    "c_min_x1.1": fmt(1.1 * c_min),
                    "c_opt": fmt(c_opt),
                    "c_max": fmt(c_max),
                    "c2": fmt(C_KMERSEEK),
                }
            )

    for row in out:
        if row["alphabet"] in EXPECTED:
            got = tuple(float(row[c]) for c in ("c_min", "c_opt", "c_max"))
            want = EXPECTED[row["alphabet"]]
            assert all(abs(g - w) < 0.005 for g, w in zip(got, want)), (row, want)
        assert row["n_ksizes"] == row["k_max"] - row["k_min"] + 1, row

    with open(OUT, "w", newline="") as fh:
        fh.write(
            "# Settings of the SCOPe40 kmerseek 0.4 sweep, written by scripts/scope40_v040_settings.py.\n"
            "# k_min..k_max from notebook 274; c_* are mismatch penalties for the extension settings\n"
            "# (see the script's docstring for each one). Every alphabet also gets an exact setting.\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    n_pairs = sum(r["n_ksizes"] for r in out)
    n_searches = sum(
        r["n_ksizes"]
        * (1 + sum(bool(r[c]) for c in ("c_min", "c_min_x1.1", "c_opt", "c_max", "c2")))
        for r in out
    )
    print(f"{len(out)} alphabets, {n_pairs} alphabet-ksize pairs, {n_searches} searches")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
