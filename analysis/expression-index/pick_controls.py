"""Choose the six controls from annotation_summary.tsv by fixed rules, and write controls.tsv.

The rules were written before any sample was indexed. Each pick takes the highest MMseqs2
bit score among proteins whose best MMseqs2 hit (against Swiss-Prot with Ascidiacea
removed) has a description of the stated kind; ties go to the protein name, alphabetically.

  everywhere (2)
    one cytoplasmic ribosomal protein: description is "Small ribosomal subunit protein eS<n>"
    or "Large ribosomal subunit protein eL<n>" (UniProt's current names; the e families
    exist only in the eukaryotic cytoplasmic ribosome). The first version of this rule
    looked for "40S/60S ribosomal protein", names Swiss-Prot no longer uses, and found
    none; it was changed before any sample was indexed.
    one cytoplasmic actin: description starts "Actin, cytoplasmic".
  Botryllus only (2)
    no row at any E-value from MMseqs2, phmmer or jackhmmer; the two longest such
    proteins. (The searches removed every Ascidiacea entry, so "matched only to a
    tunicate" cannot be read from them.)
  muscle and mitochondria-rich samples (2)
    one mitochondrial protein: description ends ", mitochondrial" and does not contain
    "ribosomal";
    one muscle protein: description contains "muscle" and does not start "Actin".

The expected pattern is written into controls.tsv here and is never changed after a run.

    python pick_controls.py annotation_summary.tsv controls.tsv
"""

import sys

import polars as pl

EXPECTED = {
    "everywhere": "present in every sample",
    "Botryllus only": "present in Botryllus samples only",
    "muscle and mitochondria-rich samples": "highest in muscle and mitochondria-rich samples",
}


def top(df, cond, n=1):
    return (df.filter(cond)
            .sort(["mmseqs2_best_bits", "protein"], descending=[True, False])
            .head(n))


def main(summary, out):
    t = pl.read_csv(summary, separator="\t")
    d = pl.col("mmseqs2_best_description")
    picks = [
        ("everywhere", "cytoplasmic ribosomal protein",
         top(t, d.str.contains(r"^(Small|Large) ribosomal subunit protein e[SL]\d+$"))),
        ("everywhere", "cytoplasmic actin", top(t, d.str.starts_with("Actin, cytoplasmic"))),
        ("Botryllus only", "no hit from any tool at any E-value",
         t.filter((pl.col("mmseqs2_rows") == 0) & (pl.col("phmmer_rows") == 0) & (pl.col("jackhmmer_rows") == 0))
          .sort(["length", "protein"], descending=[True, False]).head(2)),
        ("muscle and mitochondria-rich samples", "mitochondrial protein",
         top(t, d.str.ends_with(", mitochondrial") & ~d.str.contains("ribosomal"))),
        ("muscle and mitochondria-rich samples", "muscle protein",
         top(t, d.str.contains("muscle") & ~d.str.starts_with("Actin"))),
    ]
    rows = []
    for group, rule, df in picks:
        for r in df.iter_rows(named=True):
            rows.append({"protein": r["protein"], "group": group, "rule": rule,
                         "expected": EXPECTED[group], "length": r["length"],
                         "mmseqs2_best_target": r["mmseqs2_best_target"],
                         "mmseqs2_best_description": r["mmseqs2_best_description"],
                         "mmseqs2_best_species": r["mmseqs2_best_species"],
                         "mmseqs2_best_bits": r["mmseqs2_best_bits"],
                         "mmseqs2_best_evalue": r["mmseqs2_best_evalue"],
                         "in_dark_txt": r["dark"]})
    if len(rows) != 6:
        raise SystemExit(f"expected 6 controls, got {len(rows)}")
    pl.DataFrame(rows).write_csv(out, separator="\t")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main(*sys.argv[1:3])
