#!/usr/bin/env python3
"""Copy every number notebook 245 uses out of the executed notebooks that measured it.

Notebook 245 runs no search. Each number it uses was printed by an earlier notebook, and
this script reads that printed table back out of the notebook's saved output, so no
number is typed by hand. It writes three tables under tables/:

* 245_inputs.tsv: one row per single number (quantity, value, unit, kind, source).
* 245_reach_by_k.tsv: notebook 230's fraction of Pfam pairs whose longest exact
  class-identical run is at least k.
* 245_chance_score_by_alphabet.tsv: notebook 241's chance match rate, mismatch penalty
  and expected score of one chance position, per alphabet.

Notebooks 241 and 242 are not on main yet (PRs #46 and #48), so they are read from those
branches with `git show`. Run from the repository root:

    git fetch origin pull/48/head:pr48
    python scripts/make_245_inputs.py

The database sizes and K values are read from the kmerseek docs, so a checkout of
seanome/kmerseek must sit next to this repository (../kmerseek).
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"

SOURCES = {
    "230": ("origin/main", "notebooks/230_hp_class_conservation_in_aligned_homologs.ipynb"),
    "232": ("origin/main", "notebooks/232_mismatch_tolerant_seed_extend.ipynb"),
    "241": ("pr48", "notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb"),
    "242": ("pr48", "notebooks/242_alphabet_ensemble.ipynb"),
}


def notebook(nb):
    ref, path = SOURCES[nb]
    text = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    return json.loads(text)


def printed_tables(nb):
    """Every polars table printed in a notebook's outputs, as lists of row dicts."""
    tables = []
    for cell in notebook(nb)["cells"]:
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain") or ""
            text = "".join(text) if isinstance(text, list) else text
            # One output can print several tables; each starts with a "┌" line.
            for block in text.split("┌")[1:]:
                rows = [ln for ln in block.splitlines() if ln.startswith("│")]
                if len(rows) < 3:
                    continue
                split = [[c.strip() for c in re.split(r"[│┆]", ln)[1:-1]] for ln in rows]
                # polars prints the header, a row of "---", then a row of dtypes.
                dtypes = {"str", "f64", "i64", "u32", "bool", "enum", "---", "cat"}
                header = split[0]
                body = [r for r in split[1:] if not all(c in dtypes for c in r)]
                tables.append([dict(zip(header, r)) for r in body])
    return tables


def find_table(nb, *columns, without=()):
    for t in printed_tables(nb):
        if t and all(c in t[0] for c in columns) and not any(c in t[0] for c in without):
            return t
    raise KeyError(f"no table in notebook {nb} with columns {columns}")


def one(rows, **match):
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} rows match {match}")
    return hits[0]


def first_number(s):
    return float(s.split()[0])


inputs = []


def add(quantity, value, unit, kind, source):
    inputs.append((quantity, value, unit, kind, source))


# --- notebook 230: conservation and reach on aligned Pfam and SCOPe pairs -------------
kappa = find_table("230", "category", "identity_bin", "protein20", "hp_thomas_dill2")
row = one(kappa, category="Pfam", identity_bin="20-30%")
add("kappa_hp_thomas_dill2_pfam_20_30", first_number(row["hp_thomas_dill2"]), "none",
    "measured", "nb 230 section 2, Pfam seed pairs at 20-30% identity")
add("kappa_protein20_pfam_20_30", first_number(row["protein20"]), "none",
    "measured", "nb 230 section 2, Pfam seed pairs at 20-30% identity")
hp = find_table("230", "dataset", "alphabet", "kappa_mean")
add("kappa_hp_thomas_dill2_scope_20_30",
    float(one(hp, dataset="SCOPe cross-family", alphabet="hp_thomas_dill2")["kappa_mean"]),
    "none", "measured", "nb 230 section 7, SCOPe cross-family pairs at 20-30% identity")
runs = find_table("230", "dataset", "identity_bin", "alphabet", "longest_run_mean")
for ds, key in (("Pfam", "pfam"), ("SCOPe cross-family", "scope")):
    r = one(runs, dataset=ds, identity_bin="20-30%", alphabet="hp_thomas_dill2")
    add(f"longest_run_mean_hp_thomas_dill2_{key}_20_30", float(r["longest_run_mean"]),
        "residues", "measured", f"nb 230 section 4, {ds} pairs at 20-30% identity")
    add(f"longest_run_null_mean_hp_thomas_dill2_{key}_20_30",
        float(r["longest_run_null_mean"]), "residues", "measured",
        f"nb 230 section 4, the same {ds} pairs with the target shuffled")

reach = find_table("230", "dataset", "identity_bin", "alphabet", "23", "26", "30")
with open(OUT / "245_reach_by_k.tsv", "w") as fh:
    ks = [c for c in reach[0] if c.isdigit()]
    fh.write("dataset\tidentity_bin\talphabet\tk\tfraction_of_pairs_reachable\n")
    for r in reach:
        for k in ks:
            fh.write(f"{r['dataset']}\t{r['identity_bin']}\t{r['alphabet']}\t{k}\t{r[k]}\n")

# --- notebook 232: mismatch-tolerant seed and extend ---------------------------------
rec = find_table("232", "fpr", "seed", "threshold", "20-30%", "30-40%")
r = one(rec, fpr="0.001", seed="10")
add("recall_seed10_extend_20_30", float(r["20-30%"]), "fraction of pairs", "measured",
    "nb 232 section 2, hp_thomas_dill2, penalty 2, null FPR 0.001 per comparison")
add("recall_seed10_extend_30_40", float(r["30-40%"]), "fraction of pairs", "measured",
    "nb 232 section 2, hp_thomas_dill2, penalty 2, null FPR 0.001 per comparison")

# --- notebook 241: the three-case alphabet sweep on the human proteome --------------
null = find_table("241", "query", "observed_best_rank", "p_random_at_least_as_good")
for q, partner in (("Ced9", "BCL2"), ("P66", "CD47")):
    r = one(null, query=q)
    add(f"best_rank_{partner}", int(r["observed_best_rank"]), "rank", "measured",
        "nb 241 section 6, best rank over every alphabet, k and metric")
    add(f"n_targets_at_best_{partner}", int(r["n_targets_at_best"]), "proteins", "measured",
        "nb 241 section 6, human proteins hit at the arm of the best rank")
    add(f"p_random_protein_as_good_{partner}", float(r["p_random_at_least_as_good"]),
        "fraction of 20,000 draws", "measured", "nb 241 section 6")
pev = find_table("241", "query", "alphabet", "ksize", "bits", "partner_value", "rank",
                 without=("metric",))
for q, partner in (("Ced9", "BCL2"), ("P66", "CD47")):
    best = min((r for r in pev if r["query"] == q), key=lambda r: float(r["partner_value"]))
    add(f"best_evalue_{partner}", float(best["partner_value"]), "E", "measured",
        f"nb 241 section 7, {best['alphabet']} k={best['ksize']}")

chance = find_table("241", "alphabet", "chance_drift", "p_match", "penalty")
with open(OUT / "245_chance_score_by_alphabet.tsv", "w") as fh:
    fh.write("alphabet\tpr_match_unrelated\tmismatch_penalty_C\t"
             "expected_score_chance_position\n")
    for r in chance:
        fh.write(f"{r['alphabet']}\t{r['p_match']}\t{r['penalty']}\t{r['chance_drift']}\n")

bits = find_table("241", "alphabet", "classes", "bits_per_position")
for r in bits:
    if r["alphabet"] == "hp_lehninger2":
        add("class_share_entropy_hp_lehninger2", float(r["bits_per_position"]), "bits",
            "measured", "nb 241 cell 1, Shannon entropy of class shares on the human proteome")

# --- notebook 242: combining alphabets ------------------------------------------------
ens = find_table("242", "query", "partner", "metric", "best_single_rank",
                 "ens_rank_by_alphabet")
b = [int(r["ens_rank_by_alphabet"]) for r in ens if r["query"] == "Ced9" and r["partner"] == "BCL2"]
add("combined_rank_BCL2_min", min(b), "rank", "measured", "nb 242 section 1, five metrics")
add("combined_rank_BCL2_max", max(b), "rank", "measured", "nb 242 section 1, five metrics")

# --- kmerseek docs: database constants and the 20-letter value -----------------------
# Read from the explainer's own DB table so the two cannot drift apart.
docs = ROOT.parent / "kmerseek" / "docs" / "karlin_altschul_explainer.html"
page = docs.read_text()
for key in ("scope40", "swissprot", "uniref50", "proteome"):
    m = re.search(rf"{key}:\s*\{{\s*name:\s*'([^']+)',\s*n:\s*([0-9.e]+),\s*K:\s*([0-9.]+)", page)
    name, n, K = m.group(1), float(m.group(2)), float(m.group(3))
    kind = "approximate, K borrowed from Swiss-Prot" if key == "proteome" else "fitted K"
    add(f"n_residues_{key}", n, "residues", "approximate",
        f"kmerseek docs Part 1, DB table ({name})")
    add(f"K_{key}", K, "none", kind, f"kmerseek docs Part 1, DB table ({name})")
add("I_20_blosum45", 0.38, "bits per aligned position", "table value",
    "BLOSUM45 relative entropy (Henikoff and Henikoff 1992), as kmerseek docs Part 2 uses it")

with open(OUT / "245_inputs.tsv", "w") as fh:
    fh.write("quantity\tvalue\tunit\tkind\tsource\n")
    for row in inputs:
        fh.write("\t".join(str(x) for x in row) + "\n")
print(f"{len(inputs)} single numbers written to tables/245_inputs.tsv")
