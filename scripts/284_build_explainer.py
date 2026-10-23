"""Fill the notebook 284 explainer template with its data, its sums code and the embedded mono font.

The page reads (all made by committed scripts):
  notebooks/284_alphabet_agreement_data.json      scripts/284_alphabet_agreement.py
  notebooks/284_agreement_by_evalue_cut.csv       scripts/284_agreement_by_evalue_cut.py
  notebooks/284_pfam998_*.csv                     scripts/284_pfam998_agreement.py
  notebooks/284_explainer_manifest.json           scripts/284_explainer_data.py
  notebooks/284_explainer_removal_gap.json        scripts/tests/284_explainer_compute_check.mjs
  notebooks/284_ced9_bcl2_family_hits.csv         scripts/284_ced9_bcl2_family_hits.py
  notebooks/284_bcl2_best_ranks.csv               scripts/284_bcl2_best_ranks.py
and fetches the tables in notebooks/284_explainer_files/ (scripts/284_explainer_data.py), which
are published next to the page. The shuffled-query control is put in when its CSVs exist.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
FONTS = ROOT / "scripts/experiments_page_fonts.css"
SHUFFLE = Path.home() / "data/botryllus/alphabet-ranking-three-cases/284"


def rows(path):
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            out.append({k: (float(v) if v.replace(".", "", 1).replace("e-", "", 1).isdigit() else v) for k, v in r.items()})
        return out


man = json.loads((NB / "284_explainer_manifest.json").read_text())
data = {
    "nb": json.loads((NB / "284_alphabet_agreement_data.json").read_text()),
    "man": man,
    "evcut": rows(NB / "284_agreement_by_evalue_cut.csv"),
    "pfam_prec": rows(NB / "284_pfam998_precision_by_agreement.csv"),
    "pfam_ap": rows(NB / "284_pfam998_average_precision.csv"),
    "removal_gap": json.loads((NB / "284_explainer_removal_gap.json").read_text()),
    "family": rows(NB / "284_ced9_bcl2_family_hits.csv"),
    "best_ranks": rows(NB / "284_bcl2_best_ranks.csv"),
    "shuffle": None,
}
shuffle_files = [SHUFFLE / f"284_shuffle_agreement_{x}.csv" for x in ("partner", "proteome")]
if all(f.exists() for f in shuffle_files):
    data["shuffle"] = {f.stem: rows(f) for f in shuffle_files}
    print("shuffled-query CSVs found; the page does not draw them yet: add them to the template")
page = (NB / "284_alphabet_agreement_explainer.template.html").read_text()
page = (page.replace("__FONTS__", FONTS.read_text())
        .replace("__COMPUTE__", (NB / "284_explainer_compute.js").read_text())
        .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
(NB / "284_alphabet_agreement_explainer.html").write_text(page)
print("page", len(page.encode()), "bytes")
