#!/usr/bin/env python3
"""Build the dark-set summary page from tools/dark-summary-data's JSON.

    python3 summary-page/build.py summary-page/dark_summary_data.json [tools commit]

Puts the JSON, the residue classes of every alphabet and the embedded Fantasque Sans Mono
font (fonts.css, the same block as the BHF explainer page of notebook 279) into
template.html and writes dark_set_summary.html next to it, the file published as the
claude.ai artifact. Every number on the page is computed in the page from the JSON.
Added 2026-10-13.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Residue classes, as in kmerseek src/rust/alphabets.rs (commit d79f863); the HP ones
# also match notebooks/hp_conservation_utils.py ALPHABET_CLUSTERS.
CLASSES = {
    "protein20": list("ACDEFGHIKLMNPQRSTVWY"),
    "uniprot18": ["A", "R", "N", "D", "C", "Q", "EP", "G", "HL", "I", "K", "M", "F", "S",
                  "T", "W", "Y", "V"],
    "hsdm17": ["A", "D", "KE", "R", "N", "T", "S", "Q", "Y", "F", "LIV", "M", "C", "W", "H",
               "G", "P"],
    "wass14": ["WM", "DI", "P", "C", "AV", "K", "T", "RE", "G", "L", "Y", "SH", "F", "NQ"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "dayhoff6": ["C", "AGPST", "DENQ", "FWY", "HKR", "ILMV"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "hp_lehninger_hpc3": ["AFGILMPVWY", "DEHKNQRST", "C"],
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
    "hp_thomas_dill2": ["ACFILMVWY", "DEGHKNPQRST"],
    "hp_kyte_doolittle2": ["ACFILMV", "DEGHKNPQRSTWY"],
    "hp_thomas_dill_no_c2": ["AFILMVWY", "CDEGHKNPQRST"],
    "hp_lehninger_c_nonpolar2": ["ACFGILMPVWY", "DEHKNQRST"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
}


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    # tools/dark-summary-data ran from a `git archive` of this commit on Sherlock (its git
    # 1.8 has no worktrees), so it could not read its own commit; record it here.
    if not data.get("commit"):
        assert len(sys.argv) == 3, "give the commit tools/dark-summary-data ran from"
        data["tools_commit"] = sys.argv[2]
    for sp, rows in data["settings"].items():
        missing = {r["alphabet"] for r in rows} - set(CLASSES)
        assert not missing, f"{sp}: no residue classes for {missing}"
        for a, cl in CLASSES.items():
            assert "".join(sorted("".join(cl))) == "ACDEFGHIKLMNPQRSTVWY", a
    page = (HERE / "template.html").read_text()
    fonts = (HERE / "fonts.css").read_text()
    for marker in ("/*__DATA__*/", "/*__FONTS__*/", "/*__CLASSES__*/"):
        assert page.count(marker) == 1, f"template must hold {marker} once"
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = (page.replace("/*__FONTS__*/", fonts).replace("/*__DATA__*/", blob)
            .replace("/*__CLASSES__*/", json.dumps(CLASSES, separators=(",", ":"))))
    # The style block must survive the font insert (the experiments page lost its CSS once)
    assert page.count("<style>") == 1 and "--bg:" in page and "Fantasque Sans Mono" in page
    out = HERE / "dark_set_summary.html"
    out.write_text(page)
    print(f"wrote {out}: {out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
