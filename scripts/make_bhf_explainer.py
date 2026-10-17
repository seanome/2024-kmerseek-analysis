#!/usr/bin/env python3
"""Build the BHF explainer page from notebook 279's data.

    python3 scripts/make_bhf_explainer.py

Reads tables/279_bhf_explainer_data.json (written by notebook 279, section 6g), the page
template scripts/279_bhf_explainer_template.html and the embedded Fantasque Sans Mono font
block scripts/experiments_page_fonts.css, and writes figures/279_bhf_explainer.html, the
file published as the claude.ai artifact. Every number on the page comes from the JSON.
Added 2026-10-13 for PR 120.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "tables/279_bhf_explainer_data.json").read_text())
    page = (ROOT / "scripts/279_bhf_explainer_template.html").read_text()
    fonts = (ROOT / "scripts/experiments_page_fonts.css").read_text()
    for marker in ("/*__DATA__*/", "/*__FONTS__*/"):
        assert page.count(marker) == 1, f"template must hold {marker} once"
    # </ inside the JSON would end the script element early
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("/*__FONTS__*/", fonts).replace("/*__DATA__*/", blob)
    assert "Fantasque Sans Mono" in page and '"vpx_regions"' in page
    out = ROOT / "figures/279_bhf_explainer.html"
    out.write_text(page)
    print(f"wrote {out}: {out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
