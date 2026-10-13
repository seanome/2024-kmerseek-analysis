"""Fill the notebook 284 explainer template with its data and the embedded mono font."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "scripts/experiments_page_fonts.css"
page = (ROOT / "notebooks/284_alphabet_agreement_explainer.template.html").read_text()
page = page.replace("__FONTS__", FONTS.read_text()).replace(
    "__DATA__", (ROOT / "notebooks/284_alphabet_agreement_data.json").read_text())
(ROOT / "notebooks/284_alphabet_agreement_explainer.html").write_text(page)
