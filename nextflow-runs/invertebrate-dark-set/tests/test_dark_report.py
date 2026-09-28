"""The dark-set report: the first screen, the species name, the effect-size direction,
the controls beside the kmerseek reach, and the one protein followed through."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures  # noqa: E402

BIN = Path(__file__).resolve().parent.parent / "bin"
SHARED = Path(__file__).resolve().parents[2] / "shared"


def build(tmp: Path, *, registry: bool = True, **kw) -> dict:
    f = fixtures.run_dir(tmp / "run", **kw)
    out = tmp / "multiqc_in"
    cmd = [sys.executable, str(BIN / "build_dark_multiqc_inputs.py"),
           "--species", "botryllus", "--clade", "Ascidiacea",
           "--dark-summary", str(f["dark_summary"]),
           "--reference-summary", str(f["reference_summary"]),
           "--run-params", str(f["run_params"]),
           "--extra-dir", str(f["extra_dir"]), "--outdir", str(out)]
    if registry:
        cmd += ["--registry", str(f["registry"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   env={"PYTHONPATH": str(SHARED), "PATH": "/usr/bin:/bin"})
    sections = {p.name.removesuffix("_mqc.json"): json.loads(p.read_text())
                for p in out.glob("*_mqc.json")}
    sections["_header"] = (out / "report_header.yaml").read_text()
    return sections


def test_the_title_and_the_prose_name_the_species(tmp_path):
    s = build(tmp_path)
    h = s["_header"]
    assert h.startswith('title: "Botryllus schlosseri dark set"')
    assert "subtitle: \"The <i>Botryllus schlosseri</i> proteins that phmmer" in h
    assert "<i>Botryllus schlosseri</i>" in s["dark_headline"]["description"]
    assert "<i>Botryllus schlosseri</i>" in s["dark_overview"]["description"]
    # Slugs stay in plot titles.
    assert s["dark_headline"]["pconfig"]["title"].startswith("botryllus:")


def test_without_a_registry_the_label_is_used(tmp_path):
    s = build(tmp_path, registry=False)
    assert s["_header"].startswith('title: "botryllus dark set"')
    # The focus proteins travel in the dark summary, so the section is still built.
    assert "dark_focus" in s


def test_the_first_screen_carries_the_answer_and_the_caveat(tmp_path):
    h = build(tmp_path)["_header"]
    assert "intro_text: " in h
    assert "45.1% of the proteome is dark: 20_448 of 45_339 proteins" in h
    assert "Counting only proteins of at least 100 residues it is" in h
    assert "shorter than placed ones (median 151 against 377 residues)" in h
    assert "more disordered (median metapredict score 0.367 against 0.206" in h
    assert "That is reach, not accuracy" in h
    assert "too permissive to be informative on its own" in h
    assert "reaches 100.0% of the placed proteins" in h
    assert "residues shuffled it reaches 100.0%" in h
    assert "Why the target has Ascidiacea removed" in h
    assert "itself has no reviewed Swiss-Prot entry" in h
    assert "BHF, the Botryllus histocompatibility factor" in h
    assert "is dark: none of phmmer, jackhmmer and mmseqs2 places it" in h
    assert "at residues 812-1004" in h


def test_the_effect_size_states_the_direction_the_data_shows(tmp_path):
    d = build(tmp_path)["dark_length"]["description"]
    # P(dark longer) = 0.195, so the sentence says shorter with 0.805.
    assert "randomly drawn dark protein is shorter than a randomly drawn placed one is 0.805" in d
    assert "rank-biserial correlation, the same comparison on a -1 to 1 scale, is -0.610" in d
    assert "below the smallest number the software prints (1e-300)" in d
    assert "p=0" not in d
    # No proteins under 50 aa in either group: that line is not printed.
    assert "Under 50 aa" not in d
    assert "Under 100 aa</b>: 18.0% of dark against 2.0% of placed" in d
    assert "genuinely" not in d
    dis = build(tmp_path / "b")["dark_disorder"]["description"]
    assert "dark protein is more disordered than a randomly drawn placed one is 0.710" in dis


def test_the_by_length_panel_names_no_species_that_is_not_on_the_page(tmp_path):
    d = build(tmp_path)["dark_headline_by_length"]["description"]
    for other in ("ciona", "mouse", "worm", "48%"):
        assert other not in d


def test_the_mask_panel_carries_the_placed_control_the_null_and_the_cutoffs(tmp_path):
    s = build(tmp_path)
    plot = s["dark_kmerseek_mask"]
    labels = [d["name"] for d in plot["pconfig"]["data_labels"]]
    assert labels == ["dark set", "placed set (control)", "shuffled dark proteins (null)"]
    assert len(plot["data"]) == 3
    assert plot["data"][0]["hp_thomas_dill2 k23"]["mask_on"] == 1.0
    assert plot["data"][1]["protein20 k10"]["mask_on"] > 0.8
    d = plot["description"]
    assert "A setting that reaches every dark protein is reaching everything" in d
    assert "The placed set is the control" in d
    assert "The shuffled dark proteins are the null" in d
    assert "Reach at stricter cutoffs</b> is in the table below" in d
    assert "Checking the label needs a structural answer key" in d
    table = s["dark_kmerseek_mask_table"]
    row = table["data"]["hp_thomas_dill2 k23"]
    assert row["fraction_placed_mask_on"] == 1.0
    assert round(row["fraction_dark_mask_on_at_10.0"], 3) == round(610 / 20_448, 3)
    assert row["fraction_shuffled_mask_on"] == 1.0
    assert "fraction_dark_mask_on_at_10.0" in table["headers"]


def test_a_run_without_the_controls_says_so(tmp_path):
    s = build(tmp_path, scores=False, shuffled=False)
    plot = s["dark_kmerseek_mask"]
    assert [d["name"] for d in plot["pconfig"]["data_labels"]] == ["dark set", "placed set (control)"]
    d = plot["description"]
    assert "No shuffled-sequence control in this run" in d
    assert "Reach at a stricter cutoff is not in this run" in d
    assert "on shuffled sequences, which is not in this run" in d
    assert "fraction_shuffled_mask_on" not in s["dark_kmerseek_mask_table"]["headers"]
    assert "shuffled-sequence control" in s["_header"]


def test_one_protein_followed_through(tmp_path):
    sec = build(tmp_path)["dark_focus"]
    body = sec["data"]
    assert "BHF, the Botryllus histocompatibility factor" in body
    assert "<td>jackhmmer</td><td>0.011</td><td>not placed</td>" in body
    assert "<td>mmseqs2</td><td>no hit</td>" in body
    assert "1366 residues, longer than" in body and "800-1600" in body
    assert "0.512, more disordered than" in body and "0.5-0.6" in body
    assert "<td>hp_thomas_dill2 k23</td><td>on</td><td>yes</td><td>4.70</td><td>812-1004</td><td>yes</td>" in body
    assert "<td>protein20 k10</td><td>on</td><td>no</td><td>-</td><td>-</td><td>no</td>" in body
    # The protein as a line with the region as a box, legend first.
    assert "<svg" in body and "the protein, 1366 residues" in body
    assert "best kmerseek region with the mask on (hp_thomas_dill2 k23)" in body
