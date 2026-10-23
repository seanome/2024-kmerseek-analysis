"""Put existing diagnostic figures beside the tables they explain.

This does not recompute a scientific result or change frozen parameters. Saved outputs
reuse the already generated figure files; a rebuilt notebook displays the same files.
"""

from pathlib import Path
import base64
import json

PANELS = {
    "261": {
        "Which data this execution reads": (
            "260_input_calls_kept_by_arm.png",
            "Retained input calls per arm in the region table used by this combiner.",
        ),
        "Greedy search on the tune split": (
            "261_intersection_precision_recall_by_n_zebrafish_tune.png",
            "Tune precision and recall along the greedy path.",
        ),
        "The frozen panel on the test split": (
            "261_multidomain_domains_landed_intersection_vs_phmmer_zebrafish_test.png",
            "The frozen intersection evaluated on multi-domain test queries; the table above also reports all-query precision and recall.",
        ),
    },
    "244": {
        "Stage 0. Which arm lands on which feature type": (
            "244_stage0_arm_by_feature_type.png",
            "Report-half landing fractions for the arms listed above.",
        ),
        "Top 10, and the cases a scan also finds": (
            "244_candidate_01_ROS1_BINDING_yeast.png",
            "The first ranked candidate, shown as protein features and tool calls; other candidates follow below.",
        ),
        "Verdict: the facts it rests on": (
            "244_candidate_04_COL9A1_DOMAIN_chicken.png",
            "The COL9A1 chicken example discussed in the verdict, with feature and call boundaries.",
        ),
    },
    "260": {
        "Which search outputs this execution reads": (
            "260_input_calls_kept_by_arm.png",
            "Calls retained by the E-value cut for each input file; the protein20 fly file retains zero.",
        ),
        "Merging calls into regions": (
            "260_PLG_tracks.png",
            "PLG illustrates the individual calls and merged boundaries; this example does not represent every query.",
        ),
        "Truth": (
            "260_PLG_tracks.png",
            "The PLG feature track shows the Pfam truth intervals used to assess landing; denominators include all eligible features, not just PLG.",
        ),
    },
    "263": {
        "7. Known cases: where Ced9-BCL2, P66-CD47 and BHF fall in the vote ranking": (
            "263_vote_rank_known_cases_ced9_bcl2_p66_cd47_bhf_human_proteome_notebook.png",
            "Frozen-vote rankings for the known-case searches; the sequence blocks below explain selected calls.",
        )
    },
    "264": {
        "One scoring function for every row": (
            "264_calls_domains_boundary_per_method_midi_plus_zebrafish_test_notebook.png",
            "The common scoring function produces these call counts, landed-domain counts and boundary errors; numeric cross-checks are above.",
        )
    },
}


def apply_to_path(path, saved_outputs=False):
    """Append displays to existing analysis cells; retain original execution history.

    Saved displays reuse existing PNGs. This does not claim a fresh-kernel rerun.
    """
    path = Path(path)
    panels = PANELS.get(path.name.split("_")[0], {})
    nb = json.loads(path.read_text())
    cells = [c for c in nb["cells"] if not c.get("metadata", {}).get("review_figure")]
    for c in cells:
        if c["cell_type"] == "code":
            source = "".join(c["source"]).split("\n# Review figure beside table")[0]
            c["source"] = source.splitlines(True)
            c["outputs"] = [
                o
                for o in c.get("outputs", [])
                if not o.get("metadata", {}).get("review_figure")
            ]
    heading = None
    last_code = None

    def finish():
        if heading not in panels or last_code is None:
            return
        filename, caption = panels[heading]
        source = (
            "".join(last_code["source"]).rstrip()
            + "\n\n# Review figure beside table: "
            + caption
            + "\n"
        )
        if filename == "260_input_calls_kept_by_arm.png":
            source += "from combiner_review_figures import make_input_figure\nmake_input_figure()\n"
        source += (
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            f'display(Image(filename=str(Path.cwd().parent / "figures" / {filename!r})))\n'
        )
        last_code["source"] = source.splitlines(True)
        if saved_outputs:
            image = path.parent.parent / "figures" / filename
            last_code.setdefault("outputs", []).append(
                {
                    "output_type": "display_data",
                    "metadata": {"review_figure": True},
                    "data": {
                        "image/png": base64.b64encode(image.read_bytes()).decode()
                    },
                }
            )

    for cell in cells:
        if cell["cell_type"] == "markdown":
            first = "".join(cell["source"]).splitlines()[0] if cell["source"] else ""
            if first.startswith("##"):
                finish()
                heading = first.lstrip("# ").strip()
                last_code = None
        elif cell["cell_type"] == "code":
            last_code = cell
    finish()
    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=1) + "\n")


def make_input_figure():
    """Plot the small committed-run reduction summary, without reading raw calls."""
    import polars as pl
    import pubfig as pf

    pf.use_style()
    x = pl.read_parquet(
        Path.home()
        / "data/qfo-pfam-region-midi-plus-0.4/260_region_table/reduced/reduce_summary.parquet"
    ).sort("species", "arm")
    fig, ax = pf.figure(183, 70)
    labels = [r["species"] + " " + r["arm"] for r in x.iter_rows(named=True)]
    ax.barh(labels, x["n_kept"], color="#666666")
    ax.invert_yaxis()
    ax.set_xlabel("Region calls retained at E-value < 10")
    ax.set_title("Input coverage: five files, four with retained calls")
    for i, n in enumerate(x["n_kept"]):
        ax.text(n, i, f" {n:,}", va="center")
    pf.save(
        fig, Path(__file__).resolve().parents[1] / "figures/260_input_calls_kept_by_arm"
    )
