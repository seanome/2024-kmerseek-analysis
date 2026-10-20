"""Figures for experiment 354 from a finished WORKDIR (peptide_source.py all).

    make_figures.py WORKDIR

a: recovery against false-assignment rate on evaluation calls, as the reporting threshold
   moves. b: proteins each search returns per call, before any threshold. c: time for one
   warm batch of every case.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

import peptide_source as ps  # noqa: E402

# One colour per search; dashed = reduced-alphabet score without verification.
COLOUR = {
    "exact": pf.OKABE_ITO["blue"],
    "align": pf.OKABE_ITO["vermillion"],
    "exact_then_align": pf.OKABE_ITO["bluish_green"],
    "reduced_massspec18_k5": pf.OKABE_ITO["reddish_purple"],
    "reduced_sdm12_k6": pf.OKABE_ITO["orange"],
    "reduced_dayhoff6_k8": pf.OKABE_ITO["sky_blue"],
}
LABEL = {
    "exact": "exact, 20 letters",
    "align": "alignment, 20 letters",
    "exact_then_align": "exact, then alignment",
    "reduced_massspec18_k5": "massspec18 k=5",
    "reduced_sdm12_k6": "sdm12 k=6",
    "reduced_dayhoff6_k8": "dayhoff6 k=8",
}


def base(method):
    return method.rsplit("_", 1)[0] if method.endswith(("_alone", "_verified")) else method


def main(workdir: Path):
    pf.use_style()
    res = workdir / "results"
    curves = pd.read_csv(res / "recall_vs_false_assignment.tsv", sep="\t")
    runtime = pd.read_csv(res / "runtime.tsv", sep="\t").set_index("method")
    cases = ps.read_cases(workdir)
    ev = cases.query("side == 'evaluation' and kind != 'source_absent'")
    ev = ev[ev["kind"] == "source_present"]["case_id"]

    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3)
    ax = axs[0]
    for m in ps.scored_methods():
        c = curves[curves["method"] == m].sort_values("false_assignment_rate")
        ax.plot(100 * c["false_assignment_rate"], 100 * c["recovery"], color=COLOUR[base(m)],
                ls="--" if m.endswith("_alone") else "-", label=LABEL[base(m)] if not m.endswith("_alone") else None,
                drawstyle="steps-post", zorder=3 if m == "exact" else 2)
    ax.plot([], [], color=pf.GREY, ls="--", label="reduced, not verified")
    ax.axvline(1, color=pf.GREY, lw=0.5)
    ax.set_xlabel("Source-absent calls assigned a protein (%)")
    ax.set_ylabel("Calls with a source in top 5 (%)")
    ax.set_xlim(-2, 100)  # a line at 0% stays off the axis
    ax.set_ylim(0, 100)

    ax = axs[1]
    data, names = [], []
    for m in ps.SEARCH_METHODS:
        cand = pd.read_csv(workdir / "candidates" / f"{m}.tsv.gz", sep="\t")
        n = cand[cand["case_id"].isin(ev)].groupby("case_id").size().reindex(ev, fill_value=0)
        data.append(np.maximum(n.to_numpy(), 0.5))
        names.append(LABEL[m])
    parts = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color="black"))
    for patch, m in zip(parts["boxes"], ps.SEARCH_METHODS):
        patch.set_facecolor(COLOUR[m])
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.set_xscale("log")
    ax.set_xlabel("Proteins returned per call (0 drawn at 0.5)")
    ax.invert_yaxis()

    ax = axs[2]
    meths = ps.scored_methods()
    y = np.arange(len(meths))
    for yi, m in zip(y, meths):
        if m.endswith("_alone"):  # outline only, dashed, as the dashed lines in a
            ax.barh(yi, runtime.loc[m, "warm_batch_wall_seconds"], color="white",
                    edgecolor=COLOUR[base(m)], ls="--")
        else:
            ax.barh(yi, runtime.loc[m, "warm_batch_wall_seconds"], color=COLOUR[base(m)])
    ax.set_yticks(y, [LABEL[base(m)] + (" (not verified)" if m.endswith("_alone") else "") for m in meths])
    ax.set_xscale("log")
    n_q = int(runtime["n_queries"].iloc[0])
    ax.set_xlabel(f"Warm batch of {n_q} calls (s)")
    ax.invert_yaxis()

    for a in axs[1:]:
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    pf.shared_legend(fig)
    for a, letter in zip(axs, "abc"):
        pf.panel_label(a, letter, dx_pt=-28 if letter == "a" else -18)
    out = workdir / "results" / "figures"
    out.mkdir(exist_ok=True)
    pf.save(fig, out / "354_recovery_candidates_runtime")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
