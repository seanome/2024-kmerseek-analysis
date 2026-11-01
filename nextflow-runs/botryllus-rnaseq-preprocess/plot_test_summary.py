"""Figure for the 2-sample test: read length before and after trimming, and which strand
read 1 comes from (Salmon, 1 million read subsample; here all 200_000 pairs).

    python plot_test_summary.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

pf.use_style()
df = pd.read_csv(HERE / "test_results" / "test_summary.tsv", sep="\t")
df["label"] = df["sample"].str.replace("_", " ") + "\nread " + df["read"].astype(str)

fig, (ax_a, ax_b) = pf.figure(pf.ONE_COLUMN_MM, 55, ncols=2)

# a: mean read length, raw vs trimmed
y = range(len(df))
ax_a.barh([i - 0.2 for i in y], df["mean_length_raw"], height=0.4,
          color=pf.GREY, label="Raw reads")
ax_a.barh([i + 0.2 for i in y], df["mean_length_trimmed"], height=0.4,
          color=pf.OKABE_ITO["blue"], label="After Trim Galore")
ax_a.set_yticks(list(y), df["label"])
ax_a.invert_yaxis()
ax_a.set_xlabel("Mean read length (bases)")
ax_a.set_xlim(0, 160)
ax_a.set_xticks([0, 50, 100, 150])

# b: share of Salmon-assigned fragments whose read 1 is on each strand of the transcript
strand = df.drop_duplicates("sample").reset_index(drop=True)
total = strand[["salmon_ISF", "salmon_ISR", "salmon_SF", "salmon_SR"]].sum(axis=1)
same = 100 * (strand["salmon_ISF"] + strand["salmon_SF"]) / total
ys = range(len(strand))
ax_b.barh(list(ys), same, color=pf.OKABE_ITO["vermillion"],
          label="Read 1 on transcript's strand")
ax_b.barh(list(ys), 100 - same, left=same, color=pf.OKABE_ITO["sky_blue"],
          label="Read 1 on opposite strand")
ax_b.axvline(50, color="black", linewidth=0.5, linestyle="--", label="50%: unstranded library")
ax_b.set_yticks(list(ys), strand["sample"].str.replace("_", " "))
ax_b.invert_yaxis()
ax_b.set_xlabel("Fragments (%)")
ax_b.set_xlim(0, 100)

pf.shared_legend(fig, ncol=2)
for ax, letter in zip((ax_a, ax_b), "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, str(HERE / "test_results" / "test_read_length_and_strand"))
