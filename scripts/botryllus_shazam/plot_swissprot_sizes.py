"""Actual full Swiss-Prot FASTA versus per-alphabet sketch ZIP storage."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PIL import Image

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
d = json.loads(source.read_text())
settings = {(r["alphabet"], r["scaled"]): r for r in d["settings"]}
assert len(settings) == 9 and d["reference_records"] == 575748
assert {(r["alphabet"], r["ksize_aa"]) for r in d["settings"]} == {
    ("protein", 7),
    ("dayhoff", 12),
    ("hp", 30),
}
raw = d["raw_fasta_bytes"] / 1e6
compressed = d["download_fasta_gz_bytes"] / 1e6
colours = {"protein": "#0072B2", "dayhoff": "#D55E00", "hp": "#009E73"}
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 100, nrows=1, ncols=3, sharey=True)
legend = [
    Line2D([], [], color=colours[m], marker="o", label=label)
    for m, label in [
        ("protein", "Protein k7"),
        ("dayhoff", "Dayhoff k12"),
        ("hp", "HP k30"),
    ]
]
legend += [
    Line2D(
        [], [], color="#555555", linestyle="--", label="Original FASTA (uncompressed)"
    ),
    Line2D(
        [], [], color="#999999", linestyle=":", label="Original FASTA.gz (download)"
    ),
]
fig.legend(
    handles=legend,
    loc="outside upper center",
    ncol=3,
    frameon=False,
    title="Swiss-Prot · 575,748 proteins · sourmash 4.9.4 / Branchwater 0.9.15",
)
max_mb = max(raw, max(r["bytes"] / 1e6 for r in settings.values()))
for ax, (mode, k) in zip(axs, [("protein", 7), ("dayhoff", 12), ("hp", 30)]):
    values = [settings[mode, s]["bytes"] / 1e6 for s in [5, 10, 100]]
    ax.axhline(raw, color="#555555", linestyle="--", zorder=1)
    ax.axhline(compressed, color="#999999", linestyle=":", zorder=1)
    ax.plot([5, 10, 100], values, marker="o", color=colours[mode], zorder=2)
    # Positions are the actual numeric sampling factors; ratio spacing separates 5 and 10.
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 100], ["5", "10", "100"])
    ax.set_xlim(3.5, 145)
    ax.minorticks_off()
    ax.set_xlabel("Scaled sampling factor")
    ax.set_ylim(0, max_mb * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    for scale, value in zip([5, 10, 100], values):
        ax.annotate(
            f"{value:,.1f}",
            (scale, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.text(
        0.5,
        0.96,
        {"protein": "Protein k7", "dayhoff": "Dayhoff k12", "hp": "HP k30"}[mode],
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
axs[0].set_ylabel("Stored file size (MB; 1 MB = 1,000,000 bytes)")
pf.save(fig, dest)
image = dest.with_suffix(".png")
with Image.open(image) as im:
    im.convert("RGB").quantize(colors=256).save(image)
with dest.with_suffix(".tsv").open("w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["file_type", "alphabet", "scaled", "bytes", "MB"])
    w.writerow(["original FASTA", "all", "", d["raw_fasta_bytes"], raw])
    w.writerow(
        ["original FASTA.gz", "all", "", d["download_fasta_gz_bytes"], compressed]
    )
    for mode in ["protein", "dayhoff", "hp"]:
        for scale in [5, 10, 100]:
            r = settings[mode, scale]
            w.writerow(["SIG.zip", mode, scale, r["bytes"], r["bytes"] / 1e6])
print(
    json.dumps(
        {
            "raw_fasta_MB": raw,
            "download_fasta_gz_MB": compressed,
            "zip_MB": {
                m: [settings[m, s]["bytes"] / 1e6 for s in [5, 10, 100]]
                for m in colours
            },
        },
        indent=2,
    )
)
