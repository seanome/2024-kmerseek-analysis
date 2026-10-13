"""Helpers for Nature-family figures. Pair with nature.mplstyle (same folder).

Copy both files into the repo (2024-kmerseek-analysis: notebooks/) so a figure can be
rebuilt from committed code; do not import them from ~/.claude.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

MM = 1 / 25.4  # inches per millimetre
ONE_COLUMN_MM = 89
TWO_COLUMN_MM = 183
MAX_HEIGHT_MM = 170  # leaves room for the legend on the page

STYLE = Path(__file__).with_name("nature.mplstyle")

# Okabe-Ito, the colour-blind-safe set Nature's figure guide lists.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "reddish_purple": "#CC79A7",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
GREY = "#999999"  # for "everything else" / context marks

#: More distinct colours than this in a PNG means a continuous colour bar, which
#: is left at full colour because quantising it puts visible steps in the bar.
CONTINUOUS_COLOUR_THRESHOLD = 3_000


def use_style():
    plt.style.use(STYLE)


def figure(width_mm=ONE_COLUMN_MM, height_mm=60, **subplots_kw):
    """Figure at its printed size, with constrained layout."""
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm is over Nature's {MAX_HEIGHT_MM} mm")
    subplots_kw.setdefault("layout", "constrained")
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), **subplots_kw)


def panel_label(ax, letter, dx_pt=-18, dy_pt=4):
    """Bold lowercase 8 pt panel letter, a fixed distance up-left of the axes corner."""
    offset = ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        letter,
        transform=ax.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def shared_legend(fig, ncol=None, **kw):
    """One legend above all panels, de-duplicated across axes."""
    handles, labels = {}, []
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    return fig.legend(
        [handles[lab] for lab in labels],
        labels,
        loc="outside upper center",
        ncol=ncol or len(labels),
        **kw,
    )


def save(fig, path_stem, formats=("svg", "png")):
    """One vector format and one raster: SVG for the journal, PNG for everything else.

    PDF was dropped on 2026-10-13: SVG is already a vector format a journal
    accepts, so writing both doubled the repository's figure weight for nothing.
    Pass formats=("pdf",) explicitly if a submission ever asks for PDF.

    Every PNG is written as a 256-colour palette image, the same way
    scripts/shrink_png.py and the pre-commit hook of the same name do it, so a
    figure is already in the committable form when it is saved rather than only
    after a commit is attempted.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out = path_stem.with_suffix(f".{ext}")
        fig.savefig(out)
        if ext == "png":
            to_256_colours(out)


def to_256_colours(path, max_colours=256):
    """Rewrite a PNG in place as a 256-colour palette image, keeping its dpi.

    Public so a PNG saved some other way can be converted with the same call.
    A figure with a continuous colour bar is left alone: 256 colours put visible
    steps in the bar, which is worse than the file size it saves. The bar is
    detected by counting distinct colours, since a continuous map puts far more
    in the image than a categorical figure ever does.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if im.mode == "P":
            return False
        colours = im.getcolors(maxcolors=1 << 24)
        if colours is not None and len(colours) > CONTINUOUS_COLOUR_THRESHOLD:
            return False
        dpi = im.info.get("dpi")
        quantized = im.convert("RGBA").quantize(
            max_colours, method=Image.Quantize.FASTOCTREE
        )
    quantized.save(path, "PNG", optimize=True, **({"dpi": dpi} if dpi else {}))
    return True
