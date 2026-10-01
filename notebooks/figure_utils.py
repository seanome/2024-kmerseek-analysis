"""Shared figure helpers: the TOOLS / hypothesis / conclusion stamp, alphabet axes, and
the mark for a missing value.

Each helper does one thing that a PR review asked for by hand:

* `finish_figure` puts the TOOLS line directly above the plot, measured from where the
  axes and legends really end, so no empty band is left between the TOOLS line and the
  legend (PR #46: "too much whitespace after the TOOLS section and before the legend",
  on every figure of notebook 241).
* `alphabet_positions` and `alphabet_axis` order alphabets by size, leave a gap between
  size groups (2-3 letters, 4-8, 9-19, then protein20), and draw each row's grid line
  from its tick mark (PR #46).
* `MISSING_STYLE` is the mark for "no value" (no E-value fit, no hit): a small dot that
  no data series uses (PR #46: a circle read as another data point).

`mhc_region_utils.finish_figure` and `hp_conservation_utils.finish_figure` both call the
one here.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

#: The mark for a missing value: a small grey dot, not a circle or a hollow data marker.
#: Use with `ax.plot(x, y, **MISSING_STYLE)` or `ax.scatter(x, y, marker=".", ...)`.
MISSING_STYLE: dict = dict(
    marker=".", markersize=4, linestyle="none", color="#9a9a9a", label="no value"
)

#: Alphabet size groups, smallest first: (lowest size, highest size).
ALPHABET_SIZE_GROUPS: tuple[tuple[int, int], ...] = ((2, 3), (4, 8), (9, 19), (20, 20))


def alphabet_size(alphabet: str) -> int:
    """Number of residue classes, read from the trailing digits of the name.

    ``hp_pbotc_1st_ed2`` -> 2, ``gbmr7`` -> 7, ``protein20`` -> 20.
    """
    m = re.search(r"(\d+)$", alphabet)
    if m is None:
        raise ValueError(f"alphabet name {alphabet!r} does not end in its class count")
    return int(m.group(1))


def alphabet_size_group(alphabet: str) -> int:
    """Index into `ALPHABET_SIZE_GROUPS` for this alphabet."""
    size = alphabet_size(alphabet)
    for i, (lo, hi) in enumerate(ALPHABET_SIZE_GROUPS):
        if lo <= size <= hi:
            return i
    raise ValueError(f"alphabet {alphabet!r} has {size} classes, outside every group")


def alphabet_positions(alphabets: Sequence[str], gap: float = 0.8) -> dict[str, float]:
    """Axis position for each alphabet: sorted by size, `gap` extra space between groups.

    Within a group, alphabets keep size order, then name order, so the layout is the
    same in every figure that shows the same alphabets.
    """
    ordered = sorted(set(alphabets), key=lambda a: (alphabet_size(a), a))
    positions: dict[str, float] = {}
    pos = 0.0
    prev_group = None
    for a in ordered:
        group = alphabet_size_group(a)
        if prev_group is not None and group != prev_group:
            pos += gap
        positions[a] = pos
        pos += 1.0
        prev_group = group
    return positions


def alphabet_axis(
    ax,
    alphabets: Sequence[str],
    axis: str = "y",
    gap: float = 0.8,
    grid_color: str = "#e3e3e3",
) -> dict[str, float]:
    """Set one tick per alphabet on `axis`, grouped by size, with grid lines at the ticks.

    Returns the alphabet -> position map to plot against. The grid lines are drawn at
    the tick positions themselves (not offset, not between ticks), so a mark can be read
    straight back to its alphabet. Larger alphabets end up at the top of a y axis.
    """
    positions = alphabet_positions(alphabets, gap=gap)
    ticks = list(positions.values())
    labels = list(positions)
    if axis == "y":
        ax.set_yticks(ticks, labels)
        ax.set_ylim(min(ticks) - 0.6, max(ticks) + 0.6)
        ax.yaxis.grid(True, color=grid_color, linewidth=0.8, zorder=0)
    elif axis == "x":
        ax.set_xticks(ticks, labels, rotation=90)
        ax.set_xlim(min(ticks) - 0.6, max(ticks) + 0.6)
        ax.xaxis.grid(True, color=grid_color, linewidth=0.8, zorder=0)
    else:
        raise ValueError(f"axis must be 'x' or 'y', not {axis!r}")
    ax.set_axisbelow(True)
    return positions


def content_top(fig) -> float:
    """Top edge of everything drawn on `fig` (axes, tick labels, legends, colorbars),
    in figure fraction. Can be above 1 when a legend hangs above the top axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tops = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is not None:
            tops.append(fig.transFigure.inverted().transform((0, bbox.y1))[1])
    extras = list(fig.legends)
    if getattr(fig, "_suptitle", None) is not None:
        extras.append(fig._suptitle)
    for artist in extras:
        bbox = artist.get_window_extent(renderer)
        tops.append(fig.transFigure.inverted().transform((0, bbox.y1))[1])
    return max(tops) if tops else 1.0


def finish_figure(
    fig,
    path,
    tools: str,
    hypothesis: str,
    conclusion: str,
    title: str | None = None,
    *,
    footer_y: float = -0.01,
    header_y: float | None = None,
    layout: bool = True,
    dpi: int = 200,
    wrap: int | None = None,
    header_pad_pt: float = 8.0,
) -> None:
    """Stamp TOOLS / hypothesis / conclusion on `fig`, then save it to `path`.

    ``tools`` names the tools shown. ``title`` stacks above the TOOLS line. The TOOLS
    line is placed `header_pad_pt` points above the measured top of the axes and legends
    (`content_top`), so whatever top margin the layout left is cropped away by
    ``bbox_inches="tight"``. Pass ``header_y`` to place it by hand. ``footer_y`` moves the
    footer down when a legend hangs below the axes. ``layout=False`` skips
    `tight_layout` for figures whose colorbars or gridspec margins it would undo.
    """
    if layout:
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001  (a colorbar layout warning is not a failure)
            pass
    w_in, h_in = fig.get_size_inches()
    wrap = wrap or max(70, int(w_in * 12))

    def line(pt: float) -> float:
        """Height of one text line of size `pt`, in figure fraction."""
        return pt / 72.0 / h_in * 1.45

    y = content_top(fig) + header_pad_pt / 72.0 / h_in if header_y is None else header_y

    is_km = "kmerseek" in tools and not tools.startswith("no kmerseek")
    tool_lines = textwrap.wrap("TOOLS: " + tools, wrap)
    fig.text(
        0.0,
        y,
        "\n".join(tool_lines),
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#8B1A1A" if is_km else "#1F3B73",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FBEAEA" if is_km else "#E8EEF8",
            edgecolor="none",
        ),
    )
    y += line(9.5) * len(tool_lines) + line(9.5) * 0.9
    if title:
        fig.text(
            0.5, y, title, ha="center", va="bottom", fontsize=12.5, fontweight="bold"
        )

    foot = textwrap.wrap("Hypothesis: " + hypothesis, wrap) + textwrap.wrap(
        "Conclusion: " + conclusion, wrap
    )
    fig.text(
        0.0,
        footer_y,
        "\n".join(foot),
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F6F6F6", edgecolor="#DDDDDD"),
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
