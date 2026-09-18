"""Data-flow diagrams for the MultiQC report overviews, as inline SVG.

Every MultiQC report in this repository opens with a section that says what was done, why,
and how the data flowed from the inputs to the panels. The picture half of that is drawn
here: boxes for sets of sequences or results, each with this run's count, and arrows for
the steps between them, labelled with the tool that ran the step.

One module, so the reports draw the same way. A builder script stages this file next to
itself (Nextflow: `path 'flow_diagram.py'` plus `PYTHONPATH=$PWD`) or, run by hand from a
pipeline's bin/, finds it at ../../shared relative to its own path.

The drawing is also clickable. A box drawn with `node=` is wrapped in a <g class="node"
data-id=...>, and every line or label drawn with `frm=`/`to=` carries data-from / data-to
(space-separated ids). `interactive()` wraps the SVG with a details panel, an optional
control, and the script that uses those attributes: hover fades everything that is not
upstream of the box under the pointer, click opens the box's details with links to the
report sections that show it, and the control swaps the counts, share bars, facts and
links for one setting (a length cut, a disorder bin, an answer key).

Drawing rules, from the clear-figures checklist:
  one mark per meaning     a colour or line style means one thing in the whole picture, and
                           the same thing it means in the panels below the picture
  legend before marks      the legend is the first row, so every mark is explained before
                           it is seen
  a count on every box     a box without a number is decoration
  words, not colour, for   a category the words already carry (one pass vs 3 iterations)
  detail                   does not also get a hue; three hues plus a neutral is the budget

Text is 12 px at about 7 px per character; box widths are chosen so the longest line fits
and `wrap` folds anything longer rather than letting it leave the canvas. Outlines and text
are `currentColor`, so the drawing follows MultiQC's light and dark themes.
"""

import json
import math
import textwrap

SVG_W = 780
SVG_FONT_PX = 12
SVG_CHAR_PX = 7
SVG_LINE_PX = 16
SVG_PAD = 8
ICON_PX = 28

# Green is "something was found" and grey is "nothing was found", in every report that uses
# these. Purple marks the query's own clade where a report removes it from the target.
C_FOUND = "#0f9d76"
C_NONE = "#7f7f7f"
C_CLADE = "#7b4fb3"
# The box whose details are open. Not a data colour: it marks a state of the page and
# appears nowhere else in any report.
C_OPEN = "#d97b00"

# Google Material Symbols (Apache 2.0), outlined, 24 px, fetched from
# fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/<name>/default/24px.svg
# and inlined, because the reports are rendered on compute nodes with no internet and have
# to stay one self-contained file each. viewBox is 0 -960 960 960 for every one.
#
#   genetics     query sequences          search      a search tool
#   database     a target database        table_rows  a table of hits or calls
#   task_alt     found / placed           search_off  nothing found
#   straighten   protein length           waves       disorder
#   summarize    the report itself
ICONS = {
    "database": "M480-120q-151 0-255.5-46.5T120-280v-400q0-66 105.5-113T480-840q149 0 254.5 47T840-680v400q0 67-104.5 113.5T480-120Zm0-479q89 0 179-25.5T760-679q-11-29-100.5-55T480-760q-91 0-178.5 25.5T200-679q14 30 101.5 55T480-599Zm0 199q42 0 81-4t74.5-11.5q35.5-7.5 67-18.5t57.5-25v-120q-26 14-57.5 25t-67 18.5Q600-528 561-524t-81 4q-42 0-82-4t-75.5-11.5Q287-543 256-554t-56-25v120q25 14 56 25t66.5 18.5Q358-408 398-404t82 4Zm0 200q46 0 93.5-7t87.5-18.5q40-11.5 67-26t32-29.5v-98q-26 14-57.5 25t-67 18.5Q600-328 561-324t-81 4q-42 0-82-4t-75.5-11.5Q287-343 256-354t-56-25v99q5 15 31.5 29t66.5 25.5q40 11.5 88 18.5t94 7Z",
    "genetics": "M200-40v-40q0-139 58-225.5T418-480q-102-88-160-174.5T200-880v-40h80v40q0 11 .5 20.5T282-840h396q1-10 1.5-19.5t.5-20.5v-40h80v40q0 139-58 225.5T542-480q102 88 160 174.5T760-80v40h-80v-40q0-11-.5-20.5T678-120H282q-1 10-1.5 19.5T280-80v40h-80Zm138-640h284q13-19 22.5-38t17.5-42H298q8 22 17.5 41.5T338-680Zm142 148q20-17 39-34t36-34H405q17 17 36 34t39 34Zm-75 172h150q-17-17-36-34t-39-34q-20 17-39 34t-36 34ZM298-200h364q-8-22-17.5-41.5T622-280H338q-13 19-22.5 38T298-200Z",
    "search": "M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z",
    "search_off": "M138.5-138.5Q80-197 80-280t58.5-141.5Q197-480 280-480t141.5 58.5Q480-363 480-280t-58.5 141.5Q363-80 280-80t-141.5-58.5ZM824-120 568-376q-12-13-25.5-26.5T516-428q38-24 61-64t23-88q0-75-52.5-127.5T420-760q-75 0-127.5 52.5T240-580q0 6 .5 11.5T242-557q-18 2-39.5 8T164-535q-2-11-3-22t-1-23q0-109 75.5-184.5T420-840q109 0 184.5 75.5T680-580q0 43-13.5 81.5T629-428l251 252-56 56Zm-615-61 71-71 70 71 29-28-71-71 71-71-28-28-71 71-71-71-28 28 71 71-71 71 28 28Z",
    "straighten": "M160-240q-33 0-56.5-23.5T80-320v-320q0-33 23.5-56.5T160-720h640q33 0 56.5 23.5T880-640v320q0 33-23.5 56.5T800-240H160Zm0-80h640v-320H680v160h-80v-160h-80v160h-80v-160h-80v160h-80v-160H160v320Zm120-160h80-80Zm160 0h80-80Zm160 0h80-80Zm-120 0Z",
    "summarize": "M348.5-611.5Q360-623 360-640t-11.5-28.5Q337-680 320-680t-28.5 11.5Q280-657 280-640t11.5 28.5Q303-600 320-600t28.5-11.5Zm0 160Q360-463 360-480t-11.5-28.5Q337-520 320-520t-28.5 11.5Q280-497 280-480t11.5 28.5Q303-440 320-440t28.5-11.5Zm0 160Q360-303 360-320t-11.5-28.5Q337-360 320-360t-28.5 11.5Q280-337 280-320t11.5 28.5Q303-280 320-280t28.5-11.5ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h440l200 200v440q0 33-23.5 56.5T760-120H200Zm0-80h560v-400H600v-160H200v560Zm0-560v160-160 560-560Z",
    "table_rows": "M760-200v-120H200v120h560Zm0-200v-160H200v160h560Zm0-240v-120H200v120h560ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Z",
    "task_alt": "M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q65 0 123 19t107 53l-58 59q-38-24-81-37.5T480-800q-133 0-226.5 93.5T160-480q0 133 93.5 226.5T480-160q133 0 226.5-93.5T800-480q0-18-2-36t-6-35l65-65q11 32 17 66t6 70q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm-56-216L254-466l56-56 114 114 400-401 56 56-456 457Z",
    "waves": "M80-146v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62.5-8.5q37.5 0 62.5 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-174.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-164 544.5-155t-64.5 9q-39 0-64.5-9t-46-19.5Q349-185 329-193.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-164 143.5-155T80-146Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-352.5q-21-10.5-41-19t-49-8.5q-29 0-49.5 8.5t-41 19Q569-342 544-333t-64 9q-39 0-64.5-9t-46-19.5Q349-363 329-371.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-342 143.5-333T80-324Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-530.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-520 544.5-511t-64.5 9q-39 0-64.5-9t-46-19.5Q349-541 329-549.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-520 143.5-511T80-502Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-708.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-698 544.5-689t-64.5 9q-39 0-64.5-9t-46-19.5Q349-719 329-727.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-698 143.5-689T80-680Z",
}


def num(value) -> str:
    """Integers with underscore groups: 20_448, never 20,448."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}".replace(",", "_")
    return f"{int(value):,}".replace(",", "_")


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.{digits}f}%"


def wrap(text: str, width_px: int) -> list[str]:
    """Fold `text` so every line fits in a box `width_px` wide, padding included."""
    return textwrap.wrap(text, max(8, (width_px - 2 * SVG_PAD) // SVG_CHAR_PX))


class Flow:
    """A top-to-bottom data-flow diagram as inline SVG.

    Coordinates are in a fixed SVG_W-wide frame; the height grows with what is drawn.
    Methods return the geometry they drew so the next row can be placed against it.
    """

    def __init__(self, width: int = SVG_W) -> None:
        self.w = width
        self.parts: list[str] = []
        self.bottom = 0

    @staticmethod
    def _edge_attrs(frm, to) -> str:
        if not frm and not to:
            return ""
        f = " ".join(frm) if isinstance(frm, (list, tuple, set)) else (frm or "")
        t = " ".join(to) if isinstance(to, (list, tuple, set)) else (to or "")
        return f' data-from="{f}" data-to="{t}"'

    # -- primitives -----------------------------------------------------------------

    def icon(self, name: str, x: float, y: float, *, fill: str = "currentColor",
             px: int = ICON_PX) -> None:
        self.parts.append(
            f'<svg x="{x}" y="{y}" width="{px}" height="{px}" viewBox="0 -960 960 960">'
            f'<path d="{ICONS[name]}" fill="{fill}"/></svg>')

    def box(self, x: int, y: int, w: int, lines: list[str], *, fill: str | None = None,
            stroke: str = "currentColor", stroke_w: float = 1.2, dashed: bool = False,
            text_fill: str = "currentColor", bold_first: bool = False,
            icon: str | None = None, node: str | None = None,
            count_line: int | None = None, share: bool = False) -> tuple[int, int, int, int]:
        """A rounded box with centred text lines and an optional icon at its left edge.

        Returns (x, y, w, h). The icon is vertically centred; the text is centred on what
        is left of the box so the two never overlap. `node` makes it clickable;
        `count_line` is the index of the line a control may rewrite; `share` adds the pale
        bar along the bottom whose width a control sets to the box's share of a total.
        """
        h = max(SVG_LINE_PX * len(lines) + 2 * SVG_PAD, ICON_PX + 2 * SVG_PAD if icon else 0)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        if node:
            self.parts.append(f'<g class="node" data-id="{node}" tabindex="0" role="button">')
        self.parts.append(
            f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill or "none"}" stroke="{stroke}" stroke-width="{stroke_w}"{dash}/>')
        if share:
            self.parts.append(
                f'<rect class="share" data-share="{node}" x="{x}" y="{y + h - 6}" width="0" '
                f'height="6" fill="#ffffff" opacity="0.55"/>')
        tx = x + w / 2
        if icon:
            self.icon(icon, x + SVG_PAD, y + (h - ICON_PX) / 2, fill=text_fill)
            tx = x + SVG_PAD + ICON_PX + (w - SVG_PAD - ICON_PX) / 2
        y_text = y + (h - SVG_LINE_PX * len(lines)) / 2
        for i, line in enumerate(lines):
            weight = ' font-weight="bold"' if bold_first and i == 0 else ""
            cls = ' class="count"' if i == count_line else ""
            ty = y_text + SVG_LINE_PX * (i + 1) - 4
            self.parts.append(
                f'<text{cls} x="{tx}" y="{ty}" text-anchor="middle" fill="{text_fill}"'
                f'{weight}>{line}</text>')
        if node:
            self.parts.append('</g>')
        self.bottom = max(self.bottom, y + h)
        return x, y, w, h

    def line(self, x1: float, y1: float, x2: float, y2: float, *, arrow: bool = False,
             frm=None, to=None) -> None:
        head = ' marker-end="url(#flow-arrow)"' if arrow else ""
        self.parts.append(
            f'<line class="edge"{self._edge_attrs(frm, to)} x1="{x1}" y1="{y1}" x2="{x2}" '
            f'y2="{y2}" stroke="currentColor" stroke-width="1.2"{head}/>')

    def label(self, x: float, y: float, lines: list[str], *, anchor: str = "middle",
              bold: bool = False, size: int | None = None, frm=None, to=None) -> None:
        weight = ' font-weight="bold"' if bold else ""
        fs = f' font-size="{size}"' if size else ""
        cls = ' class="elabel"' if frm or to else ""
        for i, line in enumerate(lines):
            self.parts.append(
                f'<text{cls}{self._edge_attrs(frm, to)} x="{x}" y="{y + SVG_LINE_PX * i}" '
                f'text-anchor="{anchor}" fill="currentColor"{weight}{fs}>{line}</text>')
        self.bottom = max(self.bottom, int(y + SVG_LINE_PX * len(lines)))

    def step(self, x: float, y1: float, y2: float, lines: list[str], *, frm=None,
             to=None) -> None:
        """An arrow from y1 down to y2 with its label in the middle, the line broken
        around the text so the words are never struck through."""
        block = SVG_LINE_PX * len(lines)
        top = (y1 + y2) / 2 - block / 2
        self.line(x, y1, x, int(top) - 4, frm=frm, to=to)
        self.label(x, int(top) + SVG_LINE_PX - 4, lines, frm=frm, to=to)
        self.line(x, int(top + block) + 4, x, y2, arrow=True, frm=frm, to=to)

    def fan(self, sources: list[float], y_from: float, targets: list[float], y_to: float,
            lines: list[str], *, frm: list[str] | None = None,
            to: list[str] | None = None) -> None:
        """Several boxes feeding several boxes through one horizontal bar, labelled above
        the bar. Two lines in and three arrows out instead of six crossing arrows. `frm`
        and `to` are the node ids in the same order as `sources` and `targets`."""
        frm = frm or []
        to = to or []
        ybar = y_from + 18 + SVG_LINE_PX * len(lines)
        for i, sx in enumerate(sources):
            self.line(sx, y_from, sx, ybar, frm=frm[i] if i < len(frm) else None, to=to)
        self.line(min(sources + targets), ybar, max(sources + targets), ybar, frm=frm, to=to)
        self.label(self.w / 2, ybar - 6 - SVG_LINE_PX * (len(lines) - 1), lines, frm=frm, to=to)
        for i, tx in enumerate(targets):
            self.line(tx, ybar, tx, y_to, arrow=True, frm=frm, to=to[i] if i < len(to) else None)

    def group(self, frm, to) -> None:
        """Open a group of marks that fades as one edge; close it with end_group()."""
        self.parts.append(f'<g class="edge-group"{self._edge_attrs(frm, to)}>')

    def end_group(self) -> None:
        self.parts.append('</g>')

    def stack(self, cx: float, y: float, n: int, marked: set[int], *, removed: bool = False,
              w: int = 70, colour: str = C_CLADE) -> int:
        """A database as a stack of entries, one line each, centred on cx from y down.

        Indices in `marked` are drawn in `colour`, or left as a gap when `removed`.
        Returns the y below the stack.
        """
        step = 6
        for i in range(n):
            if i in marked and removed:
                continue
            c = colour if i in marked else "currentColor"
            self.parts.append(
                f'<rect x="{cx - w / 2}" y="{y + i * step}" width="{w}" height="3" rx="1.5" '
                f'fill="{c}"/>')
        self.bottom = max(self.bottom, int(y + n * step))
        return int(y + n * step)

    def swatch(self, x: int, y: int, text: str, *, fill: str | None = None,
               stroke: str = "currentColor", stroke_w: float = 1.5, dashed: bool = False,
               arrow: bool = False, icon: str | None = None, opacity: float = 1.0) -> int:
        """One legend entry at (x, y); returns the x where the next one can start."""
        if arrow:
            self.line(x, y - 4, x + 22, y - 4, arrow=True)
        elif icon:
            self.icon(icon, x, y - 15, px=20)
        else:
            dash = ' stroke-dasharray="4 3"' if dashed else ""
            op = f' opacity="{opacity}"' if opacity < 1 else ""
            self.parts.append(
                f'<rect x="{x}" y="{y - 10}" width="22" height="12" rx="2" '
                f'fill="{fill or "none"}" stroke="{stroke}" stroke-width="{stroke_w}"{dash}{op}/>')
        self.parts.append(f'<text x="{x + 28}" y="{y}" fill="currentColor">{text}</text>')
        self.bottom = max(self.bottom, y + 4)
        return x + 28 + SVG_CHAR_PX * len(text) + 22

    def legend(self, rows: list[list[dict]], y: int = 18) -> int:
        """Legend rows, each a list of swatch() keyword dicts with a `text`. Returns the y
        below the last row. Rows are the caller's: a row that would run past the right
        edge is the caller's to split, and this raises rather than clipping."""
        for row in rows:
            x = 20
            for entry in row:
                x = self.swatch(x, y, **entry)
            if x - 22 > self.w:
                raise ValueError(f"legend row runs past {self.w} px: {[e['text'] for e in row]}")
            y += 22
        return y

    def render(self) -> str:
        h = self.bottom + SVG_PAD
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {h}" '
            f'width="100%" style="max-width:{self.w}px;font-family:sans-serif;'
            f'font-size:{SVG_FONT_PX}px;display:block;margin:0 auto">'
            '<defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>'
            + "".join(self.parts) + "</svg>")


def columns(n: int, gap: int = 20, width: int = SVG_W, margin: int = 20) -> list[tuple[int, int]]:
    """(x, w) for n equal columns across the frame."""
    w = (width - 2 * margin - gap * (n - 1)) // n
    return [(margin + i * (w + gap), w) for i in range(n)]


def mid(col: tuple[int, int]) -> int:
    return col[0] + col[1] // 2


INTERACTION_LEGEND = [
    dict(text="faded: not upstream of the box under the pointer", opacity=0.25),
    dict(text="orange edge: the box whose details are open", stroke=C_OPEN, stroke_w=2.5),
]

CSS = """
<style>
.flow-block .grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,1fr);gap:18px;align-items:start}
@media (max-width:820px){.flow-block .grid{grid-template-columns:1fr}}
.flow-block .diagram{border:1px solid rgba(127,127,127,.3);border-radius:8px;padding:8px;overflow-x:auto}
.flow-block .node{cursor:pointer;transition:opacity .15s}
.flow-block .node:focus{outline:none}
.flow-block .node:focus-visible rect.box,.flow-block .node.selected rect.box{stroke:%(open)s;stroke-width:3}
.flow-block .edge,.flow-block .elabel,.flow-block .edge-group{transition:opacity .15s}
.flow-block .faded{opacity:.22}
.flow-block .share{transition:width .25s}
.flow-block .panel{border:1px solid rgba(127,127,127,.3);border-radius:8px;padding:12px 14px;position:sticky;top:12px}
.flow-block .panel h5{margin:0 0 4px;font-size:15px;font-weight:600}
.flow-block .panel .n{font-family:monospace;font-size:14px;margin:0 0 8px;opacity:.85}
.flow-block .panel p{margin:0 0 8px}
.flow-block .panel dl{margin:0 0 10px;display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12.5px}
.flow-block .panel dt{opacity:.7}
.flow-block .panel dd{margin:0;font-family:monospace;font-size:12px}
.flow-block .links a{display:inline-block;font-size:12px;border:1px solid rgba(127,127,127,.5);border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;text-decoration:none}
.flow-block .links a:hover{border-color:currentColor}
.flow-block .control{margin:0 0 10px}
.flow-block .control label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.7;margin-bottom:5px}
.flow-block .seg{display:inline-flex;flex-wrap:wrap;border:1px solid rgba(127,127,127,.5);border-radius:6px;overflow:hidden}
.flow-block .seg button{background:transparent;color:inherit;border:0;border-right:1px solid rgba(127,127,127,.5);padding:5px 11px;font:inherit;cursor:pointer}
.flow-block .seg button:last-child{border-right:0}
.flow-block .seg button[aria-pressed="true"]{background:%(open)s;color:#fff}
.flow-block .foot{font-size:12px;opacity:.75;margin-top:10px}
@media (prefers-reduced-motion: reduce){.flow-block .node,.flow-block .edge,.flow-block .elabel,.flow-block .share{transition:none}}
</style>
"""

JS = """
<script>
(function(){
  var root=document.getElementById('%(uid)s'); if(!root) return;
  var svg=root.querySelector('svg');
  var data=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  var nodes={}; svg.querySelectorAll('.node').forEach(function(g){nodes[g.getAttribute('data-id')]=g;});
  var edges=[]; svg.querySelectorAll('[data-from]').forEach(function(e){
    var f=e.getAttribute('data-from'), t=e.getAttribute('data-to');
    edges.push({el:e,from:f?f.split(' '):[],to:t?t.split(' '):[]});});
  var parents={}; edges.forEach(function(e){e.to.forEach(function(t){e.from.forEach(function(f){(parents[t]=parents[t]||[]).push(f);});});});
  function upstream(id){var s={};s[id]=1;var st=[id];while(st.length){var n=st.pop();(parents[n]||[]).forEach(function(p){if(!s[p]){s[p]=1;st.push(p);}});}return s;}
  var selected=null, option=(data.control&&data.control.options.length)?data.control.options[0]:null;
  function highlight(id){
    if(!id){Object.keys(nodes).forEach(function(k){nodes[k].classList.remove('faded');});edges.forEach(function(e){e.el.classList.remove('faded');});return;}
    var up=upstream(id);
    Object.keys(nodes).forEach(function(k){nodes[k].classList.toggle('faded',!up[k]);});
    edges.forEach(function(e){var a=e.from.some(function(f){return up[f];}),b=e.to.some(function(t){return up[t];});e.el.classList.toggle('faded',!(a&&b));});
  }
  function opt(key,id){return option&&option[key]&&option[key][id]!==undefined?option[key][id]:null;}
  function select(id){
    selected=id;
    Object.keys(nodes).forEach(function(k){nodes[k].classList.toggle('selected',k===id);});
    highlight(id);
    var d=data.details[id]||{title:id,count:'',text:'',facts:[],links:[]};
    root.querySelector('.p-title').textContent=d.title||id;
    var sub=opt('subs',id); root.querySelector('.p-count').textContent=sub!==null?sub:(d.count||'');
    root.querySelector('.p-text').textContent=d.text||'';
    var dl=root.querySelector('.p-facts'); dl.innerHTML='';
    var facts=(opt('facts',id)||[]).concat(d.facts||[]);
    facts.forEach(function(kv){var dt=document.createElement('dt');dt.textContent=kv[0];var dd=document.createElement('dd');dd.textContent=kv[1];dl.appendChild(dt);dl.appendChild(dd);});
    var L=root.querySelector('.p-links'); L.innerHTML='';
    (opt('links',id)||d.links||[]).forEach(function(nl){var a=document.createElement('a');a.href='#'+nl[1];a.textContent='\u2192 '+nl[0];L.appendChild(a);});
  }
  Object.keys(nodes).forEach(function(id){var g=nodes[id];
    g.addEventListener('mouseenter',function(){highlight(id);});
    g.addEventListener('mouseleave',function(){highlight(selected);});
    g.addEventListener('focus',function(){highlight(id);});
    g.addEventListener('blur',function(){highlight(selected);});
    g.addEventListener('click',function(){select(id);});
    g.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();select(id);}});
  });
  function applyOption(){
    if(!option) return;
    Object.keys(option.subs||{}).forEach(function(id){var g=nodes[id]; if(!g) return;
      var t=g.querySelector('text.count'); if(t) t.textContent=option.subs[id];});
    Object.keys(option.shares||{}).forEach(function(id){var g=nodes[id]; if(!g) return;
      var sp=svg.querySelector('[data-share="'+id+'"]'); if(sp){var w=+g.querySelector('rect.box').getAttribute('width'); sp.setAttribute('width',w*option.shares[id]);}});
    if(selected) select(selected);
  }
  root.querySelectorAll('.seg button').forEach(function(b){b.addEventListener('click',function(){
    root.querySelectorAll('.seg button').forEach(function(x){x.setAttribute('aria-pressed','false');});
    b.setAttribute('aria-pressed','true');
    var id=b.getAttribute('data-option'); option=data.control.options.filter(function(o){return o.id===id;})[0]||null; applyOption();});});
  applyOption();
})();
</script>
"""


def interactive(svg: str, details: dict, *, control: dict | None = None,
                footnote: str | None = None, uid: str = "flow-block") -> str:
    """The diagram with its details panel, an optional control, and the script.

    `details` is {node id: {title, count, text, facts: [[k, v]], links: [[name, anchor]]}}.
    `control` is {label, options: [{id, label, subs: {id: text}, shares: {id: fraction},
    facts: {id: [[k, v]]}, links: {id: [[name, anchor]]}}]}; the first option is active
    to start with. Everything is data, so the script is the same in every report.
    """
    controls = ""
    if control and control.get("options"):
        buttons = "".join(
            f'<button type="button" data-option="{o["id"]}" '
            f'aria-pressed="{"true" if i == 0 else "false"}">{o["label"]}</button>'
            for i, o in enumerate(control["options"]))
        controls = (f'<div class="control"><label>{control["label"]}</label>'
                    f'<div class="seg" role="group" aria-label="{control["label"]}">{buttons}</div></div>')
    payload = json.dumps({"details": details, "control": control})
    foot = f'<p class="foot">{footnote}</p>' if footnote else ""
    return (
        CSS % {"open": C_OPEN}
        + f'<div id="{uid}" class="flow-block">'
        + controls
        + '<div class="grid">'
        + f'<div class="diagram">{svg}</div>'
        + '<aside class="panel" aria-live="polite"><h5 class="p-title">Click a box</h5>'
          '<p class="n p-count"></p>'
          '<p class="p-text">Each box opens here with what it is, how it was made, and which '
          'panels below show it. Hover a box first to see what feeds it.</p>'
          '<dl class="p-facts"></dl><div class="links p-links"></div></aside>'
        + '</div>' + foot
        + f'<script type="application/json">{payload}</script>'
        + JS % {"uid": uid}
        + '</div>')
