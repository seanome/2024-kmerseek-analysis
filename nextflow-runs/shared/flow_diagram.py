"""Data-flow diagrams for the MultiQC report overviews.

Every MultiQC report in this repository opens with a section that says what was done, why,
and how the data flowed from the inputs to the panels. The picture half of that is drawn
here, from a spec each report's builder writes: boxes (a name and one number each),
junctions where two inputs meet, swimlanes naming what kind of thing each row holds, the
arrows between boxes, what each box opens with when clicked, and a control whose options
carry the numbers that change with it. One script, shared by every report, routes and
draws the spec in the browser, so the four reports are drawn the same way and a change
to the drawing rules is one change.

A builder stages this file next to itself (Nextflow: `path 'flow_diagram.py'` plus
`PYTHONPATH=$PWD`) or, run by hand from a pipeline's bin/, finds it at ../../shared.

Drawing rules, from the clear-figures checklist and the 2026-09-18 review:
  no crossing lines      two inputs meet at a junction (the dot) and one bus fans out
                         from it; an arrow drops straight down when its source sits over
                         its target; a side input enters a wide box at its own x
  row labels             a swimlane label on the left says what a row holds (inputs,
                         search, hits, ...) before any box is read
  one name, one number   a box holds a title and one count; everything else is in the
                         panel when the box is clicked
  labels beside lines    a step label sits beside its arrow, never on it
  legend before marks    the legend is above the drawing, and every bar's meaning is in it
  bars that move         a bar along the bottom of a box is its value on one scale per
                         row; switching the control slides it and leaves an orange tick
                         where it was, so the size and direction of the change stay visible
  no fake numbers        a bar shows only what the report itself carries

Spec, as JSON:
  title, subtitle, warning?   text above the drawing
  height                      SVG height; width is 760 plus a 34 px margin for lane labels
  kinds: {kind: {color, fill?, label}}          box outline (or fill) colours, in the legend
  barLegend?                  what a bar along the bottom of a box means (and its full width)
  lanes: [[y0, y1, label], ...]                  swimlanes, top to bottom
  headers: [[x, y, text], ...]                   column headings
  nodes: {id: {x, y, w, h, icon, kind?, title, sub?, bar?, strip?, dashed?, samples?} |
          {junction: true, x, y, label?, labelSide?}}
                              samples: the names MultiQC's toolbox highlight/hide is matched
                              against for that box (tool or encoding names)
  classOf?: {label prefix: kind}                 colours a facts row by the class its label starts with
  iconLegend?                 one line naming what each icon stands for (a default is supplied)
  edges: [{from, to, label?, bus?, lx?, ly?, anchor?}, ...]
  details: {id: {text, facts: [[k, v]], links: [[name, anchor]]}}
  control?: {label, options: [{id, label, subs: {id: text}, bars: {id: fraction},
             facts: {id: [[k, v]]}, links: {id: [[name, anchor]]}}]}
  footnote?
"""

import json

# Green is "something was found" and grey is "nothing was found", in every report that uses
# these. Purple marks the query's own clade where a report removes it from the target.
C_FOUND = "#0f9d76"
C_NONE = "#7f7f7f"
C_CLADE = "#7b4fb3"
# The box whose details are open, and the tick where a bar was before the last click. Not
# data colours: they mark a state of the page and appear nowhere else in any report.
C_OPEN = "#d97b00"

# Google Material Symbols (Apache 2.0), outlined, 24 px, fetched from
# fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/<name>/default/24px.svg
# and inlined, because the reports are rendered on compute nodes with no internet and have
# to stay one self-contained file each. viewBox is 0 -960 960 960 for every one.
#
#   genetics     query sequences          search        a search tool
#   database     a target database        table_rows    a table of hits or calls
#   task_alt     found / placed           search_off    nothing found
#   straighten   protein length           waves         disorder
#   summarize    the report itself        deployed_code predicted structures
ICONS = {
    "database": "M480-120q-151 0-255.5-46.5T120-280v-400q0-66 105.5-113T480-840q149 0 254.5 47T840-680v400q0 67-104.5 113.5T480-120Zm0-479q89 0 179-25.5T760-679q-11-29-100.5-55T480-760q-91 0-178.5 25.5T200-679q14 30 101.5 55T480-599Zm0 199q42 0 81-4t74.5-11.5q35.5-7.5 67-18.5t57.5-25v-120q-26 14-57.5 25t-67 18.5Q600-528 561-524t-81 4q-42 0-82-4t-75.5-11.5Q287-543 256-554t-56-25v120q25 14 56 25t66.5 18.5Q358-408 398-404t82 4Zm0 200q46 0 93.5-7t87.5-18.5q40-11.5 67-26t32-29.5v-98q-26 14-57.5 25t-67 18.5Q600-328 561-324t-81 4q-42 0-82-4t-75.5-11.5Q287-343 256-354t-56-25v99q5 15 31.5 29t66.5 25.5q40 11.5 88 18.5t94 7Z",
    "genetics": "M200-40v-40q0-139 58-225.5T418-480q-102-88-160-174.5T200-880v-40h80v40q0 11 .5 20.5T282-840h396q1-10 1.5-19.5t.5-20.5v-40h80v40q0 139-58 225.5T542-480q102 88 160 174.5T760-80v40h-80v-40q0-11-.5-20.5T678-120H282q-1 10-1.5 19.5T280-80v40h-80Zm138-640h284q13-19 22.5-38t17.5-42H298q8 22 17.5 41.5T338-680Zm142 148q20-17 39-34t36-34H405q17 17 36 34t39 34Zm-75 172h150q-17-17-36-34t-39-34q-20 17-39 34t-36 34ZM298-200h364q-8-22-17.5-41.5T622-280H338q-13 19-22.5 38T298-200Z",
    "search": "M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z",
    "search_off": "M138.5-138.5Q80-197 80-280t58.5-141.5Q197-480 280-480t141.5 58.5Q480-363 480-280t-58.5 141.5Q363-80 280-80t-141.5-58.5ZM824-120 568-376q-12-13-25.5-26.5T516-428q38-24 61-64t23-88q0-75-52.5-127.5T420-760q-75 0-127.5 52.5T240-580q0 6 .5 11.5T242-557q-18 2-39.5 8T164-535q-2-11-3-22t-1-23q0-109 75.5-184.5T420-840q109 0 184.5 75.5T680-580q0 43-13.5 81.5T629-428l251 252-56 56Zm-615-61 71-71 70 71 29-28-71-71 71-71-28-28-71 71-71-71-28 28 71 71-71 71 28 28Z",
    "straighten": "M160-240q-33 0-56.5-23.5T80-320v-320q0-33 23.5-56.5T160-720h640q33 0 56.5 23.5T880-640v320q0 33-23.5 56.5T800-240H160Zm0-80h640v-320H680v160h-80v-160h-80v160h-80v-160h-80v160h-80v-160H160v320Zm120-160h80-80Zm160 0h80-80Zm160 0h80-80Zm-120 0Z",
    "summarize": "M348.5-611.5Q360-623 360-640t-11.5-28.5Q337-680 320-680t-28.5 11.5Q280-657 280-640t11.5 28.5Q303-600 320-600t28.5-11.5Zm0 160Q360-463 360-480t-11.5-28.5Q337-520 320-520t-28.5 11.5Q280-497 280-480t11.5 28.5Q303-440 320-440t28.5-11.5Zm0 160Q360-303 360-320t-11.5-28.5Q337-360 320-360t-28.5 11.5Q280-337 280-320t11.5 28.5Q303-280 320-280t28.5-11.5ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h440l200 200v440q0 33-23.5 56.5T760-120H200Zm0-80h560v-400H600v-160H200v560Zm0-560v160-160 560-560Z",
    "table_rows": "M760-200v-120H200v120h560Zm0-200v-160H200v160h560Zm0-240v-120H200v120h560ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Z",
    "task_alt": "M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q65 0 123 19t107 53l-58 59q-38-24-81-37.5T480-800q-133 0-226.5 93.5T160-480q0 133 93.5 226.5T480-160q133 0 226.5-93.5T800-480q0-18-2-36t-6-35l65-65q11 32 17 66t6 70q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm-56-216L254-466l56-56 114 114 400-401 56 56-456 457Z",
    "waves": "M80-146v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62.5-8.5q37.5 0 62.5 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-174.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-164 544.5-155t-64.5 9q-39 0-64.5-9t-46-19.5Q349-185 329-193.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-164 143.5-155T80-146Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-352.5q-21-10.5-41-19t-49-8.5q-29 0-49.5 8.5t-41 19Q569-342 544-333t-64 9q-39 0-64.5-9t-46-19.5Q349-363 329-371.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-342 143.5-333T80-324Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-530.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-520 544.5-511t-64.5 9q-39 0-64.5-9t-46-19.5Q349-541 329-549.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-520 143.5-511T80-502Zm0-178v-78q29 0 49.5-9t41.5-19.5q21-10.5 46.5-19t63-8.5q37.5 0 62 8.5t45.5 19q21 10.5 42 19.5t50 9q29 0 50-9t42-19.5q21-10.5 46-19t62-8.5q38 0 63 8.5t46 19q21 10.5 42 19.5t49 9v78q-38 0-63.5-9T770-708.5q-21-10.5-41-19t-49-8.5q-28 0-48.5 8.5t-41 19Q570-698 544.5-689t-64.5 9q-39 0-64.5-9t-46-19.5Q349-719 329-727.5t-48.5-8.5q-28.5 0-49 8.5t-41.5 19Q169-698 143.5-689T80-680Z",    "deployed_code": "M440-91 160-252q-19-11-29.5-29T120-321v-318q0-22 10.5-40t29.5-29l280-161q19-11 40-11t40 11l280 161q19 11 29.5 29t10.5 40v318q0 22-10.5 40T800-252L520-91q-19 11-40 11t-40-11Zm0-92v-284L200-608v276l240 149Zm80 0 240-149v-276L520-467v284ZM480-536l237-137-237-137-237 137 237 137Z",
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


def header_yaml(items: list[tuple[str, str]]) -> str:
    """MultiQC's report_header_info (the key-value block under the report title), as
    YAML written by hand: the scoring containers carry no PyYAML, and a list of quoted
    strings needs none. Every value is double-quoted, so colons and commas are safe."""
    def q(v) -> str:
        # MultiQC renders these values as HTML and drops everything from a bare "<" on:
        # "E <= 0.001; dark when none has" rendered as "E " until 2026-09-19. Comparison
        # signs become the one-character symbols, and any other "<" is escaped.
        v = str(v).replace("<=", "\u2264").replace(">=", "\u2265").replace("<", "&lt;")
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "report_header_info:\n" + "".join(f"  - {q(k)}: {q(v)}\n" for k, v in items)


def yaml_html(key: str, html: str) -> str:
    """One top-level MultiQC config key holding HTML (intro_text, report_comment), as a
    double-quoted YAML scalar. Unlike header_yaml this keeps every "<": the value is meant
    to be markup. Newlines become spaces so the scalar stays on one line."""
    v = " ".join(html.split())
    return f'{key}: "' + v.replace("\\", "\\\\").replace('"', '\\"') + '"\n'


# The v3 renderer and its style, from the 2026-09-19 review (flow.js, flow.css), inlined so
# a report stays one self-contained file. The style is built on currentColor, so the block
# takes MultiQC's light or dark theme with no configuration. Beyond drawing the spec it
# gives: a resizable diagram | panel split (drag, arrow keys, double-click resets, width
# remembered per report), key chips that light one class of box, "made from" and "goes
# into" chips in the panel, mini bars beside numbers with a ± SD, pin one option to
# compare against, keyboard navigation between boxes, the selection and option in the URL
# hash, MultiQC toolbox highlight/hide applied to boxes by their sample names, and SVG
# export with the styles baked in.
CSS = "<style>" + r"""
/* flow-block v3: Material-3-style tokens built on currentColor, so the block takes the host page's theme (MultiQC light or dark) with no configuration. */
.flow-block{
  --fb-accent:#d97b00;                 /* one attention colour: selection ring, ghost tick, drag handle when active */
  --fb-ink:currentColor;
  --fb-ink-2:color-mix(in oklab,currentColor 72%,transparent);
  --fb-ink-3:color-mix(in oklab,currentColor 52%,transparent);
  --fb-surf-1:color-mix(in oklab,currentColor 4%,transparent);
  --fb-surf-2:color-mix(in oklab,currentColor 7%,transparent);
  --fb-surf-3:color-mix(in oklab,currentColor 11%,transparent);
  --fb-outline:color-mix(in oklab,currentColor 22%,transparent);
  --fb-outline-2:color-mix(in oklab,currentColor 12%,transparent);
  --fb-hover:color-mix(in oklab,currentColor 8%,transparent);
  --fb-radius:10px;
  --fb-flow-col:62%;
  font-size:13px;line-height:1.4;container-type:inline-size;
}
.flow-block *{box-sizing:border-box}
.flow-block .fb-top{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;margin:0 0 10px}
.flow-block .fb-control{display:flex;align-items:center;gap:10px}
.flow-block .fb-control label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--fb-ink-3);margin:0}
.flow-block .seg{display:inline-flex;border:1px solid var(--fb-outline);border-radius:999px;overflow:hidden;background:transparent}
.flow-block .seg button{background:transparent;color:var(--fb-ink);border:0;border-right:1px solid var(--fb-outline);padding:5px 14px;font:inherit;font-size:12.5px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;transition:background .12s}
.flow-block .seg button:last-child{border-right:0}
.flow-block .seg button:hover{background:var(--fb-hover)}
.flow-block .seg button[aria-pressed="true"]{background:color-mix(in oklab,var(--fb-accent) 18%,transparent);font-weight:500}
.flow-block .seg button[aria-pressed="true"]::before{content:"";width:12px;height:7px;border:2px solid currentColor;border-top:0;border-right:0;transform:translateY(-2px) rotate(-45deg);display:inline-block}
.flow-block .seg button:focus-visible,.flow-block .chip:focus-visible,.flow-block .fb-handle:focus-visible,.flow-block .fb-iconbtn:focus-visible{outline:2px solid var(--fb-accent);outline-offset:2px}
.flow-block .warn{border-left:3px solid var(--fb-accent);background:color-mix(in oklab,var(--fb-accent) 10%,transparent);border-radius:6px;padding:7px 12px;margin:0 0 10px;font-size:12.5px}
.flow-block .fb-legend{display:flex;flex-wrap:wrap;gap:4px 6px;margin:0 0 10px;font-size:11.5px}
.flow-block .chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--fb-outline);border-radius:8px;padding:2px 9px;min-height:24px;color:var(--fb-ink);text-decoration:none;background:transparent;font:inherit;font-size:11.5px;cursor:pointer;transition:background .12s}
.flow-block a.chip:hover,.flow-block button.chip:hover{background:var(--fb-hover)}
.flow-block .chip .sw{width:18px;height:10px;border:1.5px solid currentColor;border-radius:2px;display:inline-block;flex:none;position:relative}
.flow-block .chip .sw.faded{opacity:.25}
.flow-block .chip .sw.bar::after{content:"";position:absolute;left:1px;right:40%;bottom:1px;height:3px;background:currentColor;opacity:.8}
.flow-block .chip .sw.ghost::after{content:"";position:absolute;left:60%;bottom:-1px;width:2px;height:8px;background:var(--fb-accent)}
.flow-block .chip .dot{width:9px;height:9px;border-radius:50%;background:currentColor;display:inline-block;flex:none;margin:0 4px}
.flow-block .chip .arrow{width:18px;height:10px;display:inline-block;flex:none}
.flow-block .chip.kind{border-width:1.5px}
.flow-block .fb-legend details{display:contents}
.flow-block .fb-legend summary{list-style:none;cursor:pointer}
.flow-block .fb-legend summary::-webkit-details-marker{display:none}

/* split layout with a drag handle */
.flow-block .fb-split{display:grid;grid-template-columns:minmax(260px,var(--fb-flow-col)) 14px minmax(240px,1fr);gap:0;align-items:stretch}
.flow-block .fb-split.fb-dragging{user-select:none;cursor:col-resize}
.flow-block .fb-diagram{border:1px solid var(--fb-outline-2);border-radius:var(--fb-radius);padding:8px;overflow:auto;background:var(--fb-surf-1);min-width:0}
.flow-block .fb-handle{cursor:col-resize;display:flex;align-items:center;justify-content:center;touch-action:none;border-radius:6px;position:relative}
.flow-block .fb-handle::before{content:"";width:4px;height:36px;border-radius:2px;background:var(--fb-outline);transition:background .12s,height .12s}
.flow-block .fb-handle:hover::before,.flow-block .fb-handle:focus-visible::before,.flow-block .fb-split.fb-dragging .fb-handle::before{background:var(--fb-accent);height:56px}
.flow-block .fb-handle::after{content:"";position:absolute;inset:0 3px}
.flow-block .fb-panel{border:1px solid var(--fb-outline-2);border-radius:var(--fb-radius);background:var(--fb-surf-2);padding:12px 14px;position:sticky;top:8px;align-self:start;min-width:0;max-height:calc(100vh - 16px);overflow:auto}
@container (max-width:720px){.flow-block .fb-split{grid-template-columns:1fr}.flow-block .fb-handle{display:none}.flow-block .fb-panel{position:static;max-height:none}}

/* diagram */
.flow-block svg.flow{display:block;width:100%;height:auto;font-family:inherit;font-size:12px;color:inherit}
.flow-block .lane{fill:currentColor;opacity:.045}
.flow-block .lanelabel{fill:currentColor;opacity:.55;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase}
.flow-block .colhead{fill:currentColor;opacity:.75;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
.flow-block .node{cursor:pointer;transition:opacity .15s}
.flow-block .node rect.box{fill:var(--fb-box-fill,var(--fb-surf-2));stroke:var(--fb-box-stroke,var(--fb-outline));stroke-width:1.5;transition:fill .12s}
.flow-block .node:hover rect.box{fill:var(--fb-box-hover,var(--fb-surf-3))}
.flow-block .node.dashed rect.box{stroke-dasharray:5 4}
.flow-block .node.filled text,.flow-block .node.filled .icon{fill:#fff}
.flow-block .node text{fill:currentColor}
.flow-block .node text.count{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;opacity:.85}
.flow-block .node .icon{fill:currentColor;opacity:.8}
.flow-block .node:focus{outline:none}
.flow-block .node:focus-visible rect.box,.flow-block .node.selected rect.box{stroke-width:3;stroke:var(--fb-accent)!important}
.flow-block .junction circle{fill:currentColor}
.flow-block .junction text{fill:currentColor;opacity:.7;font-size:11px}
.flow-block .edge{stroke:currentColor;stroke-width:1.2;fill:none;transition:opacity .15s;opacity:.85}
.flow-block .elabel{fill:currentColor;opacity:.7;font-size:10.5px;transition:opacity .15s}
.flow-block .faded{opacity:.18!important}
.flow-block .vbar{fill:var(--fb-bar,currentColor);opacity:.85;transition:width .45s cubic-bezier(.2,.8,.2,1)}
.flow-block .filled .vbar{fill:#fff;opacity:.6}
.flow-block .vtrack{fill:currentColor;opacity:.08}
.flow-block .vghost{fill:var(--fb-accent);transition:x .45s cubic-bezier(.2,.8,.2,1),opacity .45s}

/* panel */
.flow-block .fb-panel h5{margin:0;font-size:14px;font-weight:600;line-height:1.3}
.flow-block .fb-kind{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--fb-ink-2);margin:0 0 4px}
.flow-block .fb-kind i{width:10px;height:10px;border-radius:3px;display:inline-block;background:var(--k,currentColor)}
.flow-block .fb-panel .n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;margin:2px 0 8px;color:var(--fb-ink-2)}
.flow-block .fb-panel p{margin:0 0 8px;font-size:12.5px}
.flow-block .fb-sec{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--fb-ink-3);margin:10px 0 4px;display:flex;align-items:center;gap:8px}
.flow-block .fb-sec::after{content:"";flex:1;height:1px;background:var(--fb-outline-2)}
.flow-block table.facts{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}
.flow-block table.facts td{padding:3px 6px 3px 0;vertical-align:top;border-bottom:1px solid var(--fb-outline-2)}
.flow-block table.facts tr:last-child td{border-bottom:0}
.flow-block table.facts td.k{color:var(--fb-ink-2);width:52%;word-wrap:break-word}
.flow-block table.facts td.v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.flow-block table.facts td.b{width:64px;padding-left:6px}
.flow-block .mbar{position:relative;height:8px;border-radius:2px;background:var(--fb-outline-2);margin-top:4px}
.flow-block .mbar i{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:var(--k,currentColor);opacity:.8}
.flow-block .mbar b{position:absolute;top:-1px;bottom:-1px;width:0;border-left:1.5px solid currentColor;opacity:.55}
.flow-block .mbar s{position:absolute;top:3px;height:2px;background:currentColor;opacity:.35;text-decoration:none}
.flow-block .fb-chips{display:flex;flex-wrap:wrap;gap:4px}
.flow-block .fb-note{color:var(--fb-ink-3);font-size:11px;margin-top:10px}
.flow-block .fb-foot{color:var(--fb-ink-3);font-size:11.5px;margin:10px 0 0;max-width:90ch}
.flow-block .fb-iconbtn{background:transparent;border:1px solid var(--fb-outline);border-radius:999px;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;color:var(--fb-ink-2);font:inherit;font-size:12px}
.flow-block .fb-iconbtn:hover{background:var(--fb-hover)}
.flow-block .fb-kbd{font-size:11px;color:var(--fb-ink-3)}
.flow-block .fb-kbd kbd{border:1px solid var(--fb-outline);border-radius:4px;padding:0 4px;font:inherit;font-size:10.5px}
@media (prefers-reduced-motion:reduce){.flow-block .vbar,.flow-block .vghost,.flow-block .node,.flow-block .edge,.flow-block .elabel{transition:none}}

/* v3.1: pin, toolbox highlight, export */
.flow-block .fb-pin.on{background:color-mix(in oklab,var(--fb-accent) 18%,transparent);border-color:var(--fb-accent)}
.flow-block .vpin{fill:none;stroke:var(--fb-bar,currentColor);stroke-width:1.5;transition:width .45s cubic-bezier(.2,.8,.2,1),opacity .3s}
.flow-block .filled .vpin{stroke:#fff}
.flow-block .mbar u{position:absolute;left:0;top:-2px;bottom:-2px;border:1.5px solid var(--k,currentColor);border-radius:2px;opacity:.9;text-decoration:none}
.flow-block table.facts .pv{color:var(--fb-ink-3);font-size:10.5px}
.flow-block .node.mqc-hl rect.box{stroke:var(--fb-hl)!important;stroke-width:3.5;filter:drop-shadow(0 0 4px var(--fb-hl))}
.flow-block .node.mqc-dim{opacity:.3}
.flow-block .node.mqc-hidden{opacity:.12}
.flow-block .node.mqc-hidden rect.box{stroke-dasharray:3 3}
.flow-block table.facts td.k.wide{width:auto;border-bottom:0;padding-bottom:0}
.flow-block table.facts td.v.long{text-align:left;white-space:normal;overflow-wrap:anywhere;padding-top:1px}
/* key: flat entries explain; outlined pills act */
.flow-block .fb-keylabel{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--fb-ink-3);align-self:center;margin-right:2px}
.flow-block .key-item{display:inline-flex;align-items:center;gap:6px;padding:2px 4px;min-height:24px;color:var(--fb-ink-2);font-size:11.5px;cursor:default}
.flow-block .key-item .sw,.flow-block .key-item .dot,.flow-block .key-item .arrow{color:var(--fb-ink)}
.flow-block .key-item .sw{width:18px;height:10px;border:1.5px solid currentColor;border-radius:2px;display:inline-block;flex:none;position:relative}
.flow-block .key-item .sw.faded{opacity:.25}
.flow-block .key-item .sw.bar::after{content:"";position:absolute;left:1px;right:40%;bottom:1px;height:3px;background:currentColor;opacity:.8}
.flow-block .key-item .sw.ghost::after{content:"";position:absolute;left:60%;bottom:-1px;width:2px;height:8px;background:var(--fb-accent)}
.flow-block .key-item .dot{width:9px;height:9px;border-radius:50%;background:currentColor;display:inline-block;flex:none;margin:0 4px}
.flow-block .key-item .arrow{width:18px;height:10px;display:inline-block;flex:none}
.flow-block .key-rule{color:var(--fb-ink-3);font-style:italic;margin-left:auto}
.flow-block .chip.kind{cursor:pointer}
.flow-block .chip.kind .chip-act{opacity:.45;font-size:12px;margin-left:-2px}
.flow-block .chip.kind:hover{background:color-mix(in oklab,var(--k) 14%,transparent)}
.flow-block .chip.kind.on{background:color-mix(in oklab,var(--k) 22%,transparent);box-shadow:inset 0 0 0 1px var(--k)}
.flow-block .chip.kind.on .chip-act{opacity:1}
.flow-block .node.kind-hl rect.box{stroke-width:3;filter:drop-shadow(0 0 3px var(--fb-box-stroke,currentColor))}
.flow-block .node.kind-dim{opacity:.28}
.flow-block button.chip,.flow-block a.chip{box-shadow:0 1px 0 color-mix(in oklab,currentColor 10%,transparent)}
.flow-block button.chip:active{transform:translateY(1px);box-shadow:none}
""" + "</style>"

JS = "<script>" + r"""
/* flow-block v3 renderer. Reads the JSON spec inside a `.flow-block` root (the same spec the
   reports already carry) and draws: the diagram, a resizable side panel, legend chips, keyboard
   navigation, and value bars in boxes and in the panel. No dependencies. */
(function(global){
'use strict';
if(global.renderFlow) return;
var NS='http://www.w3.org/2000/svg', OX=34, W=760;
function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e;}
function h(t,cls,text){var e=document.createElement(t);if(cls)e.className=cls;if(text!==undefined&&text!==null)e.textContent=text;return e;}
function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v);}catch(e){return null;}}

function renderFlow(root){
  var spec=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  var icons=spec.icons||{}, nodes=spec.nodes, edges=spec.edges, kinds=spec.kinds||{}, details=spec.details||{};
  var view=root.querySelector('.view')||root; view.innerHTML='';
  var key='fb:'+(root.id||spec.title||'flow');

  // ----- top bar: control + keyboard hint
  var top=h('div','fb-top'); view.appendChild(top);
  var option=(spec.control&&spec.control.options.length)?spec.control.options[0]:null, prevOption=null;
  if(spec.control){
    var c=h('div','fb-control'); c.appendChild(h('label',null,spec.control.label));
    var seg=h('div','seg'); seg.setAttribute('role','group'); seg.setAttribute('aria-label',spec.control.label);
    spec.control.options.forEach(function(o,i){var b=h('button',null,o.label);b.type='button';b.dataset.id=o.id;b.setAttribute('aria-pressed',i===0?'true':'false');seg.appendChild(b);});
    seg.addEventListener('click',function(e){var b=e.target.closest('button');if(!b||b.dataset.id===option.id)return;
      seg.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');
      prevOption=option;option=spec.control.options.filter(function(o){return o.id===b.dataset.id;})[0];applyOption();writeHash();});
    c.appendChild(seg); top.appendChild(c);
  }
  // pin-to-compare and export live beside the control
  var pinned=null, pinBtn=null;
  if(spec.control){
    pinBtn=h('button','chip fb-pin','pin this option to compare'); pinBtn.type='button'; pinBtn.title='draw the pinned option as a hollow bar in every box and as a second column in the panel';
    pinBtn.addEventListener('click',function(){ if(pinned&&pinned.id===option.id){pinned=null;} else pinned=option; pinBtn.textContent=pinned?('pinned: '+pinned.label+' ✕'):'pin this option to compare'; pinBtn.classList.toggle('on',!!pinned); applyOption(); });
    top.appendChild(pinBtn);
  }
  var exp=h('span','fb-chips'); var dl=h('button','chip','⤓ SVG'); dl.type='button'; dl.title='download the drawing as it is now, styles baked in'; var cp=h('button','chip','copy SVG'); cp.type='button'; cp.title='copy the SVG text to the clipboard (for places that block downloads)';
  exp.appendChild(dl); exp.appendChild(cp); top.appendChild(exp);
  var kbd=h('span','fb-kbd'); kbd.innerHTML='<kbd>↑</kbd><kbd>↓</kbd> upstream / downstream · <kbd>←</kbd><kbd>→</kbd> along a row · <kbd>Esc</kbd> clear · drag the bar between the columns'; top.appendChild(kbd);
  // URL state: #<root id>=<box>,<option>  (several blocks join with &)
  function readHash(){var out={};(location.hash||'').replace(/^#/,'').split('&').forEach(function(kv){var i=kv.indexOf('=');if(i>0)out[decodeURIComponent(kv.slice(0,i))]=decodeURIComponent(kv.slice(i+1));});return out;}
  function writeHash(){if(!root.id)return;var hs=readHash();var parts=[];Object.keys(hs).forEach(function(k){if(k!==root.id&&/-flow$/.test(k))parts.push(encodeURIComponent(k)+'='+encodeURIComponent(hs[k]));});
    var mine=[selected||'',option?option.id:''].join(',').replace(/,$/,''); if(mine)parts.push(encodeURIComponent(root.id)+'='+encodeURIComponent(mine));
    try{history.replaceState(null,'','#'+parts.join('&'));}catch(e){}}
  function applyHash(){var v=readHash()[root.id];if(!v)return false;var bits=v.split(',');
    if(bits[1]&&spec.control){var o=spec.control.options.filter(function(x){return x.id===bits[1];})[0];if(o&&o!==option){prevOption=option;option=o;view.querySelectorAll('.seg button').forEach(function(b){b.setAttribute('aria-pressed',b.dataset.id===o.id?'true':'false');});applyOption();}}
    if(bits[0]&&nodes[bits[0]]&&!nodes[bits[0]].junction)select(bits[0],true); return true;}
  if(spec.warning) view.appendChild(h('p','warn',spec.warning));

  // ----- key. Flat entries explain a mark. Kind entries are buttons: hover previews the boxes of that
  // class, click keeps them lit (several can be on); the rule is stated in the row itself.
  var lg=h('div','fb-legend'); lg.setAttribute('aria-label','key');
  lg.appendChild(h('span','fb-keylabel','Key'));
  function keyItem(sw,text){var s=h('span','key-item');s.innerHTML=sw+'<span>'+text+'</span>';lg.appendChild(s);return s;}
  keyItem('<i class="sw"></i>','a set of sequences or results, with its count');
  keyItem('<svg class="arrow" viewBox="0 0 22 12"><line x1="0" y1="6" x2="16" y2="6" stroke="currentColor" stroke-width="1.5"/><path d="M15 2 L21 6 L15 10 z" fill="currentColor"/></svg>','a step, labelled with what it does');
  if(Object.keys(nodes).some(function(k){return nodes[k].junction;})) keyItem('<i class="dot"></i>','two inputs meet here and go on together');
  var activeKinds={}, previewKind=null;
  Object.keys(kinds).forEach(function(k){var kd=kinds[k];var b=h('button','chip kind');b.type='button';b.setAttribute('aria-pressed','false');b.title='click: keep the '+kd.label+' boxes lit · hover: preview';
    b.innerHTML='<i class="sw" style="border-color:'+kd.color+';'+(kd.fill?'background:'+kd.color+';':'background:color-mix(in oklab,'+kd.color+' 14%,transparent);')+'border-width:2px"></i><span>'+kd.label+'</span><span class="chip-act" aria-hidden="true">⌖</span>';
    b.style.borderColor=kd.color; b.style.setProperty('--k',kd.color);
    b.addEventListener('mouseenter',function(){previewKind=k;applyKinds();}); b.addEventListener('mouseleave',function(){previewKind=null;applyKinds();});
    b.addEventListener('click',function(){if(activeKinds[k])delete activeKinds[k];else activeKinds[k]=1;b.setAttribute('aria-pressed',activeKinds[k]?'true':'false');b.classList.toggle('on',!!activeKinds[k]);applyKinds();});
    lg.appendChild(b);});
  if(Object.keys(nodes).some(function(k){return nodes[k].dashed;})) keyItem('<i class="sw" style="border-style:dashed"></i>','dashed: an arm this run did not do');
  if(spec.barLegend) keyItem('<i class="sw bar"></i>',spec.barLegend);
  if(spec.control) keyItem('<i class="sw ghost"></i>','orange tick: where the bar was before the last click');
  keyItem('<i class="sw faded"></i>','faded: not upstream of the box under the pointer');
  keyItem('<i class="sw" style="border-color:var(--fb-accent);border-width:2.5px"></i>','orange edge: the box whose details are open');
  if(spec.iconLegend) keyItem('<span style="opacity:.7">icons</span>',spec.iconLegend);
  lg.appendChild(h('span','key-item key-rule','outlined pills are buttons; flat entries only explain a mark'));
  view.appendChild(lg);
  function applyKinds(){var on=Object.keys(activeKinds);var any=on.length||previewKind;
    Object.keys(nodeEls).forEach(function(id){var n=nodes[id],ne=nodeEls[id];if(n.junction)return;var lit=any&&(activeKinds[n.kind]||previewKind===n.kind);
      ne.g.classList.toggle('kind-hl',!!lit);ne.g.classList.toggle('kind-dim',!!any&&!lit);});}

  // ----- split: diagram | handle | panel
  var split=h('div','fb-split'), dwrap=h('div','fb-diagram'), handle=h('div','fb-handle'), panel=h('aside','fb-panel');
  handle.setAttribute('role','separator'); handle.setAttribute('aria-orientation','vertical'); handle.setAttribute('aria-label','resize the diagram and the details panel'); handle.tabIndex=0; handle.title='drag · arrow keys · double-click resets';
  panel.setAttribute('aria-live','polite');
  split.appendChild(dwrap); split.appendChild(handle); split.appendChild(panel); view.appendChild(split);
  var saved=store(key+':col'); if(saved) root.style.setProperty('--fb-flow-col',saved);
  (function(){
    var startX=0,startW=0,dragging=false;
    function setCol(px){var total=split.getBoundingClientRect().width;var min=260,max=total-14-240;px=Math.max(min,Math.min(max,px));root.style.setProperty('--fb-flow-col',px+'px');handle.setAttribute('aria-valuenow',Math.round(100*px/total));store(key+':col',px+'px');}
    handle.addEventListener('pointerdown',function(e){dragging=true;startX=e.clientX;startW=dwrap.getBoundingClientRect().width;split.classList.add('fb-dragging');handle.setPointerCapture(e.pointerId);e.preventDefault();});
    handle.addEventListener('pointermove',function(e){if(!dragging)return;setCol(startW+(e.clientX-startX));});
    function end(e){if(!dragging)return;dragging=false;split.classList.remove('fb-dragging');try{handle.releasePointerCapture(e.pointerId);}catch(x){}}
    handle.addEventListener('pointerup',end); handle.addEventListener('pointercancel',end);
    handle.addEventListener('dblclick',function(){root.style.removeProperty('--fb-flow-col');store(key+':col','');});
    handle.addEventListener('keydown',function(e){var w=dwrap.getBoundingClientRect().width;var step=e.shiftKey?64:16;
      if(e.key==='ArrowLeft'){setCol(w-step);e.preventDefault();}else if(e.key==='ArrowRight'){setCol(w+step);e.preventDefault();}
      else if(e.key==='Home'){setCol(260);e.preventDefault();}else if(e.key==='End'){setCol(9999);e.preventDefault();}});
  })();

  // ----- diagram
  var svg=el('svg',{class:'flow',viewBox:'0 0 '+(W+OX)+' '+spec.height,role:'img','aria-label':spec.title||'data flow'});
  var mid='ah-'+(root.id||'x'); var defs=el('defs',{}); var mk=el('marker',{id:mid,viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'7',markerHeight:'7',orient:'auto-start-reverse'});
  mk.appendChild(el('path',{d:'M0 0 L10 5 L0 10 z',fill:'currentColor'})); defs.appendChild(mk); svg.appendChild(defs); dwrap.appendChild(svg);
  (spec.lanes||[]).forEach(function(l,i){if(i%2===0)svg.appendChild(el('rect',{class:'lane',x:0,y:l[0],width:W+OX,height:l[1]-l[0],rx:6}));
    var t=el('text',{class:'lanelabel',x:12,y:(l[0]+l[1])/2,'text-anchor':'middle',transform:'rotate(-90 12 '+((l[0]+l[1])/2)+')'});t.textContent=l[2];svg.appendChild(t);});
  var G=el('g',{transform:'translate('+OX+',0)'}); svg.appendChild(G);
  (spec.headers||[]).forEach(function(hd){var t=el('text',{class:'colhead',x:hd[0],y:hd[1],'text-anchor':'middle'});t.textContent=hd[2];G.appendChild(t);});

  var parents={},children={}; edges.forEach(function(e){(parents[e.to]=parents[e.to]||[]).push(e.from);(children[e.from]=children[e.from]||[]).push(e.to);});
  function upstream(id){var s={};s[id]=1;var st=[id];while(st.length){var n=st.pop();(parents[n]||[]).forEach(function(p){if(!s[p]){s[p]=1;st.push(p);}});}return s;}
  function through(list,map){ // expand junctions to the boxes behind them
    var out=[];(list||[]).forEach(function(id){if(nodes[id].junction)through(map[id],map).forEach(function(x){if(out.indexOf(x)<0)out.push(x);});else if(out.indexOf(id)<0)out.push(id);});return out;}
  var cx=function(n){return n.junction?n.x:n.x+n.w/2;}, bottom=function(n){return n.junction?n.y+5:n.y+n.h;}, topOf=function(n){return n.junction?n.y-5:n.y;};
  function route(e){
    var A=nodes[e.from],B=nodes[e.to];
    if(B.junction&&A.junction) return {d:'M'+A.x+' '+(A.y+5)+' L'+A.x+' '+B.y+' L'+(B.x+(A.x<B.x?-6:6))+' '+B.y,lx:(A.x+B.x)/2,ly:B.y-6};
    if(B.junction){var x1=cx(A);if(Math.abs(x1-B.x)<1)return {d:'M'+x1+' '+bottom(A)+' L'+x1+' '+(B.y-6),lx:x1,ly:(bottom(A)+B.y)/2};
      if(Math.abs(A.y+A.h/2-B.y)<A.h/2){var side=x1<B.x?A.x+A.w:A.x;return {d:'M'+side+' '+B.y+' L'+(B.x+(x1<B.x?-6:6))+' '+B.y,lx:(side+B.x)/2,ly:B.y-8};}
      return {d:'M'+x1+' '+bottom(A)+' L'+x1+' '+B.y+' L'+(B.x+(x1<B.x?-6:6))+' '+B.y,lx:x1,ly:(bottom(A)+B.y)/2};}
    var x1=cx(A),x2=cx(B),y1=bottom(A),y2=topOf(B);
    if(!A.junction&&x1>=B.x&&x1<=B.x+B.w) return {d:'M'+x1+' '+y1+' L'+x1+' '+y2,lx:x1,ly:(y1+y2)/2,straight:true};
    var my=A.junction?(A.y+(e.bus||22)):(e.bus!==undefined?e.bus:(y1+y2)/2);
    return {d:'M'+x1+' '+y1+' L'+x1+' '+my+' L'+x2+' '+my+' L'+x2+' '+y2,lx:x2,ly:my};
  }
  var edgeEls=[];
  edges.forEach(function(e){var r=route(e);var p=el('path',{d:r.d,class:'edge','marker-end':'url(#'+mid+')'});G.appendChild(p);edgeEls.push({a:e.from,b:e.to,el:p});
    if(e.label){var lines=Array.isArray(e.label)?e.label:[e.label];var wpx=Math.max.apply(null,lines.map(function(l){return l.length;}))*6.2;
      var lx=e.lx!==undefined?e.lx:Math.min(Math.max(r.lx,wpx/2+4),W-4-wpx/2);var baseY=e.ly!==undefined?e.ly:(r.straight?r.ly+4:r.ly-6);
      lines.forEach(function(ln,i){var t=el('text',{x:lx,y:baseY-(lines.length-1-i)*13,'text-anchor':e.anchor||'middle',class:'elabel'});t.textContent=ln;G.appendChild(t);edgeEls.push({a:e.from,b:e.to,el:t});});}});

  var nodeEls={},selected=null;
  function wrap(str,maxChars){var words=String(str).split(' '),lines=[],cur='';words.forEach(function(w){if((cur+' '+w).trim().length>maxChars){lines.push(cur.trim());cur=w;}else cur+=' '+w;});if(cur.trim())lines.push(cur.trim());return lines;}
  Object.keys(nodes).forEach(function(id){var n=nodes[id];
    if(n.junction){var jg=el('g',{class:'junction'});jg.appendChild(el('circle',{cx:n.x,cy:n.y,r:5}));if(n.label){var jt=el('text',{x:n.x+(n.labelSide==='left'?-10:10),y:n.y+4,'text-anchor':n.labelSide==='left'?'end':'start'});jt.textContent=n.label;jg.appendChild(jt);}G.appendChild(jg);nodeEls[id]={g:jg};return;}
    var kd=kinds[n.kind];
    var g=el('g',{class:'node'+(kd&&kd.fill?' filled':'')+(n.dashed?' dashed':''),tabindex:'0',role:'button','data-id':id});
    if(kd){g.style.setProperty('--fb-box-stroke',kd.color);g.style.setProperty('--fb-bar',kd.color);if(kd.fill){g.style.setProperty('--fb-box-fill',kd.color);g.style.setProperty('--fb-box-hover',kd.color);}else{g.style.setProperty('--fb-box-fill','color-mix(in oklab,'+kd.color+' 12%,transparent)');g.style.setProperty('--fb-box-hover','color-mix(in oklab,'+kd.color+' 22%,transparent)');}}
    g.appendChild(el('rect',{class:'box',x:n.x,y:n.y,width:n.w,height:n.h,rx:6}));
    var narrow=n.w<160;
    if(!narrow&&icons[n.icon]){var ic=el('g',{transform:'translate('+(n.x+10)+','+(n.y+(n.h-22)/2+22-(n.bar?4:0))+') scale('+(22/960)+')'});ic.appendChild(el('path',{class:'icon',d:icons[n.icon]}));g.appendChild(ic);}
    var tx=narrow?n.x+n.w/2:n.x+n.w/2+12, maxChars=Math.floor((narrow?n.w-12:n.w-44)/6.6);
    var titleLines=wrap(n.title,maxChars), textH=n.h-(n.bar?10:0), subEl=el('g',{});
    function drawText(sub){subEl.innerHTML='';var subLines=sub?wrap(sub,maxChars):[];var total=titleLines.length+subLines.length;var y0=n.y+textH/2-(total-1)*6.5+4;
      titleLines.forEach(function(ln,i){var t=el('text',{x:tx,y:y0+i*13,'text-anchor':'middle','font-weight':'600'});t.textContent=ln;subEl.appendChild(t);});
      subLines.forEach(function(ln,i){var t=el('text',{x:tx,y:y0+(titleLines.length+i)*13,'text-anchor':'middle',class:'count'});t.textContent=ln;subEl.appendChild(t);});}
    drawText(n.sub||''); g.appendChild(subEl);
    var barEl=null,ghostEl=null;
    if(n.bar){g.appendChild(el('rect',{class:'vtrack',x:n.x+2,y:n.y+n.h-8,width:n.w-4,height:6,rx:1}));barEl=el('rect',{class:'vbar',x:n.x+2,y:n.y+n.h-8,width:0,height:6,rx:1});ghostEl=el('rect',{class:'vghost',x:n.x+2,y:n.y+n.h-10,width:2,height:10,opacity:0});g.appendChild(barEl);g.appendChild(ghostEl);var pinEl=el('rect',{class:'vpin',x:n.x+2,y:n.y+n.h-8,width:0,height:6,rx:1,opacity:0});g.appendChild(pinEl);}
    g.setAttribute('aria-label',n.title+(n.sub?' '+n.sub:''));
    g.addEventListener('mouseenter',function(){highlight(id);});g.addEventListener('mouseleave',function(){highlight(selected);});
    g.addEventListener('focus',function(){highlight(id);});g.addEventListener('blur',function(){highlight(selected);});
    g.addEventListener('click',function(){select(id);});
    g.addEventListener('keydown',function(ev){
      if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();select(id);return;}
      var target=null;
      if(ev.key==='ArrowUp')target=through(parents[id],parents)[0];
      else if(ev.key==='ArrowDown')target=through(children[id],children)[0];
      else if(ev.key==='ArrowLeft'||ev.key==='ArrowRight'){var row=Object.keys(nodes).filter(function(k){return !nodes[k].junction&&Math.abs(nodes[k].y-n.y)<20;}).sort(function(a,b){return nodes[a].x-nodes[b].x;});var i=row.indexOf(id);target=row[i+(ev.key==='ArrowRight'?1:-1)];}
      else if(ev.key==='Escape'){selected=null;activeKinds={};lg.querySelectorAll('.chip.kind').forEach(function(b){b.setAttribute('aria-pressed','false');b.classList.remove('on');});applyKinds();Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.remove('selected');});highlight(null);resetPanel();return;}
      if(target){ev.preventDefault();nodeEls[target].g.focus();select(target);}
    });
    G.appendChild(g); nodeEls[id]={g:g,drawText:drawText,barEl:barEl,ghostEl:ghostEl,pinEl:barEl?g.querySelector('.vpin'):null,lastFrac:null,samples:(n.samples||[id,n.title]).map(function(x){return String(x).toLowerCase();})};
  });
  function highlight(id){
    if(!id){Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.remove('faded');});edgeEls.forEach(function(e){e.el.classList.remove('faded');});return;}
    var up=upstream(id);
    Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.toggle('faded',!up[k]);});
    edgeEls.forEach(function(e){e.el.classList.toggle('faded',!(up[e.a]&&up[e.b]));});
  }

  // ----- panel
  function resetPanel(){panel.innerHTML='';panel.appendChild(h('h5',null,'Click a box'));panel.appendChild(h('p',null,'Each box opens here with what it is, its numbers, what feeds it, and the report panels that show it. Hover a box first to see what feeds it.'));}
  resetPanel();
  var numRe=/^\s*([-\d._]+(?:\.\d+)?)\s*(?:±\s*([\d._]+(?:\.\d+)?))?\s*(%?)/;
  function parseNum(v){var m=numRe.exec(String(v));if(!m)return null;var mean=parseFloat(m[1].replace(/_/g,''));if(isNaN(mean))return null;return {mean:mean,sd:m[2]!==undefined?parseFloat(m[2].replace(/_/g,'')):null,pct:m[3]==='%'};}
  function kindOfLabel(label){var l=label.toLowerCase();var map=spec.classOf||{};for(var name in map){if(l.indexOf(name.toLowerCase())===0)return map[name];}for(var k in kinds){if(l.indexOf(k.toLowerCase())===0)return k;}return null;}
  function factsTable(facts,pinFacts){
    var pinMap={};(pinFacts||[]).forEach(function(f){pinMap[f[0]]=f[1];});
    var parsed=facts.map(function(f){return parseNum(f[1]);});
    var barable=parsed.filter(function(p){return p&&(p.sd!==null||p.pct);});
    var scale=null; if(barable.length>=2){scale=barable.some(function(p){return p.pct;})?100:Math.max.apply(null,barable.map(function(p){return p.mean+(p.sd||0);}));}
    var t=h('table','facts');
    facts.forEach(function(f,i){var tr=h('tr'),k=h('td','k',f[0]),v=h('td','v',f[1]);var kd=kindOfLabel(f[0]);if(kd&&kinds[kd]){k.style.setProperty('--k',kinds[kd].color);k.innerHTML='<span class="fb-kind" style="margin:0 6px 0 0"><i></i></span>'+f[0];}
      var p=parsed[i];
      if(!p&&String(f[1]).length>24){ // long text value: label on one line, the whole value on its own full-width line below
        k.colSpan=2;k.className='k wide';tr.appendChild(k);t.appendChild(tr);var tr2=h('tr');var v2=h('td','v long',f[1]);v2.colSpan=2;tr2.appendChild(v2);
        if(pinFacts&&pinMap[f[0]]!==undefined&&pinMap[f[0]]!==f[1]){var v3=h('td','v long pv','pinned: '+pinMap[f[0]]);v3.colSpan=2;var tr3=h('tr');tr3.appendChild(v3);t.appendChild(tr2);t.appendChild(tr3);return;}
        t.appendChild(tr2);return;}
      tr.appendChild(k);tr.appendChild(v);
      if(pinFacts){var pv=h('div','pv','pinned: '+(pinMap[f[0]]!==undefined?pinMap[f[0]]:'·'));v.appendChild(pv);}if(scale&&p&&(p.sd!==null||p.pct)){var b=h('td','b');var bar=h('div','mbar');if(kd&&kinds[kd])bar.style.setProperty('--k',kinds[kd].color);var fill=h('i');fill.style.width=Math.min(100,100*p.mean/scale)+'%';bar.appendChild(fill);
        if(p.sd){var lo=Math.max(0,p.mean-p.sd),hi=Math.min(scale,p.mean+p.sd);var s=h('s');s.style.left=(100*lo/scale)+'%';s.style.width=(100*(hi-lo)/scale)+'%';bar.appendChild(s);var m=h('b');m.style.left=(100*p.mean/scale)+'%';bar.appendChild(m);}
        var pp=pinFacts&&pinMap[f[0]]!==undefined?parseNum(pinMap[f[0]]):null; if(pp){var pb=h('u');pb.style.width=Math.min(100,100*pp.mean/scale)+'%';bar.appendChild(pb);}
        b.appendChild(bar);tr.appendChild(b);} else if(scale){tr.appendChild(h('td','b'));}
      t.appendChild(tr);});
    if(scale){var cap=h('div','fb-note',(scale===100?'bars: percent of 100':'bars: mean, with a line for ± SD; full width = '+(Math.round(scale*1000)/1000)+' (the largest mean + SD in this list)')+(pinFacts?'; hollow bar and second column: the pinned option':''));var wrapd=h('div');wrapd.appendChild(t);wrapd.appendChild(cap);return wrapd;}
    return t;
  }
  function navChips(ids){var d=h('div','fb-chips');ids.forEach(function(id){var b=h('button','chip',nodes[id].title);b.type='button';var kd=kinds[nodes[id].kind];if(kd)b.style.borderColor=kd.color;b.addEventListener('click',function(){select(id);nodeEls[id].g.focus();});d.appendChild(b);});return d;}
  function select(id,fromHash){
    if(nodes[id].junction)return;
    selected=id;Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.toggle('selected',k===id);});highlight(id);
    var n=nodes[id],d=details[id]||{},o=option||{};
    panel.innerHTML='';
    var kd=kinds[n.kind];if(kd){var kc=h('div','fb-kind');kc.style.setProperty('--k',kd.color);kc.innerHTML='<i></i>'+kd.label;panel.appendChild(kc);}
    panel.appendChild(h('h5',null,d.title||n.title));
    var count=(o.subs&&o.subs[id])||d.count||n.sub||'';if(count)panel.appendChild(h('p','n',count));
    if(d.text)panel.appendChild(h('p',null,d.text));
    var facts=((o.facts&&o.facts[id])||[]).concat(d.facts||[]);
    var pinFacts=(pinned&&pinned!==option&&pinned.facts&&pinned.facts[id])||null;
    if(facts.length){panel.appendChild(h('div','fb-sec','numbers'+(option?' · '+option.label:'')+(pinFacts?' · vs pinned '+pinned.label:'')));panel.appendChild(factsTable(facts,pinFacts));}
    var fedBy=through(parents[id],parents),feeds=through(children[id],children);
    if(fedBy.length){panel.appendChild(h('div','fb-sec','made from'));panel.appendChild(navChips(fedBy));}
    if(feeds.length){panel.appendChild(h('div','fb-sec','goes into'));panel.appendChild(navChips(feeds));}
    var links=((o.links&&o.links[id])||[]).concat(d.links||[]);
    if(links.length){panel.appendChild(h('div','fb-sec','in the report'));var L=h('div','fb-chips');links.forEach(function(nl){var a=h('a','chip','→ '+nl[0]);a.href='#'+nl[1];L.appendChild(a);});panel.appendChild(L);}
    if(!fromHash)writeHash();
  }
  function applyOption(){
    if(!option)return;
    Object.keys(option.subs||{}).forEach(function(id){var ne=nodeEls[id];if(!ne||!ne.drawText)return;ne.drawText(option.subs[id]);ne.g.setAttribute('aria-label',nodes[id].title+' '+option.subs[id]);});
    Object.keys(option.bars||{}).forEach(function(id){var ne=nodeEls[id];if(!ne||!ne.barEl)return;var w=nodes[id].w-4,frac=option.bars[id];
      if(ne.lastFrac!==null&&prevOption!==null){ne.ghostEl.setAttribute('x',nodes[id].x+2+w*Math.min(1,ne.lastFrac)-1);ne.ghostEl.setAttribute('opacity',1);}
      ne.barEl.setAttribute('width',Math.max(0,w*Math.min(1,frac)));ne.lastFrac=frac;});
    Object.keys(nodeEls).forEach(function(id){var ne=nodeEls[id];if(!ne.pinEl)return;var w=nodes[id].w-4;var pf=pinned&&pinned!==option&&pinned.bars?pinned.bars[id]:undefined;
      if(pf===undefined){ne.pinEl.setAttribute('opacity',0);}else{ne.pinEl.setAttribute('width',Math.max(0,w*Math.min(1,pf)));ne.pinEl.setAttribute('opacity',1);}});
    if(selected)select(selected,true);
  }
  applyOption();
  if(spec.footnote)view.appendChild(h('p','fb-foot',spec.footnote));

  // ----- 2. MultiQC toolbox: highlight and hide samples, matched against each box's sample names
  var hlState={texts:[],cols:[],regex:false}, hideState={texts:[],regex:false};
  function matches(samples,texts,regex){for(var i=0;i<texts.length;i++){var t=String(texts[i]);if(!t)continue;for(var j=0;j<samples.length;j++){try{if(regex?new RegExp(t,'i').test(samples[j]):samples[j].indexOf(t.toLowerCase())>=0)return i;}catch(e){}}}return -1;}
  function applyToolbox(){var any=hlState.texts.some(function(t){return t;});
    Object.keys(nodeEls).forEach(function(id){var ne=nodeEls[id];if(!ne.samples)return;var g=ne.g;
      var hi=matches(ne.samples,hideState.texts,hideState.regex)>=0; g.classList.toggle('mqc-hidden',hi);
      var m=any?matches(ne.samples,hlState.texts,hlState.regex):-1;
      g.classList.toggle('mqc-hl',m>=0); g.classList.toggle('mqc-dim',any&&m<0); g.style.setProperty('--fb-hl',m>=0?(hlState.cols[m]||'#d97b00'):'transparent');});}
  function onHighlights(texts,cols,regex){hlState={texts:texts||[],cols:cols||[],regex:!!regex};applyToolbox();}
  function onHide(texts,regex){hideState={texts:texts||[],regex:!!regex};applyToolbox();}
  if(global.jQuery){global.jQuery(document).on('mqc_highlights',function(e,t,c,r){onHighlights(t,c,r);});global.jQuery(document).on('mqc_hidesamples',function(e,t,r){onHide(t,r);});}
  document.addEventListener('flow:highlight',function(e){onHighlights(e.detail.texts,e.detail.cols,e.detail.regex);});
  document.addEventListener('flow:hide',function(e){onHide(e.detail.texts,e.detail.regex);});
  if(global.mqc_highlight_f_texts&&global.mqc_highlight_f_texts.length)onHighlights(global.mqc_highlight_f_texts,global.mqc_highlight_f_cols,false);
  if(global.mqc_hide_f_texts&&global.mqc_hide_f_texts.length)onHide(global.mqc_hide_f_texts,false);

  // ----- 4. export: the SVG as it is now, computed styles written onto every element
  var STYLE_PROPS=['fill','stroke','stroke-width','stroke-dasharray','opacity','font-family','font-size','font-weight','letter-spacing','font-variant-numeric'];
  function exportSvg(){var clone=svg.cloneNode(true);var src=svg.querySelectorAll('*'),dst=clone.querySelectorAll('*');
    for(var i=0;i<src.length;i++){var cs=getComputedStyle(src[i]);STYLE_PROPS.forEach(function(p){var v=cs.getPropertyValue(p);if(v&&v!=='none'||p==='fill')dst[i].style.setProperty(p,v);});
      if(dst[i].tagName==='text'&&cs.textTransform==='uppercase')dst[i].textContent=dst[i].textContent.toUpperCase();
      if(dst[i].tagName==='g'&&(dst[i].classList.contains('faded')||dst[i].classList.contains('mqc-dim')))dst[i].style.opacity=cs.opacity;}
    var vb=svg.getAttribute('viewBox').split(' ');clone.setAttribute('xmlns',NS);clone.setAttribute('width',vb[2]);clone.setAttribute('height',vb[3]);clone.removeAttribute('class');
    var bgc=getComputedStyle(document.body).backgroundColor;if(!bgc||bgc==='rgba(0, 0, 0, 0)')bgc='#ffffff';var bg=el('rect',{x:0,y:0,width:vb[2],height:vb[3],fill:bgc});clone.insertBefore(bg,clone.firstChild);
    var ttl=el('title',{});ttl.textContent=(spec.title||'data flow')+(option?' — '+spec.control.label+': '+option.label:'');clone.insertBefore(ttl,clone.firstChild);
    return '<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(clone);}
  function fname(){return ((root.id||'flow')+(option?'.'+option.id:'')+'.svg').replace(/[^\w.-]+/g,'_');}
  dl.addEventListener('click',function(){var blob=new Blob([exportSvg()],{type:'image/svg+xml'});var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=fname();document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},1000);});
  cp.addEventListener('click',function(){var txt=exportSvg();var done=function(ok){cp.textContent=ok?'copied ✓':'copy failed';setTimeout(function(){cp.textContent='copy SVG';},1500);};
    if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(txt).then(function(){done(true);},function(){done(false);});else done(false);});

  // ----- 1. open on the state in the URL, and follow it if it changes
  applyHash(); global.addEventListener('hashchange',applyHash);
}
global.renderFlow=renderFlow;
global.renderAllFlows=function(){document.querySelectorAll('.flow-block').forEach(function(r){if(!r.dataset.rendered){r.dataset.rendered='1';renderFlow(r);}});};
if(document.readyState!=='loading')global.renderAllFlows();else document.addEventListener('DOMContentLoaded',global.renderAllFlows);
})(window);
""" + "</script>"


def block(spec: dict, *, uid: str = "flow-block") -> str:
    """The overview's drawing as one HTML block: the spec as JSON, the renderer, and the
    container it fills. Legend, control, diagram, panel and footnote are all drawn by the
    script from the spec, so what the reader sees is exactly what the spec says."""
    payload = dict(spec)
    payload["icons"] = ICONS
    payload.setdefault("iconLegend", ICON_LEGEND)
    return (CSS + f'<div id="{uid}" class="flow-block"><div class="view"></div>'
            + f'<script type="application/json">{json.dumps(payload)}</script></div>' + JS)


ICON_LEGEND = ("DNA: query proteins; cylinder: target database; magnifier: a search; rows: a table; "
               "cube: predicted structures; sheet: the report")
