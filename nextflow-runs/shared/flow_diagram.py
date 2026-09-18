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
  nodes: {id: {x, y, w, h, icon, kind?, title, sub?, bar?, strip?} | {junction: true, x, y,
          label?, labelSide?}}
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
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "report_header_info:\n" + "".join(f"  - {q(k)}: {q(v)}\n" for k, v in items)


CSS = """
<style>
.flow-block .grid{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:16px;align-items:start}
@media (max-width:820px){.flow-block .grid{grid-template-columns:1fr}}
.flow-block .warn{border:1px solid %(open)s;border-radius:6px;padding:8px 12px;margin:0 0 12px;font-size:13px}
.flow-block .legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;margin-bottom:10px}
.flow-block .legend span{display:inline-flex;align-items:center;gap:7px}
.flow-block .sw{width:22px;height:12px;border:1.5px solid currentColor;border-radius:2px;display:inline-block;flex:none;position:relative}
.flow-block .sw.faded{opacity:.25}
.flow-block .sw.bar::after{content:"";position:absolute;left:1px;right:40%%;bottom:1px;height:4px;background:currentColor;opacity:.8}
.flow-block .sw.ghost::after{content:"";position:absolute;left:60%%;bottom:0;width:2px;height:8px;background:%(open)s}
.flow-block .dot{width:10px;height:10px;border-radius:50%%;background:currentColor;display:inline-block;flex:none;margin:0 6px}
.flow-block .arrow{width:22px;height:12px;display:inline-block;flex:none}
.flow-block .diagram{border:1px solid rgba(127,127,127,.3);border-radius:8px;padding:8px;overflow-x:auto}
.flow-block svg.flow{display:block;width:100%%;max-width:1000px;margin:0 auto;font-family:inherit;font-size:12px;color:inherit}
.flow-block .lane{fill:currentColor;opacity:.05}
.flow-block .lanelabel{fill:currentColor;opacity:.65;font-size:10px;letter-spacing:.08em;text-transform:uppercase}
.flow-block .node{cursor:pointer;transition:opacity .15s}
.flow-block .node rect.box{fill:none;stroke:currentColor;stroke-width:1.5}
.flow-block .node.filled text,.flow-block .node.filled .icon{fill:#fff}
.flow-block .node text{fill:currentColor}
.flow-block .node text.count{font-family:monospace;font-size:11px}
.flow-block .node .icon{fill:currentColor}
.flow-block .node:focus{outline:none}
.flow-block .node:focus-visible rect.box,.flow-block .node.selected rect.box{stroke-width:3;stroke:%(open)s !important}
.flow-block .junction circle{fill:currentColor}
.flow-block .junction text{fill:currentColor;opacity:.7;font-size:11px}
.flow-block .edge{stroke:currentColor;stroke-width:1.2;fill:none;transition:opacity .15s}
.flow-block .elabel{fill:currentColor;opacity:.75;font-size:11px;transition:opacity .15s}
.flow-block .faded{opacity:.22}
.flow-block .vbar{fill:currentColor;opacity:.8;transition:width .4s ease}
.flow-block .filled .vbar{fill:#fff;opacity:.6}
.flow-block .vghost{fill:%(open)s;transition:x .4s ease,opacity .4s}
.flow-block .panel{border:1px solid rgba(127,127,127,.3);border-radius:8px;padding:12px 14px;position:sticky;top:12px}
.flow-block .panel h5{margin:0 0 4px;font-size:15px;font-weight:600}
.flow-block .panel .n{font-family:monospace;font-size:13px;margin:0 0 8px;opacity:.85}
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
.flow-block .seg button:focus-visible{outline:2px solid %(open)s;outline-offset:2px}
.flow-block .foot{font-size:12px;opacity:.75;margin-top:10px}
@media (prefers-reduced-motion: reduce){.flow-block .node,.flow-block .edge,.flow-block .elabel,.flow-block .vbar,.flow-block .vghost{transition:none}}
</style>
"""

# The renderer, ported from the 2026-09-18 mockup (kmerseek-report-flows v2). Everything it
# reads is the spec; the control's options carry their numbers, so no code is per report.
JS = r"""
<script>
(function(){
  var root=document.getElementById('%(uid)s'); if(!root) return;
  var spec=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  var icons=spec.icons;
  var OX=34, W=760;
  var view=root.querySelector('.view');
  var NS='http://www.w3.org/2000/svg';
  function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e;}
  function h(t,cls,text){var e=document.createElement(t);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}

  if(spec.warning){view.appendChild(h('p','warn',spec.warning));}
  var option=(spec.control&&spec.control.options.length)?spec.control.options[0]:null, prevOption=null;
  if(spec.control){
    var c=h('div','control'); var lab=h('label',null,spec.control.label); c.appendChild(lab);
    var seg=h('div','seg'); seg.setAttribute('role','group'); seg.setAttribute('aria-label',spec.control.label);
    spec.control.options.forEach(function(o,i){var b=h('button',null,o.label);b.type='button';b.setAttribute('data-id',o.id);b.setAttribute('aria-pressed',i===0?'true':'false');seg.appendChild(b);});
    seg.addEventListener('click',function(e){var b=e.target.closest('button');if(!b||b.getAttribute('data-id')===(option&&option.id))return;
      seg.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');
      prevOption=option;option=spec.control.options.filter(function(o){return o.id===b.getAttribute('data-id');})[0];applyOption();});
    c.appendChild(seg); view.appendChild(c);
  }

  // Legend: the fixed grammar first, then this report's box colours, then the marks the
  // page adds (bar, ghost tick, faded, open).
  var lg=h('div','legend'); lg.setAttribute('aria-label','legend');
  function item(sw,text){var s=document.createElement('span');s.innerHTML=sw+text;lg.appendChild(s);}
  item('<i class="sw"></i>','a set of sequences or results, with its count');
  item('<svg class="arrow" viewBox="0 0 22 12"><line x1="0" y1="6" x2="16" y2="6" stroke="currentColor" stroke-width="1.5"/><path d="M15 2 L21 6 L15 10 z" fill="currentColor"/></svg>','a step, labelled with what it does');
  item('<i class="dot"></i>','two inputs meet here and go on together');
  Object.keys(spec.kinds||{}).forEach(function(k){var kd=spec.kinds[k];item('<i class="sw" style="border-color:'+kd.color+';'+(kd.fill?'background:'+kd.color+';':'')+'border-width:2px"></i>',kd.label);});
  item('<i class="sw" style="border-style:dashed"></i>','dashed: an arm this run did not do');
  item('<span style="opacity:.8">icons</span>','DNA: query proteins; cylinder: target database; magnifier: a search; rows: a table; cube: predicted structures; sheet: the report');
  if(spec.barLegend) item('<i class="sw bar"></i>',spec.barLegend);
  if(spec.control) item('<i class="sw ghost"></i>','orange tick: where the bar was before the last click');
  item('<i class="sw faded"></i>','faded: not upstream of the box under the pointer');
  item('<i class="sw" style="border-color:%(open)s;border-width:2.5px"></i>','orange edge: the box whose details are open');
  view.appendChild(lg);

  var grid=h('div','grid'); var dwrap=h('div','diagram');
  var svg=el('svg',{class:'flow',viewBox:'0 0 '+(W+OX)+' '+spec.height,role:'img','aria-label':spec.title||'data flow'});
  var defs=el('defs',{}); var m=el('marker',{id:'ah-%(uid)s',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'7',markerHeight:'7',orient:'auto-start-reverse'});
  m.appendChild(el('path',{d:'M0 0 L10 5 L0 10 z',fill:'currentColor'})); defs.appendChild(m); svg.appendChild(defs);
  dwrap.appendChild(svg); grid.appendChild(dwrap);
  var panel=h('aside','panel'); panel.setAttribute('aria-live','polite');
  panel.innerHTML='<h5 class="p-title">Click a box</h5><p class="n p-count"></p><p class="p-text">Each box opens here with what it is, how it was made, and which panels below show it.</p><dl class="p-facts"></dl><div class="links p-links"></div>';
  grid.appendChild(panel); view.appendChild(grid);
  if(spec.footnote){view.appendChild(h('p','foot',spec.footnote));}

  // Row lanes first, so they sit behind everything; every other lane is shaded.
  (spec.lanes||[]).forEach(function(l,i){var y0=l[0],y1=l[1];
    if(i%%2===0) svg.appendChild(el('rect',{class:'lane',x:0,y:y0,width:W+OX,height:y1-y0}));
    var t=el('text',{class:'lanelabel',x:12,y:(y0+y1)/2,'text-anchor':'middle',transform:'rotate(-90 12 '+((y0+y1)/2)+')'});t.textContent=l[2];svg.appendChild(t);});
  var G=el('g',{transform:'translate('+OX+',0)'}); svg.appendChild(G);
  (spec.headers||[]).forEach(function(hd){var t=el('text',{x:hd[0],y:hd[1],'text-anchor':'middle','font-weight':'600',fill:'currentColor'});t.textContent=hd[2];G.appendChild(t);});

  var nodes=spec.nodes, edges=spec.edges;
  var parents={}; edges.forEach(function(e){(parents[e.to]=parents[e.to]||[]).push(e.from);});
  function upstream(id){var s={};s[id]=1;var st=[id];while(st.length){var n=st.pop();(parents[n]||[]).forEach(function(p){if(!s[p]){s[p]=1;st.push(p);}});}return s;}
  function isJ(id){return !!nodes[id].junction;}
  function cx(n){return n.junction?n.x:n.x+n.w/2;} function bottom(n){return n.junction?n.y+5:n.y+n.h;} function top(n){return n.junction?n.y-5:n.y;}

  // Routing: a straight drop when the source sits over the target; otherwise down, across,
  // down. Into a junction from a box: down to the junction's row, then across into it. Out
  // of a junction: down to a bus a little below it, across, down.
  function route(e){
    var A=nodes[e.from],B=nodes[e.to];
    if(B.junction&&A.junction){return {d:'M'+A.x+' '+(A.y+5)+' L'+A.x+' '+B.y+' L'+(B.x+(A.x<B.x?-6:6))+' '+B.y,lx:(A.x+B.x)/2,ly:B.y-6};}
    if(B.junction){var x1=cx(A); if(Math.abs(x1-B.x)<1) return {d:'M'+x1+' '+bottom(A)+' L'+x1+' '+(B.y-6),lx:x1,ly:(bottom(A)+B.y)/2};
      if(Math.abs(A.y+A.h/2-B.y)<A.h/2){var side=x1<B.x?A.x+A.w:A.x; return {d:'M'+side+' '+B.y+' L'+(B.x+(x1<B.x?-6:6))+' '+B.y,lx:(side+B.x)/2,ly:B.y-8};}
      return {d:'M'+x1+' '+bottom(A)+' L'+x1+' '+B.y+' L'+(B.x+(x1<B.x?-6:6))+' '+B.y,lx:x1,ly:(bottom(A)+B.y)/2};}
    var x1=cx(A),x2=cx(B),y1=bottom(A),y2=top(B);
    if(!A.junction&&x1>=B.x&&x1<=B.x+B.w) return {d:'M'+x1+' '+y1+' L'+x1+' '+y2,lx:x1,ly:(y1+y2)/2,straight:true};
    var my=A.junction?(A.y+(e.bus||22)):(e.bus!==undefined?e.bus:(y1+y2)/2);
    return {d:'M'+x1+' '+y1+' L'+x1+' '+my+' L'+x2+' '+my+' L'+x2+' '+y2,lx:x2,ly:my};
  }
  var edgeEls=[];
  edges.forEach(function(e){
    var r=route(e);
    var pth=el('path',{d:r.d,class:'edge','marker-end':'url(#ah-%(uid)s)'}); G.appendChild(pth); edgeEls.push({a:e.from,b:e.to,el:pth});
    if(e.label){var lines=Array.isArray(e.label)?e.label:[e.label];
      var wpx=Math.max.apply(null,lines.map(function(l){return l.length;}))*6.2;
      // Beside the line, never on it: a label on a straight drop sits to the right of
      // the line, or to the left when it would run off the canvas.
      var anchor=e.anchor||'middle', lx;
      if(e.lx!==undefined){lx=e.lx;}
      else if(r.straight){ if(r.lx+8+wpx<=W-4){lx=r.lx+8;anchor='start';} else if(r.lx-8-wpx>=4){lx=r.lx-8;anchor='end';} else {lx=Math.min(Math.max(r.lx,wpx/2+4),W-4-wpx/2);} }
      else {lx=Math.min(Math.max(r.lx,wpx/2+4),W-4-wpx/2);}
      var baseY=e.ly!==undefined?e.ly:(r.straight?r.ly+4:r.ly-6);
      lines.forEach(function(ln,i){var t=el('text',{x:lx,y:baseY-(lines.length-1-i)*13,'text-anchor':anchor,class:'elabel'});t.textContent=ln;G.appendChild(t);edgeEls.push({a:e.from,b:e.to,el:t});});}
  });

  var nodeEls={}, selected=null;
  function wrap(str,maxChars){var words=str.split(' '),lines=[],cur='';words.forEach(function(w){if((cur+' '+w).trim().length>maxChars){lines.push(cur.trim());cur=w;}else cur+=' '+w;});if(cur.trim())lines.push(cur.trim());return lines;}
  Object.keys(nodes).forEach(function(id){var n=nodes[id];
    if(n.junction){var g=el('g',{class:'junction'});g.appendChild(el('circle',{cx:n.x,cy:n.y,r:5}));
      if(n.label){var t=el('text',{x:n.x+(n.labelSide==='left'?-10:10),y:n.y+4,'text-anchor':n.labelSide==='left'?'end':'start'});t.textContent=n.label;g.appendChild(t);}
      G.appendChild(g);nodeEls[id]={g:g};return;}
    var kind=spec.kinds&&spec.kinds[n.kind];
    var g=el('g',{class:'node'+(kind&&kind.fill?' filled':''),tabindex:'0',role:'button','data-id':id});
    var box=el('rect',{class:'box',x:n.x,y:n.y,width:n.w,height:n.h,rx:4});
    if(kind){box.style.stroke=kind.color;box.style.strokeWidth=2;if(kind.fill)box.style.fill=kind.color;}
    if(n.dashed){box.setAttribute('stroke-dasharray','6 4');}
    g.appendChild(box);
    if(n.strip){g.appendChild(el('rect',{x:n.x+1,y:n.y+n.h-7,width:n.w-2,height:6,fill:n.strip}));}
    var narrow=n.w<160;
    if(!narrow&&icons[n.icon]){var ic=el('g',{transform:'translate('+(n.x+10)+','+(n.y+(n.h-22)/2+22-(n.bar?4:0))+') scale('+(22/960)+')'});ic.appendChild(el('path',{class:'icon',d:icons[n.icon]}));g.appendChild(ic);}
    var tx=narrow?n.x+n.w/2:n.x+n.w/2+12, maxChars=Math.floor((narrow?n.w-12:n.w-44)/6.6);
    var titleLines=wrap(n.title,maxChars), textH=n.h-(n.bar?10:0);
    var subEl=el('g',{'data-sub':id});
    function drawText(sub){subEl.innerHTML='';var subLines=sub?wrap(sub,maxChars):[];var total=titleLines.length+subLines.length;var y0=n.y+textH/2-(total-1)*6.5+4;
      titleLines.forEach(function(ln,i){var t=el('text',{x:tx,y:y0+i*13,'text-anchor':'middle','font-weight':'600'});t.textContent=ln;subEl.appendChild(t);});
      subLines.forEach(function(ln,i){var t=el('text',{x:tx,y:y0+(titleLines.length+i)*13,'text-anchor':'middle',class:'count'});t.textContent=ln;subEl.appendChild(t);});}
    drawText(n.sub||''); g.appendChild(subEl);
    var barEl=null,ghostEl=null;
    if(n.bar){barEl=el('rect',{class:'vbar',x:n.x+2,y:n.y+n.h-8,width:0,height:6});ghostEl=el('rect',{class:'vghost',x:n.x+2,y:n.y+n.h-10,width:2,height:10,opacity:0});g.appendChild(barEl);g.appendChild(ghostEl);}
    g.setAttribute('aria-label',n.title+(n.sub?' '+n.sub:''));
    g.addEventListener('mouseenter',function(){highlight(id);});g.addEventListener('mouseleave',function(){highlight(selected);});
    g.addEventListener('focus',function(){highlight(id);});g.addEventListener('blur',function(){highlight(selected);});
    g.addEventListener('click',function(){select(id);});g.addEventListener('keydown',function(ev){if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();select(id);}});
    G.appendChild(g); nodeEls[id]={g:g,drawText:drawText,barEl:barEl,ghostEl:ghostEl,lastFrac:null};
  });

  function highlight(id){
    if(!id){Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.remove('faded');});edgeEls.forEach(function(e){e.el.classList.remove('faded');});return;}
    var up=upstream(id);
    Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.toggle('faded',!up[k]);});
    edgeEls.forEach(function(e){e.el.classList.toggle('faded',!(up[e.a]&&up[e.b]));});
  }
  function opt(key,id){return option&&option[key]&&option[key][id]!==undefined?option[key][id]:null;}
  function select(id){
    if(isJ(id))return;
    selected=id;Object.keys(nodeEls).forEach(function(k){nodeEls[k].g.classList.toggle('selected',k===id);});highlight(id);
    var d=spec.details[id]||{};
    panel.querySelector('.p-title').textContent=nodes[id].title;
    var sub=opt('subs',id); panel.querySelector('.p-count').textContent=sub!==null?sub:(nodes[id].sub||'');
    panel.querySelector('.p-text').textContent=d.text||'';
    var facts=(opt('facts',id)||[]).concat(d.facts||[]), links=opt('links',id)||d.links||[];
    var dl=panel.querySelector('.p-facts');dl.innerHTML='';facts.forEach(function(kv){var dt=document.createElement('dt');dt.textContent=kv[0];var dd=document.createElement('dd');dd.textContent=kv[1];dl.appendChild(dt);dl.appendChild(dd);});
    var L=panel.querySelector('.p-links');L.innerHTML='';links.forEach(function(nl){var a=document.createElement('a');a.href='#'+nl[1];a.textContent='→ '+nl[0];L.appendChild(a);});
  }
  function applyOption(){
    if(!option)return;
    Object.keys(option.subs||{}).forEach(function(id){var ne=nodeEls[id];if(!ne||!ne.drawText)return;ne.drawText(option.subs[id]);ne.g.setAttribute('aria-label',nodes[id].title+' '+option.subs[id]);});
    Object.keys(option.bars||{}).forEach(function(id){var ne=nodeEls[id];if(!ne||!ne.barEl)return;var w=nodes[id].w-4;var frac=option.bars[id];
      if(ne.lastFrac!==null&&prevOption!==null){ne.ghostEl.setAttribute('x',nodes[id].x+2+w*ne.lastFrac-1);ne.ghostEl.setAttribute('opacity',1);}
      ne.barEl.setAttribute('width',Math.max(0,w*Math.min(1,frac)));ne.lastFrac=frac;});
    if(selected)select(selected);
  }
  applyOption();
})();
</script>
"""


def block(spec: dict, *, uid: str = "flow-block") -> str:
    """The overview's drawing as one HTML block: the spec as JSON, the renderer, and the
    container it fills. Legend, control, diagram, panel and footnote are all drawn by the
    script from the spec, so what the reader sees is exactly what the spec says."""
    payload = dict(spec)
    payload["icons"] = ICONS
    return (
        CSS % {"open": C_OPEN}
        + f'<div id="{uid}" class="flow-block"><div class="view"></div>'
        + f'<script type="application/json">{json.dumps(payload)}</script>'
        + JS % {"uid": uid, "open": C_OPEN}
        + '</div>')
