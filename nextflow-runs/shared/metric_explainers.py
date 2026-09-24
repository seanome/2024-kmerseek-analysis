"""Small pictures and widgets that explain a metric, for the MultiQC report overviews.

Each explainer is one HTML block: a title, a definition in one or two sentences, a
drawing, and where the reader can act on it (a slider, a pair of buttons) the numbers
that change with it. A widget that needs example data uses a toy set and says so in its
first line; the report's own numbers never appear inside an explainer, so nothing here
can be mistaken for a result.

One script and one style, shared by every report, keyed by the widget kind:

  iou        overlap over union between a call and the true domain, with the call's
             ends on sliders and the 50% rule applied
  threshold  precision, recall and F1 as a score threshold moves over a toy set of
             calls, the precision-recall curve it traces, Fmax, the area under the
             curve (AUPRC) and recall at a precision floor
  reachable  which answer-key instances a target proteome can reach at all, the
             denominator of recall
  bh         Benjamini-Hochberg correction over a toy set of p-values, with precision
             and recall as the sweep report defines them
  fraction   a proteome split into placed and dark, and the mask pair

Colours: green is a true positive or a placed protein, red a false positive, grey the
dark set or a rejected call; the orange marks the threshold, a state of the page and
not a data colour. Every mark is in the block's own legend, above its drawing.
"""

import json

C_TP = "#0f9d76"
C_FP = "#c0392b"
C_NONE = "#7f7f7f"
C_MARK = "#d97b00"

CSS = """
<style>
.mx{border:1px solid rgba(127,127,127,.3);border-radius:8px;padding:12px 14px;margin:0 0 14px}
.mx h5{margin:0 0 4px;font-size:15px;font-weight:600}
.mx .def{margin:0 0 8px}
.mx .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;margin:0 0 8px}
.mx .legend span{display:inline-flex;align-items:center;gap:6px}
.mx .sw{width:18px;height:11px;border-radius:2px;display:inline-block;border:1.5px solid currentColor}
.mx svg{display:block;width:100%;max-width:720px;font-family:inherit;font-size:12px;color:inherit;overflow:visible}
.mx text{fill:currentColor}
.mx .axis{stroke:currentColor;stroke-width:1;opacity:.6}
.mx .ctrl{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;font-size:12.5px;margin:6px 0}
.mx .ctrl label{display:inline-flex;align-items:center;gap:8px}
.mx input[type=range]{width:180px}
.mx .seg{display:inline-flex;border:1px solid rgba(127,127,127,.5);border-radius:6px;overflow:hidden}
.mx .seg button{background:transparent;color:inherit;border:0;border-right:1px solid rgba(127,127,127,.5);padding:3px 10px;font:inherit;cursor:pointer}
.mx .seg button:last-child{border-right:0}
.mx .seg button[aria-pressed="true"]{background:%(mark)s;color:#fff}
.mx .out{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:12.5px;margin:6px 0 0}
.mx .out dt{opacity:.7}.mx .out dd{margin:0;font-family:monospace}
.mx .toy{font-size:12px;opacity:.75;margin:0 0 6px}
</style>
"""

JS = r"""
<script>
(function(){
  if(window.__mxLoaded) return; window.__mxLoaded=true;
  var NS='http://www.w3.org/2000/svg';
  function el(t,a,txt){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;return e;}
  function f3(x){return (Math.round(x*1000)/1000).toFixed(3);}
  function dl(root,pairs){var d=root.querySelector('.out');d.innerHTML='';pairs.forEach(function(p){var dt=document.createElement('dt');dt.textContent=p[0];var dd=document.createElement('dd');dd.textContent=p[1];d.appendChild(dt);d.appendChild(dd);});}

  // --- iou: a protein line, the true domain as a box, the call as a bar under it ------
  function iou(root,cfg){
    var svg=root.querySelector('svg'), L=cfg.length, X=function(aa){return 40+aa/L*640;};
    var s1=root.querySelector('[data-k=start]'), s2=root.querySelector('[data-k=end]');
    function draw(){
      var cs=+s1.value, ce=+s2.value; if(ce<cs+5){ce=cs+5;s2.value=ce;}
      svg.innerHTML='';
      svg.appendChild(el('line',{x1:X(0),y1:40,x2:X(L),y2:40,stroke:'currentColor','stroke-width':2}));
      svg.appendChild(el('text',{x:X(0),y:28},'protein, '+L+' aa'));
      svg.appendChild(el('rect',{x:X(cfg.dom[0]),y:30,width:X(cfg.dom[1])-X(cfg.dom[0]),height:20,fill:'none',stroke:'currentColor','stroke-width':2}));
      svg.appendChild(el('text',{x:X(cfg.dom[0]),y:66},'true domain '+cfg.dom[0]+'-'+cfg.dom[1]));
      var o0=Math.max(cs,cfg.dom[0]), o1=Math.min(ce,cfg.dom[1]), ov=Math.max(0,o1-o0);
      var un=(ce-cs)+(cfg.dom[1]-cfg.dom[0])-ov, v=ov/un, tp=v>=cfg.min;
      if(ov>0) svg.appendChild(el('rect',{x:X(o0),y:30,width:X(o1)-X(o0),height:20,fill:tp?'%(tp)s':'%(fp)s',opacity:.35}));
      svg.appendChild(el('rect',{x:X(cs),y:84,width:X(ce)-X(cs),height:12,rx:2,fill:tp?'%(tp)s':'%(fp)s'}));
      svg.appendChild(el('text',{x:X(cs),y:112},'the call '+cs+'-'+ce+(tp?': true positive':': false positive, right family in the wrong place')));
      dl(root,[['overlap',ov+' aa'],['union',un+' aa'],['overlap over union',f3(v)],['rule',(v>=cfg.min?'≥ ':'< ')+cfg.min+' → '+(tp?'true positive':'false positive')]]);
    }
    s1.addEventListener('input',draw); s2.addEventListener('input',draw); draw();
  }

  // --- threshold: a toy set of calls, precision/recall/F1 as the cutoff moves -------------
  function threshold(root,cfg){
    var svg=root.querySelector('svg'), sl=root.querySelector('[data-k=thr]');
    var calls=cfg.calls.slice().sort(function(a,b){return b.s-a.s;}), n=calls.length, nInst=cfg.instances;
    // the curve: every cutoff between consecutive scores
    var pts=[]; for(var k=1;k<=n;k++){var kept=calls.slice(0,k),tp=kept.filter(function(c){return c.tp;}).length; var found={};kept.forEach(function(c){if(c.tp)found[c.i]=1;}); var rec=Object.keys(found).length/nInst, pre=tp/k; pts.push({thr:calls[k-1].s,pre:pre,rec:rec,f1:pre+rec?2*pre*rec/(pre+rec):0,k:k});}
    var fmax=pts.reduce(function(m,p){return p.f1>m.f1?p:m;},pts[0]);
    // area under the PR curve, stepwise in recall
    var au=0,prev=0; pts.forEach(function(p){au+=(p.rec-prev)*p.pre;prev=p.rec;});
    var atFloor=pts.filter(function(p){return p.pre>=cfg.floor;}).reduce(function(m,p){return p.rec>m?p.rec:m;},0);
    var X=function(r){return 60+r*280;}, Y=function(p){return 160-p*130;};
    function draw(){
      var thr=+sl.value/100; svg.innerHTML='';
      // left: the calls on a score axis
      svg.appendChild(el('text',{x:400,y:22},'the '+n+' calls by score; cutoff at '+thr.toFixed(2)));
      svg.appendChild(el('line',{x1:400,y1:40,x2:400,y2:170,class:'axis'}));
      svg.appendChild(el('text',{x:404,y:44},'score 1'));svg.appendChild(el('text',{x:404,y:172},'score 0'));
      var yThr=170-thr*130; svg.appendChild(el('line',{x1:395,y1:yThr,x2:700,y2:yThr,stroke:'%(mark)s','stroke-width':2}));
      svg.appendChild(el('text',{x:704,y:yThr+4,fill:'%(mark)s'},'cutoff'));
      calls.forEach(function(c,j){var y=170-c.s*130, x=420+(j%9)*30; svg.appendChild(el('circle',{cx:x,cy:y,r:6,fill:c.tp?'%(tp)s':'%(fp)s',opacity:c.s>=thr?1:.25}));});
      // right: the PR curve
      svg.appendChild(el('line',{x1:X(0),y1:Y(0),x2:X(1),y2:Y(0),class:'axis'}));svg.appendChild(el('line',{x1:X(0),y1:Y(0),x2:X(0),y2:Y(1),class:'axis'}));
      svg.appendChild(el('text',{x:X(0.5),y:Y(0)+18,'text-anchor':'middle'},'recall: instances found / reachable instances'));
      svg.appendChild(el('text',{x:X(0)-8,y:Y(0.5),'text-anchor':'middle',transform:'rotate(-90 '+(X(0)-8)+' '+Y(0.5)+')'},'precision: true calls / calls kept'));
      var d='M'+X(0)+' '+Y(pts[0].pre); prev=0; pts.forEach(function(p){d+=' L'+X(prev)+' '+Y(p.pre)+' L'+X(p.rec)+' '+Y(p.pre);prev=p.rec;});
      svg.appendChild(el('path',{d:d+' L'+X(prev)+' '+Y(0)+' L'+X(0)+' '+Y(0)+' Z',fill:'currentColor',opacity:.08}));
      svg.appendChild(el('path',{d:d,fill:'none',stroke:'currentColor','stroke-width':1.5}));
      svg.appendChild(el('line',{x1:X(0),y1:Y(cfg.floor),x2:X(1),y2:Y(cfg.floor),stroke:'currentColor','stroke-dasharray':'4 3',opacity:.6}));
      svg.appendChild(el('text',{x:X(0)+4,y:Y(cfg.floor)-4},'precision floor '+cfg.floor));
      svg.appendChild(el('circle',{cx:X(fmax.rec),cy:Y(fmax.pre),r:5,fill:'none',stroke:'currentColor','stroke-width':2}));
      svg.appendChild(el('text',{x:X(fmax.rec)+8,y:Y(fmax.pre)-6},'Fmax'));
      var cur=pts.filter(function(p){return p.thr>=thr;}).pop();
      if(cur){svg.appendChild(el('circle',{cx:X(cur.rec),cy:Y(cur.pre),r:6,fill:'%(mark)s'}));}
      var kept=calls.filter(function(c){return c.s>=thr;}), tp=kept.filter(function(c){return c.tp;}).length, found={}; kept.forEach(function(c){if(c.tp)found[c.i]=1;});
      var pre=kept.length?tp/kept.length:0, rec=Object.keys(found).length/nInst, f1=pre+rec?2*pre*rec/(pre+rec):0;
      dl(root,[['calls kept at this cutoff',kept.length+' of '+n],['precision',kept.length?f3(pre):'no calls'],['recall',f3(rec)+' ('+Object.keys(found).length+' of '+nInst+' reachable instances)'],['F1',f3(f1)],['Fmax over every cutoff',f3(fmax.f1)+' at cutoff '+fmax.thr.toFixed(2)],['AUPRC, the shaded area',f3(au)],['recall at precision ≥ '+cfg.floor,f3(atFloor)]]);
    }
    sl.addEventListener('input',draw); draw();
  }

  // --- reachable: the query's answer key against what a target proteome has -------------
  function reachable(root,cfg){
    var svg=root.querySelector('svg'); svg.innerHTML='';
    svg.appendChild(el('text',{x:20,y:22},'query protein and its answer-key domains'));
    svg.appendChild(el('line',{x1:20,y1:44,x2:420,y2:44,stroke:'currentColor','stroke-width':2}));
    var tf={}; cfg.target.forEach(function(f){tf[f]=1;});
    cfg.domains.forEach(function(d){var ok=!!tf[d.fam]; svg.appendChild(el('rect',{x:d.x,y:34,width:d.w,height:20,fill:ok?'%(tp)s':'%(none)s',opacity:.8})); svg.appendChild(el('text',{x:d.x+d.w/2,y:48,'text-anchor':'middle',fill:'#fff','font-weight':'600'},d.fam)); svg.appendChild(el('text',{x:d.x+d.w/2,y:72,'text-anchor':'middle'},ok?'reachable':'not in the target'));});
    svg.appendChild(el('text',{x:20,y:106},'families with at least one instance in the target proteome: '+cfg.target.join(', ')));
    var reach=cfg.domains.filter(function(d){return tf[d.fam];}).length;
    svg.appendChild(el('text',{x:20,y:128},'recall is over the '+reach+' reachable instances, not all '+cfg.domains.length+': a family the target does not have cannot be found by transfer'));
  }

  // --- bh: ranked p-values against the Benjamini-Hochberg line --------------------------
  function bh(root,cfg){
    var svg=root.querySelector('svg'), ps=cfg.p.slice().sort(function(a,b){return a.p-b.p;}), m=ps.length, alpha=cfg.alpha, method='bh';
    var X=function(i){return 50+i*40;}, Y=function(p){return 170-p*120/(alpha*3)};
    function draw(){
      svg.innerHTML='';
      svg.appendChild(el('line',{x1:40,y1:170,x2:X(m)+10,y2:170,class:'axis'}));svg.appendChild(el('line',{x1:40,y1:40,x2:40,y2:170,class:'axis'}));
      svg.appendChild(el('text',{x:X(m/2),y:192,'text-anchor':'middle'},'hits, ranked by p-value (1 = smallest)'));
      svg.appendChild(el('text',{x:36,y:44,'text-anchor':'end'},(alpha*3).toFixed(2)));svg.appendChild(el('text',{x:36,y:174,'text-anchor':'end'},'0'));
      var rej=[], last=-1;
      ps.forEach(function(q,i){var lim=method==='bh'?alpha*(i+1)/m:alpha/m; if(q.p<=lim) last=i;});
      var d=''; for(var i=0;i<m;i++){var lim=method==='bh'?alpha*(i+1)/m:alpha/m; d+=(i?' L':'M')+X(i+1)+' '+Y(lim);}
      svg.appendChild(el('path',{d:d,fill:'none',stroke:'%(mark)s','stroke-width':2}));
      svg.appendChild(el('text',{x:X(m)+6,y:Y(method==='bh'?alpha:alpha/m)+4,fill:'%(mark)s'},method==='bh'?'BH line: α · rank / m':'Bonferroni: α / m'));
      ps.forEach(function(q,i){var r=i<=last; if(r) rej.push(q); svg.appendChild(el('circle',{cx:X(i+1),cy:Y(Math.min(q.p,alpha*3)),r:6,fill:q.o?'%(tp)s':'%(fp)s',opacity:r?1:.3}));});
      var tp=rej.filter(function(q){return q.o;}).length, orthAlpha=ps.filter(function(q){return q.o&&q.p<alpha;}).length;
      dl(root,[['hits below the line (rejected = called)',rej.length+' of '+m],['precision: orthologs among the called',rej.length?f3(tp/rej.length):'none called'],['recall: orthologs called / ortholog hits with p < α',orthAlpha?f3(tp/orthAlpha)+' ('+tp+' of '+orthAlpha+')':'n/a'],['α',String(alpha)]]);
    }
    root.querySelectorAll('.seg button').forEach(function(b){b.addEventListener('click',function(){root.querySelectorAll('.seg button').forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');method=b.getAttribute('data-m');draw();});});
    draw();
  }

  // --- fraction: a proteome bar split into placed and dark, and the mask pair -----------
  function fraction(root,cfg){
    var svg=root.querySelector('svg'); svg.innerHTML='';
    var W=600;
    svg.appendChild(el('text',{x:20,y:20},'every protein in the proteome, in one bar'));
    svg.appendChild(el('rect',{x:20,y:30,width:W*cfg.placed,height:26,fill:'%(tp)s'}));
    svg.appendChild(el('rect',{x:20+W*cfg.placed,y:30,width:W*(1-cfg.placed),height:26,fill:'%(none)s'}));
    svg.appendChild(el('text',{x:20+W*cfg.placed/2,y:48,'text-anchor':'middle',fill:'#fff','font-weight':'600'},'placed'));
    svg.appendChild(el('text',{x:20+W*cfg.placed+W*(1-cfg.placed)/2,y:48,'text-anchor':'middle',fill:'#fff','font-weight':'600'},'dark'));
    svg.appendChild(el('text',{x:20,y:78},'fraction dark = dark / proteins counted; the length cut changes only which proteins are counted'));
    svg.appendChild(el('text',{x:20,y:116},'inside the dark set, kmerseek reach with the mask on and off'));
    var D=W*(1-cfg.placed), x0=20+W*cfg.placed;
    svg.appendChild(el('rect',{x:x0,y:126,width:D,height:14,fill:'%(none)s',opacity:.35}));
    svg.appendChild(el('rect',{x:x0,y:126,width:D*cfg.on,height:14,fill:'%(tp)s'}));
    svg.appendChild(el('text',{x:x0+D+8,y:138},'mask on: reached'));
    svg.appendChild(el('rect',{x:x0,y:148,width:D,height:14,fill:'%(none)s',opacity:.35}));
    svg.appendChild(el('rect',{x:x0,y:148,width:D*cfg.off,height:14,fill:'%(tp)s'}));
    svg.appendChild(el('rect',{x:x0+D*cfg.on,y:148,width:D*(cfg.off-cfg.on),height:14,fill:'%(fp)s'}));
    svg.appendChild(el('text',{x:x0+D+8,y:160},'mask off: reached; the red part is lost to masking'));
    svg.appendChild(el('text',{x:20,y:190},'lost to masking = reached with the mask off minus reached with it on: found only through low-complexity sequence'));
  }

  var kinds={iou:iou,threshold:threshold,reachable:reachable,bh:bh,fraction:fraction};
  document.querySelectorAll('.mx[data-kind]').forEach(function(root){
    if(root.__done) return; root.__done=true;
    var cfg=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
    kinds[root.getAttribute('data-kind')](root,cfg);
  });
})();
</script>
"""

def _legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{c};border-color:{c}"></i>{t}</span>' if c else f'<span>{t}</span>'
        for c, t in items) + '</div>'


def _block(kind, title, definition, body, cfg, toy=None, uid=""):
    toy_p = f'<p class="toy">{toy}</p>' if toy else ""
    return (f'<div class="mx" data-kind="{kind}" id="{uid or "mx-" + kind}"><h5>{title}</h5>'
            f'<p class="def">{definition}</p>{toy_p}{body}'
            f'<script type="application/json">{json.dumps(cfg)}</script><dl class="out"></dl></div>')


def iou(min_overlap: float = 0.5, length: int = 400, dom=(120, 260)) -> str:
    """Overlap over union between a call and the true domain, with the call on sliders."""
    body = (_legend([("currentColor", "outline: the true domain on the query protein"), (C_TP, "the call, when it passes"),
                     (C_FP, "the call, when it fails: right family in the wrong place")])
            + f'<div class="ctrl"><label>call start <input type="range" data-k="start" min="0" max="{length}" value="150"></label>'
              f'<label>call end <input type="range" data-k="end" min="0" max="{length}" value="330"></label></div>'
              '<svg viewBox="0 0 720 120"></svg>')
    return _block("iou", "Overlap over union, and the 50% rule",
                  f"A call is a true positive only when the query has that family <i>and</i> the call's "
                  f"interval overlaps a real instance by at least {min_overlap:.0%}, measured as overlap "
                  f"divided by union. Move the call's ends and watch the number cross the line.",
                  body, {"length": length, "dom": list(dom), "min": min_overlap},
                  toy="Toy example: one protein, one true domain, one call.")


def threshold(floor: float = 0.95, title="Precision, recall, Fmax and AUPRC over a score threshold",
              mention_fdr: bool = True) -> str:
    """A toy set of calls and what one cutoff does to precision, recall, F1, Fmax, AUPRC."""
    calls = [{"s": 0.97, "tp": True, "i": 1}, {"s": 0.93, "tp": True, "i": 2}, {"s": 0.90, "tp": False, "i": 0},
             {"s": 0.86, "tp": True, "i": 3}, {"s": 0.80, "tp": True, "i": 4}, {"s": 0.74, "tp": False, "i": 0},
             {"s": 0.70, "tp": True, "i": 5}, {"s": 0.62, "tp": False, "i": 0}, {"s": 0.55, "tp": True, "i": 6},
             {"s": 0.47, "tp": False, "i": 0}, {"s": 0.40, "tp": False, "i": 0}, {"s": 0.33, "tp": True, "i": 7},
             {"s": 0.25, "tp": False, "i": 0}, {"s": 0.15, "tp": False, "i": 0}]
    body = (_legend([(C_TP, "a true call"), (C_FP, "a false call"), (C_MARK, "the cutoff, and where it sits on the curve"),
                     ("currentColor", "ring: Fmax, the best F1 over every cutoff"), (None, "shaded: AUPRC, the area under the curve")])
            + '<div class="ctrl"><label>score cutoff <input type="range" data-k="thr" min="0" max="100" value="60"></label></div>'
              '<svg viewBox="0 0 760 200"></svg>')
    return _block("threshold", title,
                  f"Every call carries a score. At one cutoff, precision is the share of kept calls that are "
                  f"true and recall is the share of reachable instances found. Sliding the cutoff traces the "
                  f"precision-recall curve; Fmax is the best F1 on it, AUPRC the area under it, and recall at "
                  f"a precision floor of {floor} is the recall reached while precision stays at or above "
                  f"{floor}" + (f" (the same thing as recall at {(1 - floor):.0%} FDR)." if mention_fdr else "."),
                  body, {"calls": calls, "instances": 8, "floor": floor},
                  toy="Toy example: 14 calls with made-up scores, 8 reachable instances.")


def reachable() -> str:
    body = (_legend([(C_TP, "an answer-key instance the target can reach: its family has an instance there"),
                     (C_NONE, "one it cannot: no transfer can name that family")])
            + '<svg viewBox="0 0 720 136"></svg>')
    cfg = {"domains": [{"fam": "PF00001", "x": 40, "w": 90}, {"fam": "PF00010", "x": 170, "w": 70},
                       {"fam": "PF00500", "x": 300, "w": 100}],
           "target": ["PF00001", "PF00010"]}
    return _block("reachable", "Reachable recall: the denominator",
                  "A call gets its family from the target protein it matched, so a family with no instance "
                  "anywhere in the target proteome can never be called there. Recall is over the instances "
                  "the target can reach; the report also prints how many that is per proteome.",
                  body, cfg, toy="Toy example: one query protein with three domains.")


def bh(alpha: float = 0.05) -> str:
    p = [{"p": 0.0004, "o": True}, {"p": 0.001, "o": True}, {"p": 0.003, "o": True}, {"p": 0.006, "o": False},
         {"p": 0.011, "o": True}, {"p": 0.014, "o": True}, {"p": 0.02, "o": False}, {"p": 0.031, "o": True},
         {"p": 0.04, "o": False}, {"p": 0.055, "o": True}, {"p": 0.08, "o": False}, {"p": 0.12, "o": True}]
    body = (_legend([(C_TP, "a hit whose gene pair is in the ortholog key"), (C_FP, "a hit whose pair is not"),
                     (C_MARK, "the correction line; hits at or below it are called"), (None, "faded: not called")])
            + '<div class="ctrl"><span>method</span><div class="seg"><button type="button" data-m="bh" aria-pressed="true">Benjamini-Hochberg</button>'
              '<button type="button" data-m="bonferroni" aria-pressed="false">Bonferroni</button></div></div>'
              '<svg viewBox="0 0 760 200"></svg>')
    return _block("bh", "Precision and recall after multiple-testing correction",
                  f"Every hit's Poisson p-value is corrected over all hits of its combo. Benjamini-Hochberg "
                  f"ranks the p-values and calls those under the line α · rank / m; Bonferroni uses "
                  f"the flat line α / m. Precision is the share of called hits that are orthologs; recall "
                  f"is the share of ortholog hits with p below α that survive the correction, so it is "
                  f"recall of the correction step, not of the search.",
                  body, {"p": p, "alpha": alpha}, toy="Toy example: 12 hits with made-up p-values, α = 0.05.")


def fraction(placed: float = 0.55, on: float = 0.3, off: float = 0.38) -> str:
    body = (_legend([(C_TP, "placed, or reached with the mask on"), (C_NONE, "dark"),
                     (C_FP, "reached only with the mask off: lost to masking")])
            + '<svg viewBox="0 0 720 200"></svg>')
    return _block("fraction", "Fraction dark, and what masking costs",
                  "The dark fraction is the dark proteins over the proteins counted. Inside the dark set, "
                  "kmerseek's reach is read as a pair: with the low-complexity mask on and off. A protein "
                  "reached only with the mask off was found through low-complexity sequence, which is "
                  "composition rather than homology, and is counted as lost to masking.",
                  body, {"placed": placed, "on": on, "off": off},
                  toy="Schematic: the bar widths are not this run's numbers; the panels below carry those.")


def bundle(*blocks: str) -> str:
    """The explainers as one HTML block with the style and script included once."""
    fill = {"%(tp)s": C_TP, "%(fp)s": C_FP, "%(none)s": C_NONE, "%(mark)s": C_MARK}
    css, js = CSS, JS
    for k, v in fill.items():
        css, js = css.replace(k, v), js.replace(k, v)
    return css + "".join(blocks) + js
