// ---- kmerseek pair view (shared by the 183 cases and the three reference pairs) ----
const fmt=n=>n==null?'—':Number(n).toLocaleString('en-US');
function pairLegend(o){
  let h='<div class="legend pl">';
  if(o.band)h+='<span class="lg"><span class="sw" style="background:var(--feat-t);border-color:var(--feat);border-width:2px"></span>'+esc(o.band.label)+' (shaded across the plot)</span>';
  if(o.calls&&o.calls.length)h+='<span class="lg"><span class="sw" style="height:8px;background:var(--km);opacity:.4;border:none;border-radius:4px"></span>'+esc(o.callLabel)+'</span>';
  h+='<span class="lg"><svg width="22" height="12" aria-hidden="true"><line x1="2" y1="10" x2="20" y2="2" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"/></svg>run of two or more shared k-mers consecutive in both proteins, numbered as in the residue blocks below</span>';
  h+='<span class="lg"><svg width="22" height="12" aria-hidden="true"><circle cx="11" cy="6" r="3" fill="var(--soft)"/></svg>a single shared k-mer with no neighbour on its diagonal</span>';
  return h+'</div>';
}
function pairSVG(p,o){
  const W=700,x0=78,x1=690,y0=64,Hp=Math.round(Math.max(220,Math.min(420,610*p.tl/p.ql))),y1=y0+Hp,H=y1+56;
  const X=q=>x0+q/p.ql*(x1-x0), Y=t=>y0+t/p.tl*(y1-y0);
  let s='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Dot plot of every k-mer the two proteins share: '+esc(o.qLabel)+' along the top, '+esc(o.tLabel)+' down the side">';
  // proteins: line + box, along the top (query) and the left (target)
  s+='<text x="'+x0+'" y="16" font-size="12" font-weight="600" fill="var(--ink)">'+esc(o.qLabel)+' ('+fmt(p.ql)+' aa) →</text>';
  s+='<line x1="'+x0+'" y1="40" x2="'+x1+'" y2="40" stroke="var(--ink)" stroke-width="2"/>';
  s+='<text transform="translate(12,'+y0+') rotate(90)" font-size="12" font-weight="600" fill="var(--ink)">'+esc(o.tLabel)+' ('+fmt(p.tl)+' aa) →</text>';
  s+='<line x1="30" y1="'+y0+'" x2="30" y2="'+y1+'" stroke="var(--ink)" stroke-width="2"/>';
  s+='<rect x="'+x0+'" y="'+y0+'" width="'+(x1-x0)+'" height="'+Hp+'" fill="none" stroke="var(--line)"/>';
  if(o.band){const b=o.band;
    if(b.q){s+='<rect x="'+X(b.q[0]-1).toFixed(1)+'" y="'+y0+'" width="'+Math.max(2,X(b.q[1])-X(b.q[0]-1)).toFixed(1)+'" height="'+Hp+'" fill="var(--feat-t)" opacity=".7"/>';
      s+='<rect x="'+X(b.q[0]-1).toFixed(1)+'" y="31" width="'+Math.max(3,X(b.q[1])-X(b.q[0]-1)).toFixed(1)+'" height="18" rx="2" fill="var(--feat-t)" stroke="var(--feat)" stroke-width="2"/>';}
    if(b.t){s+='<rect x="'+x0+'" y="'+Y(b.t[0]-1).toFixed(1)+'" width="'+(x1-x0)+'" height="'+Math.max(2,Y(b.t[1])-Y(b.t[0]-1)).toFixed(1)+'" fill="var(--feat-t)" opacity=".7"/>';
      s+='<rect x="21" y="'+Y(b.t[0]-1).toFixed(1)+'" width="18" height="'+Math.max(3,Y(b.t[1])-Y(b.t[0]-1)).toFixed(1)+'" rx="2" fill="var(--feat-t)" stroke="var(--feat)" stroke-width="2"/>';}
  }
  // search calls first, wide and pale, so runs stay visible on top
  (o.calls||[]).forEach(c=>{s+='<line x1="'+X(c[0]).toFixed(1)+'" y1="'+Y(c[2]).toFixed(1)+'" x2="'+X(c[1]).toFixed(1)+'" y2="'+Y(c[3]).toFixed(1)+'" stroke="var(--km)" stroke-opacity=".4" stroke-width="9" stroke-linecap="round"/>';});
  const num={}; p.bl.forEach((b,i)=>num[b.i]=i+1);
  p.rg.forEach((g,i)=>{const single=(g[1]-g[0])<=p.k;
    if(single)s+='<circle cx="'+X((g[0]+g[1])/2).toFixed(1)+'" cy="'+Y((g[2]+g[3])/2).toFixed(1)+'" r="2.4" fill="var(--soft)"/>';
    else s+='<line x1="'+X(g[0]).toFixed(1)+'" y1="'+Y(g[2]).toFixed(1)+'" x2="'+X(g[1]).toFixed(1)+'" y2="'+Y(g[3]).toFixed(1)+'" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"/>';});
  p.rg.forEach((g,i)=>{if(num[i]){const tx=Math.min(x1-10,X(g[1])+5),ty=Math.max(y0+11,Y(g[2])-3);
    s+='<text x="'+tx.toFixed(1)+'" y="'+ty.toFixed(1)+'" font-size="10.5" font-weight="600" fill="var(--ink)" paint-order="stroke" stroke="var(--panel)" stroke-width="3">'+num[i]+'</text>';}});
  // ticks
  const tk=L=>{const st=L<=600?100:L<=1500?200:500;const r=[1];for(let t=st;t<L;t+=st)if(L-t>L*.05)r.push(t);r.push(L);return r};
  tk(p.ql).forEach(t=>{const x=X(t-.5).toFixed(1);s+='<line x1="'+x+'" y1="'+y1+'" x2="'+x+'" y2="'+(y1+5)+'" stroke="var(--ink)"/><text x="'+x+'" y="'+(y1+17)+'" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+t+'</text>'});
  tk(p.tl).forEach(t=>{const y=Y(t-.5).toFixed(1);s+='<line x1="'+(x0-5)+'" y1="'+y+'" x2="'+x0+'" y2="'+y+'" stroke="var(--ink)"/><text x="'+(x0-8)+'" y="'+(+y+4)+'" text-anchor="end" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+t+'</text>'});
  s+='<text x="'+((x0+x1)/2)+'" y="'+(y1+36)+'" text-anchor="middle" font-size="11" fill="var(--soft)">residue position (aa, 1-based); each axis is scaled to its own protein\'s length</text>';
  return s+'</svg>';
}
function pairBlocks(p,o){
  const num={}; p.bl.forEach((b,i)=>num[b.i]=i+1);
  const w=Math.max(o.qShort.length,o.tShort.length)+1;
  let out='';
  p.bl.forEach((b,n)=>{const g=p.rg[b.i],L=g[1]-g[0];
    let id=0,cl=0,m='',mc='';for(let j=0;j<L;j++){const a=b.q[j]===b.t[j],c=b.qe[j]===b.te[j];id+=a;cl+=c;m+=a?'|':' ';mc+=c?'|':' '}
    const isCall=(o.calls||[]).some(c=>c[0]-c[2]===g[0]-g[2]&&c[0]<g[1]&&c[1]>g[0]);
    out+='<b>'+(n+1)+'.</b> '+L+' aa, '+id+' of '+L+' residues identical ('+Math.round(100*id/L)+'%), '+cl+' of '+L+' '+esc(p.a)+' classes identical'+(isCall?'  <span class="h">on the diagonal of '+esc(o.callShort)+'</span>':'')+'\n';
    const ps=String(g[0]+1).padStart(5),pt=String(g[2]+1).padStart(5);
    out+=o.qShort.padEnd(w)+ps+' '+esc(b.q)+' '+g[1]+'\n'+' '.repeat(w+6)+m+'\n'+o.tShort.padEnd(w)+pt+' '+esc(b.t)+' '+g[3]+'\n';
    out+=' '.repeat(w)+'classes'.padStart(5).slice(-5)+' '+esc(b.qe)+'\n'+' '.repeat(w+6)+mc+'\n'+' '.repeat(w+6)+esc(b.te)+'\n\n';
  });
  return out.replace(/\n+$/,'');
}
function pairClasses(p){return Object.entries(p.cls).map(([k,v])=>k+' = '+v).join(', ')}
function pairView(p,o){
  const nk=p.km.length/2, nr=p.rg.length, runs=p.rg.filter(g=>g[1]-g[0]>p.k).length;
  const longest=nr?Math.max(...p.rg.map(g=>g[1]-g[0])):0;
  let h='<div class="pv"><h3>'+(o.title||'Every k-mer the two proteins share')+'</h3>';
  h+='<p class="note"><code>kmerseek pair -a '+esc(p.a)+' -k '+p.k+'</code> on the whole of both proteins: '+fmt(nk)+' shared k-mers, '+fmt(runs)+' runs of two or more and '+fmt(nr-runs)+' single k-mers; longest run '+longest+' aa. pair keeps every shared k-mer: it has no low-complexity mask and no score cutoff, so it shows k-mers the search dropped. Classes: '+esc(pairClasses(p))+'.</p>';
  if(o.extra)h+='<p class="note">'+o.extra+'</p>';
  h+=pairLegend(o)+'<div class="fig">'+pairSVG(p,o)+'</div>';
  h+='<p class="note">Residue blocks: the '+Math.min(p.bl.length,8)+' longest runs'+(p.bl.length>8?' and the run on the call\'s diagonal':'')+', position against position, no gaps. Top pair of rows: residues, | = identical. Bottom: the same residues in the '+esc(p.a)+' alphabet, | = same class.</p>';
  h+='<div class="seq">'+pairBlocks(p,o)+'</div></div>';
  return h;
}
function casePair(r){
  const p=PAIRS[String(r.id)]; if(!p)return '';
  const c=p.call, how=p.callHow==='exact'?'kmerseek\'s call in the run is one of pair\'s runs, with the same ends on both proteins.':p.callHow==='contains'?'pair\'s run on the same diagonal contains kmerseek\'s call and is longer than it.':'No pair run lies on the diagonal of kmerseek\'s call.';
  return pairView(p,{qLabel:'human '+r.gene+' ('+r.acc+')',tLabel:r.species+' '+r.target,qShort:'human '+r.gene,tShort:r.species+' '+r.target,
    band:{q:[r.fs,r.fe],label:'the Swiss-Prot feature on the human protein, '+r.fs+'–'+r.fe},
    calls:[[c[0]-1,c[1],c[2]-1,c[3]]],callLabel:'kmerseek\'s call in the run (chosen arm), human '+c[0]+'–'+c[1]+', target '+c[2]+'–'+c[3],callShort:'kmerseek\'s call',extra:how});
}

const GHR='https://github.com/seanome/2024-kmerseek-analysis';
const LK={nb241:'<a href="'+GHR+'/blob/olgabot/alphabet-ranking-three-cases/notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb">notebook 241</a> (<a href="'+GHR+'/pull/46">PR #46</a>)',
  nb241s6:'<a href="'+GHR+'/blob/olgabot/alphabet-ranking-three-cases/notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb">notebook 241</a>, section 6',
  up:a=>'<a href="https://www.uniprot.org/uniprotkb/'+a+'/entry">'+a+'</a>'};
// ---- the three reference pairs: alphabet and k chosen in the side panel, hits in the main figure ----
const BH1={q:[160,179],t:[136,155]};
const MOTIFS={Ced9:[['BH4',80,99],['BH1',160,179],['BH2',213,229]],P66:[],BHF:[]};
const REF={
  Ced9:{title:'Ced9 → human',q:'Ced9',ql:280,partner:'BCL2',qLabel:'C. elegans Ced9',
    lede:'Ced9 (<i>C. elegans</i>, 280 aa) and human BCL2 (239 aa) are known homologues: the same SCOP superfamily, confirmed by structure, under 30% identity. The known shared region is the BH1 motif (UniProt '+LK.up('P41958')+' and '+LK.up('P10415')+'): Ced9 160–179 and BCL2 136–155. It is shaded gold everywhere on this page.',
    band:{q:BH1.q,t:BH1.t,label:'the BH1 motif, the known shared region (Ced9 160–179, BCL2 136–155; UniProt)'},
    found:m=>m.sr.some(g=>g[0]<BH1.q[1]&&g[1]>BH1.q[0]-1&&g[2]<BH1.t[1]&&g[3]>BH1.t[0]-1),
    foundText:'kmerseek\'s search reported a region on BCL2 that overlaps BH1 on both proteins',
    best:'Over all 102 alphabet, k and ranking-statistic combinations in '+LK.nb241+', BCL2\'s best rank is 213 of 18,064 human proteins hit (polarity4, mean IDF, k=9). A human protein drawn at random from the same hit lists ranks as well or better in 97% of 20,000 draws ('+LK.nb241s6+').'},
  P66:{title:'P66 → human',q:'P66',ql:597,partner:'CD47',qLabel:'B. burgdorferi P66',
    lede:'P66 (<i>Borreliella burgdorferi</i>, 597 aa) and human CD47 (323 aa). CD47 is the proposed partner of P66. '+LK.nb241+' lists this as a claim that has not been shown and asks whether any alphabet supports it. There is no known shared region to shade.',
    band:null, found:m=>m.sr.length>0, foundText:'kmerseek\'s search reported a region on CD47',
    best:'Over all 121 alphabet, k and ranking-statistic combinations in '+LK.nb241+', CD47\'s best rank is 166 of 18,775 human proteins hit (hp_lehninger_hpc3, E-value, k=14). A human protein drawn at random from the same hit lists ranks as well or better in 90% of draws ('+LK.nb241s6+').'},
  BHF:{title:'BHF → human',q:'BHF',ql:252,partner:null,qLabel:'Botryllus BHF',
    lede:'BHF (<i>Botryllus schlosseri</i> histocompatibility factor, 252 aa) has no known human homologue and no annotated domains. 71 alphabet and k values have a Karlin-Altschul fit; 61 of them give some region a finite E-value, and 3 put a human protein just under E = 1 (SCN10A 0.61 at polarity4 k=13, SFI1 0.84 at hp_lehninger_c_nonpolar2 k=28, FXYD5 0.99 at gbmr7 k=8). '+LK.nb241+' reads this as no better than chance.',
    band:null, found:null, foundText:'some human protein has a region with E-value under 1'},
};
const LETTERS={}; const ALPHAS=[]; const PLAN={};
Object.keys(META241).forEach(key=>{const [q,a,k]=key.split('|');if(q!=='Ced9')return;if(!PLAN[a]){PLAN[a]=[];ALPHAS.push(a);LETTERS[a]=Object.keys(PAIRS[key].cls).length}PLAN[a].push({k:+k,bits:META241[key].bits})});
ALPHAS.sort((x,y)=>LETTERS[x]-LETTERS[y]||ALPHAS.indexOf(x)-ALPHAS.indexOf(y));
let marginW=330; try{const w=+localStorage.getItem('kmgMarginW');if(w>=200&&w<=700)marginW=w}catch(e){}
let refSel='Ced9', refArm={Ced9:['hp_lehninger2',17],P66:['hp_lehninger2',17],BHF:['polarity4',13]}, refRow={};
function armFound(q,a,k){if(q==='BHF'){const r=BHFARMS[a+'|'+k];return r.bestE!=null&&r.bestE<1}return REF[q].found(META241[q+'|'+a+'|'+k])}
const eFmt=e=>e==null?'no E':e<0.01?e.toExponential(1):e<10?e.toFixed(2):fmt(Math.round(e));
function classTip(a){const c=PAIRS['Ced9|'+a+'|'+PLAN[a][0].k].cls;return a+', '+Object.keys(c).length+' letters: '+Object.entries(c).map(([k,v])=>k+' = '+v).join(', ')}
function margin(q){
  const [sa,sk]=refArm[q]; let n=0,tot=0; ALPHAS.forEach(a=>PLAN[a].forEach(({k})=>{tot++;if(armFound(q,a,k))n++}));
  let h='<div class="mhead">Choose an alphabet and k</div><p class="mnote"><span class="chip on demo">k</span> filled: '+REF[q].foundText+' ('+n+' of '+tot+'). <span class="chip demo">k</span> open: it did not. Alphabets run from fewest to most letters; hover a name for its classes. k runs small to large.</p>';
  ALPHAS.forEach(a=>{
    h+='<div class="malpha'+(a===sa?' cur':'')+'"><span class="mname" tabindex="0" data-tip="'+esc(classTip(a))+'">'+a+'</span><span class="chips">';
    PLAN[a].forEach(({k,bits})=>{h+='<button class="chip'+(armFound(q,a,k)?' on':'')+(a===sa&&k===sk?' sel':'')+'" data-a="'+a+'" data-k="'+k+'" title="'+a+', k = '+k+', '+bits+' bits per seed" aria-pressed="'+(a===sa&&k===sk)+'">'+k+'</button>'});
    h+='</span></div>'});
  return h;
}
function hitsSVG(q,H,cls){
  const R=REF[q],L=R.ql,W=700,x0=200,x1=690,X=p=>x0+p/L*(x1-x0),rh=22,top=52,Hh=top+H.rows.length*rh+44;
  let s='<svg viewBox="0 0 '+W+' '+Hh+'" role="img" aria-label="'+esc(R.qLabel)+' along the top; one row per human protein hit, with its matched regions drawn at '+esc(q)+' positions">';
  s+='<text x="'+x0+'" y="14" font-size="12" font-weight="600" fill="var(--ink)">'+esc(R.qLabel)+' ('+L+' aa)</text>';
  s+='<line x1="'+x0+'" y1="34" x2="'+x1+'" y2="34" stroke="var(--ink)" stroke-width="2"/>';
  MOTIFS[q].forEach(([nm,a,b])=>{const bh=nm==='BH1';s+='<rect x="'+X(a-1).toFixed(1)+'" y="26" width="'+(X(b)-X(a-1)).toFixed(1)+'" height="16" rx="2" fill="'+(bh?'var(--feat-t)':'var(--panel)')+'" stroke="'+(bh?'var(--feat)':'var(--soft)')+'" stroke-width="'+(bh?2:1.2)+'"/><text x="'+((X(a-1)+X(b))/2).toFixed(1)+'" y="21" text-anchor="middle" font-size="10.5" fill="var(--ink)">'+nm+'</text>'});
  if(q==='Ced9')s+='<rect x="'+X(BH1.q[0]-1).toFixed(1)+'" y="'+(top-4)+'" width="'+(X(BH1.q[1])-X(BH1.q[0]-1)).toFixed(1)+'" height="'+(H.rows.length*rh+4)+'" fill="var(--feat-t)" opacity=".55"/>';
  H.rows.forEach((r,i)=>{const y=top+i*rh,isP=R.partner&&r.gene===R.partner,sel=refRow[q]===r.gene;
    s+='<g class="hr" tabindex="0" role="button" data-g="'+esc(r.gene)+'" aria-label="'+esc(r.gene)+'">';
    s+='<rect x="2" y="'+y+'" width="'+(W-4)+'" height="'+(rh-2)+'" rx="3" fill="'+(isP?'var(--accent-t)':'transparent')+'" stroke="'+(sel?'var(--ink)':'none')+'" stroke-width="2"/>';
    s+='<text x="8" y="'+(y+14)+'" font-size="11" font-family="var(--mono)" fill="var(--soft)">'+r.pos+'.</text><text x="50" y="'+(y+14)+'" font-size="11.5" font-family="var(--mono)" fill="var(--ink)" font-weight="'+(isP?600:400)+'">'+esc(r.gene).slice(0,11)+'</text>';
    s+='<text x="'+(x0-8)+'" y="'+(y+14)+'" text-anchor="end" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+(H.by==='E-value'?'E '+eFmt(r.E):'IDF '+r.idf)+'</text>';
    s+='<line x1="'+x0+'" y1="'+(y+10)+'" x2="'+x1+'" y2="'+(y+10)+'" stroke="var(--line)"/>';
    r.rg.forEach(g=>{s+='<rect x="'+X(g[0]).toFixed(1)+'" y="'+(y+4)+'" width="'+Math.max(2,X(g[1])-X(g[0])).toFixed(1)+'" height="12" rx="1.5" fill="var(--ink)"/>'});
    s+='</g>'});
  const yA=top+H.rows.length*rh+4; const st=L<=300?50:100;
  for(let t=0;t<=L;t+=st){const p=t||1,x=X(p-.5).toFixed(1);s+='<line x1="'+x+'" y1="'+yA+'" x2="'+x+'" y2="'+(yA+5)+'" stroke="var(--ink)"/><text x="'+x+'" y="'+(yA+17)+'" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+p+'</text>'}
  s+='<text x="'+((x0+x1)/2)+'" y="'+(yA+34)+'" text-anchor="middle" font-size="11" fill="var(--soft)">position on '+esc(R.qLabel)+' (aa); every hit\'s regions are drawn at the '+esc(q)+' positions they match</text>';
  return s+'</svg>';
}
function rowResidues(q,r,cls){
  const enc=s=>[...s].map(c=>cls[c]||'?').join(''); let out='';
  r.rg.forEach((g,n)=>{const L=g[1]-g[0],qe=enc(g[6]),te=enc(g[7]);let id=0,cl=0,m='',mc='';
    for(let j=0;j<L;j++){const a=g[6][j]===g[7][j],c=qe[j]===te[j];id+=a;cl+=c;m+=a?'|':' ';mc+=c?'|':' '}
    out+='<b>'+(n+1)+'.</b> '+q+' '+(g[0]+1)+'–'+g[1]+', '+r.gene+' '+(g[2]+1)+'–'+g[3]+': '+L+' aa, E '+eFmt(g[4])+', mean IDF '+g[5]+', '+id+' of '+L+' residues identical, '+cl+' of '+L+' classes identical\n';
    const w=Math.max(q.length,r.gene.length)+1;
    out+=q.padEnd(w)+String(g[0]+1).padStart(5)+' '+esc(g[6])+' '+g[1]+'\n'+' '.repeat(w+6)+m+'\n'+r.gene.padEnd(w)+String(g[2]+1).padStart(5)+' '+esc(g[7])+' '+g[3]+'\n';
    out+='classes'.padEnd(w+6)+qe+'\n'+' '.repeat(w+6)+mc+'\n'+' '.repeat(w+6)+te+'\n\n'});
  return out.replace(/\n+$/,'');
}
function renderRef(){
  const el=$('refcard'); document.querySelectorAll('#refpick button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===refSel));
  const q=refSel,R=REF[q],[a,k]=refArm[q],H=HITS241[q+'|'+a+'|'+k],pc=PAIRS['Ced9|'+a+'|'+k],cls={};
  Object.entries(pc.cls).forEach(([sym,res])=>[...res].forEach(c=>cls[c]=sym));
  let h='<div class="head"><h2>'+R.title+'</h2><span class="tag">'+LK.nb241+'</span></div><p class="lede2">'+R.lede+'</p>';
  h+='<div class="refgrid" id="refgrid" style="--mw:'+marginW+'px"><aside class="margin" id="margin">'+margin(q)+'</aside><div class="resizer" id="resizer" role="separator" aria-orientation="vertical" aria-label="Drag to resize the alphabet panel" tabindex="0"></div><div class="main">';
  const bits=PLAN[a].find(x=>x.k===k).bits;
  h+='<h3 class="hh">'+a+' ('+LETTERS[a]+' letters), k = '+k+', '+bits+' bits per seed</h3>';
  if(R.partner){const m=META241[q+'|'+a+'|'+k];h+='<div class="nums">';
    ['E-value','mean IDF','bit score','shared k-mers'].forEach(mt=>{const r=m.ranks[mt];
      h+='<div class="n">'+R.partner+' rank by '+mt+'<b>'+(!r?'not computed':r[0]==null?'not hit':fmt(r[0])+' of '+fmt(r[1])+(r[2]?' ('+fmt(r[2])+' tied)':''))+'</b></div>'});
    h+='</div><p class="note">'+R.best+'</p>'}
  h+=toolsPanel(q,a,k,H);
  h+='<h3 class="hh">kmerseek\'s top hits at '+a+', k = '+k+'</h3>';
  if(!H){h+='<p class="note">The search at this alphabet and k hit no human protein.</p>'}else{
    if(!refRow[q]||!H.rows.some(r=>r.gene===refRow[q]))refRow[q]=(R.partner&&H.rows.some(r=>r.gene===R.partner))?R.partner:H.rows[0].gene;
    h+='<p class="note">The top '+Math.min(20,H.n)+' of '+fmt(H.n)+' human proteins hit, ordered by each protein\'s best region '+H.by+(H.by==='mean IDF'?' (no region has a finite E-value at this alphabet and k)':'')+(R.partner&&!H.rows.some(r=>r.gene===R.partner)?'. '+R.partner+' is not among the proteins hit.':R.partner&&H.rows[H.rows.length-1].gene===R.partner&&H.rows.length>20?'. '+R.partner+' is added as the last row with its own position.':'.')+' Click a protein to see its matched residues.</p>';
    h+='<div class="legend pl">'+(q==='Ced9'?'<span class="lg"><span class="sw" style="background:var(--feat-t);border-color:var(--feat);border-width:2px"></span>BH1 motif, the known shared region (shaded down every row)</span><span class="lg"><span class="sw" style="background:var(--panel);border-color:var(--soft)"></span>another UniProt motif on Ced9 (BH4, BH2)</span>':'')
      +'<span class="lg"><span class="sw" style="background:var(--ink);border-color:var(--ink);height:8px"></span>a region kmerseek\'s search reported, drawn at the '+q+' positions it matches</span>'
      +(R.partner?'<span class="lg"><span class="sw" style="background:var(--accent-t);border-color:var(--accent-t)"></span>the '+(q==='Ced9'?'known homologue':'proposed partner')+', '+R.partner+'</span>':'')
      +'<span class="lg"><span class="sw" style="background:transparent;border-color:var(--ink);border-width:2px"></span>the protein whose residues are shown below</span></div>';
    h+='<div class="fig">'+hitsSVG(q,H,cls)+'</div>';
    const r=H.rows.find(x=>x.gene===refRow[q]);
    h+='<h3 class="hh">'+q+' and '+esc(r.gene)+': the matched residues</h3><p class="note">Each region, position against position, no gaps. Top rows: residues, | = identical. Bottom rows: '+a+' classes, | = same class. Classes: '+esc(pairClasses(pc))+'.</p><div class="seq">'+rowResidues(q,r,cls)+'</div>';
  }
  const pk=R.partner?q+'|'+a+'|'+k:'BHFarm|'+a+'|'+k, p=PAIRS[pk];
  if(R.partner){const m=META241[pk];h+=pairView(p,{title:'Every k-mer '+q+' and '+R.partner+' share',qLabel:R.qLabel,tLabel:'human '+R.partner,qShort:q,tShort:R.partner,band:R.band,calls:m.sr.map(g=>g.slice(0,4)),callShort:'a search region',callLabel:'a region kmerseek\'s search reported on '+R.partner})}
  else if(p){const b=BHFARMS[a+'|'+k];h+=pairView(p,{title:'Every k-mer BHF and '+b.shown+' share',qLabel:R.qLabel,tLabel:'human '+b.shown,qShort:'BHF',tShort:b.shown,band:null,calls:[],callShort:'',extra:b.shown+' has the best '+b.shownBy+' at this alphabet and k. It is the best of the hits, not a known partner.'})}
  h+='</div></div>';
  const sc=$('margin')?$('margin').scrollTop:0;
  el.innerHTML=h; $('margin').scrollTop=sc;
  wireResizer(); wireTips(el);
  el.querySelectorAll('.chip[data-a]').forEach(b=>b.onclick=()=>{refArm[q]=[b.dataset.a,+b.dataset.k];renderRef()});
  el.querySelectorAll('.hr').forEach(g=>{const go=()=>{refRow[q]=g.dataset.g;renderRef()};g.onclick=go;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}});
}
document.querySelectorAll('#refpick button').forEach(b=>b.onclick=()=>{refSel=b.dataset.k;renderRef()});
function showTab(t){const cases=t!=='pairs';document.querySelector('.wrap').classList.toggle('wide',!cases);$('tabcases').hidden=!cases;$('tabpairs').hidden=cases;
  $('t-cases').setAttribute('aria-selected',cases);$('t-pairs').setAttribute('aria-selected',!cases);if(!cases)renderRef();}
$('t-cases').onclick=()=>{history.replaceState(null,'','#cases');showTab('cases')};
$('t-pairs').onclick=()=>{history.replaceState(null,'','#pairs');showTab('pairs')};

// ---- other tools: each tool's calls drawn on the query, like the short-feature cases ----
const TOOLNAMES=['phmmer','jackhmmer (3 rounds)','MMseqs2','MMseqs2 (3 rounds)','blastp'];
const iouBH1=(a,b)=>{const i=Math.max(0,Math.min(b,BH1.q[1])-Math.max(a,BH1.q[0])+1);return i/((b-a+1)+(BH1.q[1]-BH1.q[0]+1)-i)};
function toolsPanel(q,a,k,H){
  const R=REF[q],L=R.ql,W=700,x0=190,x1=690,X=p=>x0+p/L*(x1-x0),rh=36,top=52,T=OTHER[q];
  const rows=[];
  // kmerseek row: 0-based half-open search regions -> 1-based inclusive
  if(R.partner){rows.push({name:'kmerseek '+a+' k'+k,km:true,calls:META241[q+'|'+a+'|'+k].sr.map(g=>({qs:g[0]+1,qe:g[1],ts:g[2]+1,te:g[3],E:g[4],gene:R.partner})),n:H?H.n:0,best:H&&H.rows[0]})}
  else{const r=H&&H.rows[0];rows.push({name:'kmerseek '+a+' k'+k,km:true,calls:r?r.rg.map(g=>({qs:g[0]+1,qe:g[1],ts:g[2]+1,te:g[3],E:g[4],gene:r.gene})):[],n:H?H.n:0,best:r})}
  TOOLNAMES.forEach(t=>{const hs=T[t];rows.push({name:t,calls:R.partner?hs.filter(x=>x.gene===R.partner):hs.filter(x=>hs.length&&x.gene===hs[0].gene),n:new Set(hs.map(x=>x.gene)).size,best:hs[0]})});
  const Hh=top+rows.length*rh+40;
  let s='<svg viewBox="0 0 '+W+' '+Hh+'" role="img" aria-label="'+esc(R.qLabel)+' as a line, one row per tool with its calls drawn at '+q+' positions">';
  s+='<text x="'+x0+'" y="14" font-size="12" font-weight="600" fill="var(--ink)">'+esc(R.qLabel)+' ('+L+' aa)</text><line x1="'+x0+'" y1="34" x2="'+x1+'" y2="34" stroke="var(--ink)" stroke-width="2"/>';
  MOTIFS[q].forEach(([nm,p0,p1])=>{const bh=nm==='BH1';s+='<rect x="'+X(p0-1).toFixed(1)+'" y="26" width="'+(X(p1)-X(p0-1)).toFixed(1)+'" height="16" rx="2" fill="'+(bh?'var(--feat-t)':'var(--panel)')+'" stroke="'+(bh?'var(--feat)':'var(--soft)')+'" stroke-width="'+(bh?2:1.2)+'"/><text x="'+((X(p0-1)+X(p1))/2).toFixed(1)+'" y="21" text-anchor="middle" font-size="10.5" fill="var(--ink)">'+nm+'</text>'});
  if(q==='Ced9')s+='<rect x="'+X(BH1.q[0]-1).toFixed(1)+'" y="'+(top-6)+'" width="'+(X(BH1.q[1])-X(BH1.q[0]-1)).toFixed(1)+'" height="'+(rows.length*rh)+'" fill="var(--feat-t)" opacity=".55"/>';
  rows.forEach((r,i)=>{const y=top+i*rh;
    s+='<text x="'+(x0-10)+'" y="'+(y+10)+'" text-anchor="end" font-size="11.5" fill="var(--ink)" font-weight="'+(r.km?600:400)+'">'+esc(r.name)+'</text>';
    s+='<line x1="'+x0+'" y1="'+(y+6)+'" x2="'+x1+'" y2="'+(y+6)+'" stroke="var(--line)"/>';
    let note;
    if(r.calls.length){r.calls.forEach(c=>{s+='<rect x="'+X(c.qs-1).toFixed(1)+'" y="'+y+'" width="'+Math.max(2,X(c.qe)-X(c.qs-1)).toFixed(1)+'" height="12" rx="1.5" fill="'+(r.km?'var(--km)':'var(--ink)')+'"/>'});
      const c=r.calls.reduce((m,x)=>(x.E!=null&&(m.E==null||x.E<m.E))?x:m,r.calls[0]);
      note=(R.partner?R.partner:c.gene)+' '+q+' '+c.qs+'–'+c.qe+' / '+(R.partner||c.gene)+' '+c.ts+'–'+c.te+', E '+eFmt(c.E)+(q==='Ced9'?', IoU with BH1 '+iouBH1(c.qs,c.qe).toFixed(2):'')+(r.calls.length>1?' ('+r.calls.length+' calls)':'')+' · '+fmt(r.n)+' proteins hit';}
    else{note='× '+(R.partner?R.partner+' not hit':'no hit at E ≤ 10')+(r.n?' · '+fmt(r.n)+' proteins hit, best '+esc(r.best.gene)+' E '+eFmt(r.best.E):'')}
    s+='<text x="'+x0+'" y="'+(y+26)+'" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+note+'</text>'});
  const yA=top+rows.length*rh-4,st=L<=300?50:100;
  for(let t=0;t<=L;t+=st){const p=t||1,x=X(p-.5).toFixed(1);s+='<line x1="'+x+'" y1="'+yA+'" x2="'+x+'" y2="'+(yA+5)+'" stroke="var(--ink)"/><text x="'+x+'" y="'+(yA+17)+'" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--soft)">'+p+'</text>'}
  s+='<text x="'+((x0+x1)/2)+'" y="'+(yA+33)+'" text-anchor="middle" font-size="11" fill="var(--soft)">position on '+esc(R.qLabel)+' (aa)</text></svg>';
  let h='<h3 class="hh">'+(R.partner?'Which tools find '+R.partner+', and where on '+q:'What each tool finds for BHF')+'</h3>';
  h+='<p class="note">Every tool searched the same 19,732 GENCODE v49 canonical human proteins, reporting hits at E ≤ 10 with the settings of the region benchmark pipeline: phmmer and jackhmmer (HMMER 3.4; jackhmmer 3 rounds), MMseqs2 18.8cc5c at sensitivity 7 (plain, and 3 rounds), blastp 2.16.0. '+(R.partner?'Each bar is a call on '+R.partner+'.':'Each bar is a call on the tool\'s best-scoring human protein.')+' The kmerseek row follows the alphabet and k chosen on the left. Foldseek, ProstT5 and Reseek have not been run on these queries.</p>';
  h+='<div class="legend pl">'+(q==='Ced9'?'<span class="lg"><span class="sw" style="background:var(--feat-t);border-color:var(--feat);border-width:2px"></span>BH1 motif, the known shared region</span>':'')+'<span class="lg"><span class="sw" style="background:var(--km);border-color:var(--km);height:8px"></span>kmerseek\'s search region</span><span class="lg"><span class="sw" style="background:var(--ink);border-color:var(--ink);height:8px"></span>another tool\'s call</span><span class="lg"><span class="sw" style="background:transparent;border:none;text-align:center;font-family:var(--mono)">×</span>'+(R.partner?'the tool did not report '+R.partner:'the tool reported no human protein')+'</span></div>';
  h+='<div class="fig">'+s+'</div>';
  h+='<details class="alltools"><summary>Every hit each tool reported for '+q+' (E ≤ 10)</summary><div class="seq">';
  TOOLNAMES.forEach(t=>{const hs=T[t];h+='<b>'+t+'</b>: '+(hs.length?'':'no hits')+'\n';hs.forEach(x=>{h+='  '+x.gene.padEnd(12)+' E '+eFmt(x.E).padStart(8)+'   '+q+' '+String(x.qs).padStart(3)+'–'+x.qe+'   '+x.gene+' '+x.ts+'–'+x.te+' ('+x.tl+' aa)\n'});h+='\n'});
  return h+'</div></details>';
}
function wireResizer(){
  const g=$('refgrid'),r=$('resizer');if(!g||!r)return;
  const set=w=>{marginW=Math.max(200,Math.min(700,Math.round(w)));g.style.setProperty('--mw',marginW+'px');try{localStorage.setItem('kmgMarginW',marginW)}catch(e){}};
  r.onpointerdown=e=>{e.preventDefault();r.setPointerCapture(e.pointerId);const x0=e.clientX,w0=marginW;
    r.onpointermove=ev=>set(w0+ev.clientX-x0);r.onpointerup=()=>{r.onpointermove=null}};
  r.onkeydown=e=>{if(e.key==='ArrowLeft')set(marginW-20);if(e.key==='ArrowRight')set(marginW+20)};
}
function wireTips(el){
  const tip=$('tip');
  el.querySelectorAll('[data-tip]').forEach(n=>{
    const show=()=>{tip.textContent=n.dataset.tip;tip.hidden=false;const b=n.getBoundingClientRect();tip.style.left=Math.min(window.innerWidth-tip.offsetWidth-8,b.left)+'px';tip.style.top=(b.bottom+6)+'px'};
    n.onmouseenter=show;n.onfocus=show;n.onmouseleave=()=>tip.hidden=true;n.onblur=()=>tip.hidden=true;});
}
showTab(location.hash==='#pairs'?'pairs':'cases');
