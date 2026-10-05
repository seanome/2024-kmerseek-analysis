"""Write notebooks/experiments.html from data/experiments.json, data/claims.json and data/open_prs.json.

The page is one file with the data embedded at the bottom, so it opens offline.
Every number a claim shows is looked up from experiments.json by row id and
number name; the build stops if a claim names a number that is not there.
Run from the repo root:

    python scripts/build_experiments_page.py [--site DIR] [--artifact FILE]

--site DIR writes the GitHub Pages copy (DIR/index.html and DIR/experiments.html).
--artifact FILE writes a body-only copy for the claude.ai artifact shelf, which
supplies its own html/head/body.

The dates on the page come from git: when the page was first committed, when
data/experiments.json last changed, and when this build ran. With git and the
branches available, each experiment is also checked against its notebook's
branch today; a notebook with commits since its numbers were read is marked
"changed since read".
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "experiments.json"
CLAIMS = REPO / "data" / "claims.json"
PRS = REPO / "data" / "open_prs.json"
PAGE = REPO / "notebooks" / "experiments.html"

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kmerseek experiments</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..700&family=Source+Sans+3:ital,wght@0,300..800;1,300..800&display=swap">
<style id="seanome-fonts">__FONTS__</style>
<style>
/* Theme: the kmerseek docs landing page (seanome/kmerseek docs/assets/site.css): its palette,
   glass panels over the moving water, light and dark. Fonts are Olga's artifact choices:
   Fraunces for the page title, Source Sans 3 for text, Fantasque Sans Mono for sequences and numbers. Colour carries one meaning
   only, the verdict badges; everything else is ink. */
:root {
  --bg: #F6F7F5; --surface: #FFFFFF; --ink: #1C211F; --hair: #D9DDDA;
  --glass: rgba(255,255,255,.86); --glass-edge: rgba(20,60,70,.20); --shadow: 0 14px 34px -26px rgba(0,40,50,.75);
  --row-hover: #EEF1EF;
  --alive: #1e7b34; --dead: #7a7f86; --bug: #c85a00; --running: #1f5fbf; --notrun: #1C211F;
  --chip-on-bg: #1C211F; --chip-on-ink: #FFFFFF;
  --sans: "Source Sans 3", -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: "Fantasque Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141716; --surface: #1B1F1E; --ink: #E6EAE8; --hair: #2E3432;
    --glass: rgba(5,22,30,.78); --glass-edge: rgba(170,230,225,.16); --shadow: 0 18px 44px -28px rgba(0,0,0,.85);
    --row-hover: #232927;
    --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #E6EAE8;
    --chip-on-bg: #E6EAE8; --chip-on-ink: #141716; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #141716; --surface: #1B1F1E; --ink: #E6EAE8; --hair: #2E3432;
  --glass: rgba(5,22,30,.78); --glass-edge: rgba(170,230,225,.16); --shadow: 0 18px 44px -28px rgba(0,0,0,.85);
  --row-hover: #232927;
  --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #E6EAE8;
  --chip-on-bg: #E6EAE8; --chip-on-ink: #141716; color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { margin: 0; overflow-x: hidden; }
body { background: var(--bg); color: var(--ink); font: 16px/1.6 var(--sans); }
canvas#water { position: fixed; inset: 0; width: 100%; height: 100%; display: block; z-index: 0; }
main, footer { position: relative; z-index: 1; }
main { max-width: 1180px; margin: 0 auto; padding: 48px 16px 24px; display: grid; gap: 18px; }
.panel { background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 10px; padding: 18px 22px; box-shadow: var(--shadow);
  backdrop-filter: blur(12px) saturate(1.15); -webkit-backdrop-filter: blur(12px) saturate(1.15); min-width: 0; }
h1 { font-family: "Fraunces", Georgia, serif; font-size: 1.9rem; line-height: 1.2; margin: 0 0 8px; text-wrap: balance; }
h2 { font-size: 1.3rem; margin: 0 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--hair); text-wrap: balance; }
h3 { font-size: 1.05rem; margin: 18px 0 6px; }
p { max-width: 72ch; }
a { color: inherit; }
.lede { margin: 0 0 8px; font-size: 1.05rem; }
.meta { margin: 0 0 10px; font-size: .88rem; max-width: 95ch; }
.tag { display: inline-block; font-size: .72rem; font-style: italic; padding: 0 4px; border: 1px dotted var(--ink); border-radius: 0; white-space: nowrap; margin-left: 4px; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .95rem; margin: 6px 0 0; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 0 0 8px; }
.legend span.what { font-style: italic; font-size: .85rem; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px; border-radius: 4px;
  font-size: .8rem; font-weight: 600; white-space: nowrap; border: 1.5px solid transparent; color: #fff; }
.badge.alive { background: var(--alive); }
.badge.dead { background: var(--dead); }
.badge.bug { background: var(--bug); }
.badge.running { background: var(--running); }
.badge.not-run { background: transparent; color: var(--notrun); border-color: var(--notrun); }
code, .val { font-family: var(--mono); font-size: .95em; }
pre { font-family: var(--mono); font-size: .9rem; line-height: 1.35; margin: 4px 0 8px; padding: 10px 12px; overflow-x: auto;
  border: 1px solid var(--hair); border-radius: 8px; background: var(--surface); white-space: pre; }
figure.mech { margin: 12px 0 0; }
figure.mech .legend span { display: inline-flex; align-items: center; gap: 6px; font-size: .85rem; }
figure.mech svg.mech { display: block; width: 100%; min-width: 760px; height: auto; color: var(--ink); }
figure.mech .legend svg { color: var(--ink); flex: none; }
figure.mech figcaption { font-size: .9rem; max-width: 95ch; margin-top: 6px; }
.steps { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 10px 0; padding: 0; list-style: none; counter-reset: s; }
.steps li { border: 1px solid var(--hair); border-radius: 8px; padding: 10px 12px; background: var(--surface); counter-increment: s; }
.steps li::before { content: counter(s) ". "; font-weight: 700; }
.claimgroup { margin: 14px 0 0; }
.claimgroup h3 { margin-top: 20px; }
.claim { border: 1px solid var(--hair); border-radius: 8px; padding: 12px 14px; margin: 8px 0; background: var(--surface); }
.claim .cname { font-weight: 700; margin: 0 0 2px; }
.claim .stmt { margin: 0 0 6px; max-width: 90ch; }
.claim .cav { margin: 6px 0 0; font-style: italic; max-width: 90ch; font-size: .92rem; }
.claim .prs { margin: 6px 0 0; font-size: .9rem; }
.claim details > summary { cursor: pointer; font-size: .92rem; }
.claim details[open] > summary { margin-bottom: 6px; }
.ev { width: 100%; border-collapse: collapse; font-size: .88rem; }
.ev td, .ev th { text-align: left; vertical-align: top; padding: 4px 6px; border-top: 1px solid var(--hair); }
.ev th { font-weight: 600; }
.ev td.v { font-family: var(--mono); white-space: nowrap; }
.ev td.b { white-space: nowrap; }
.rowlink { font: inherit; font-family: var(--mono); background: none; border: 1px solid var(--hair); border-radius: 4px; padding: 0 5px; color: var(--ink); cursor: pointer; white-space: nowrap; }
.rowlink:hover, .rowlink:focus-visible { border-color: var(--ink); }
details.gloss { border: 1px solid var(--hair); border-radius: 8px; padding: 8px 12px; margin: 12px 0; background: var(--surface); }
details.gloss summary { cursor: pointer; font-weight: 600; }
dl.terms { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 10px 0 4px; }
dl.terms dt { font-weight: 600; }
dl.terms dd { margin: 0; max-width: 80ch; }
@media (max-width: 600px) { dl.terms { grid-template-columns: 1fr; } dl.terms dd { margin-bottom: 6px; } }
.controls { display: grid; gap: 8px; margin: 12px 0; }
input[type=search] { width: 100%; max-width: 480px; padding: 8px 10px; font: inherit; color: var(--ink);
  background: var(--surface); border: 1px solid var(--hair); border-radius: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chips .label { font-size: .85rem; font-style: italic; margin-right: 4px; }
.chip { font: inherit; font-size: .82rem; padding: 3px 10px; border-radius: 999px; cursor: pointer;
  background: var(--surface); color: var(--ink); border: 1px dashed var(--hair); }
.chip[aria-pressed="true"] { background: var(--chip-on-bg); color: var(--chip-on-ink); border-style: solid; border-color: var(--chip-on-bg); }
.chip[aria-pressed="true"]::before { content: "\2713\00a0"; }
.clear { font: inherit; font-size: .85rem; background: none; border: none; color: var(--ink); text-decoration: underline; cursor: pointer; }
.count { font-size: .85rem; font-style: italic; }
.tablewrap { overflow-x: auto; border: 1px solid var(--hair); border-radius: 8px; position: relative; z-index: 0; background: var(--surface); isolation: isolate; }
table.main { border-collapse: collapse; width: 100%; min-width: 980px; font-size: .88rem; }
table.main > thead th, table.main > tbody > tr > td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--hair); }
table.main > thead th { background: var(--surface); }
th button { font: inherit; font-weight: 600; color: var(--ink); background: none; border: none; padding: 0; cursor: pointer; text-align: left; }
th button .arrow { display: inline-block; width: 1em; }
tr.row { cursor: pointer; }
tr.row:hover, tr.row:focus { background: var(--row-hover); outline: none; }
tr.row.open { background: var(--row-hover); }
tr.row.flash { outline: 2px solid var(--ink); outline-offset: -2px; }
td.id { font-family: var(--mono); white-space: nowrap; }
td.title { min-width: 180px; }
td.question { min-width: 220px; }
td.result { min-width: 240px; max-width: 340px; overflow-wrap: anywhere; }
td.date { white-space: nowrap; font-family: var(--mono); }
.rlist { margin: 0; padding-left: 16px; }
.rlist li { margin: 0 0 2px; }
tr.detail > td { background: var(--row-hover); padding: 12px 14px 16px; }
.detail h3 { font-size: .9rem; margin: 10px 0 4px; }
.detail h3:first-child { margin-top: 0; }
table.inner { border-collapse: collapse; font-size: .84rem; }
table.inner td, table.inner th { text-align: left; vertical-align: top; border-bottom: 1px solid var(--hair); padding: 3px 8px; }
table.prs { border-collapse: collapse; width: 100%; min-width: 720px; font-size: .88rem; }
table.prs td, table.prs th { text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid var(--hair); }
table.prs td.n { font-family: var(--mono); white-space: nowrap; }
.muted { font-style: italic; }
footer { max-width: 1180px; margin: 0 auto; padding: 0 16px 60px; font-size: .9rem; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
footer p { margin: 0; }
footer .note { background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 8px; padding: 6px 12px; }
#pause { font: inherit; font-size: .85rem; color: var(--ink); background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 999px; padding: 4px 12px; cursor: pointer; }
#pause:hover, #pause:focus-visible { border-color: var(--ink); }
button:focus-visible, a:focus-visible, summary:focus-visible, tr.row:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
@media (max-width: 700px) { h1 { font-size: 1.45rem; } main { padding-top: 24px; } .panel { padding: 14px 14px; } }
</style>
</head>
<body>
<canvas id="water" aria-hidden="true"></canvas>
<main>
<section class="panel">
<h1>kmerseek experiments</h1>
<p class="lede">What kmerseek can and cannot do, and every experiment behind each answer. Every number on this page is copied from a notebook output or a results file. Open an experiment's row to see the cell or file each number came from.</p>
<p class="meta">__META__</p>
<nav class="toc" aria-label="Sections">
  <a href="#what">What kmerseek does</a>
  <a href="#claims">What we can and cannot claim</a>
  <a href="#table">Every experiment</a>
  <a href="#prs">Open pull requests</a>
</nav>

</section>
<section class="panel">
<h2 id="what">What kmerseek does</h2>
<p>kmerseek rewrites each protein in a two-letter alphabet, hydrophobic (H) or polar (P), finds stretches of k letters that two proteins share exactly, joins them into one region without gaps, and moves a feature's label across that region. The region ends where the shared pattern ends, so a call can be as short as the feature it lands on.</p>
__HERO__

</section>
<section class="panel">
<h2 id="claims">What we can and cannot claim</h2>
<p>Twelve claims, one sentence each. Open a claim's numbers to see each value as the notebook printed it, whether it supports the claim, argues against it or sets a limit, and a button that opens that experiment in the table below.</p>
<details class="gloss">
  <summary>What IoU, identity, alphabet, k, E-value and the benchmark names mean</summary>
  <dl class="terms">
    <dt>identity</dt><dd>The share of aligned positions where the two proteins have the same amino acid. Remote homologs are usually under 30%.</dd>
    <dt>alphabet</dt><dd>How the 20 amino acids are grouped before searching. An H/P alphabet has two groups, hydrophobic and polar; names like hp_thomas_dill2 or hp_pbotc_1st_ed2 are different ways to split the letters. protein20 keeps all 20.</dd>
    <dt>k</dt><dd>The seed length, in letters of the chosen alphabet. A longer seed is less likely to match by chance, but needs a longer unbroken stretch of agreement.</dd>
    <dt>setting</dt><dd>One alphabet with one k. Notebooks call this an arm.</dd>
    <dt>IoU</dt><dd>The overlap between a call and the true feature, divided by the length they cover together. 1 is a perfect boundary; 0.5 is the usual pass mark.</dd>
    <dt>E-value</dt><dd>How many hits this good one search would find by chance. Under 1 means fewer than one expected by chance.</dd>
    <dt>Cohen's kappa</dt><dd>Agreement between two sequences' classes beyond what chance gives: 0 is chance, 1 is always the same.</dd>
    <dt>Swiss-Prot feature</dt><dd>A reviewed annotation on part of a protein: a domain, repeat, zinc finger, membrane segment, binding site.</dd>
    <dt>Pfam</dt><dd>A database of protein domain families, each defined by a profile built from an alignment.</dd>
    <dt>SCOPe40</dt><dd>A set of protein structures under 40% identity to each other, grouped into families, superfamilies and folds. Foldseek's own benchmark.</dd>
    <dt>ELM</dt><dd>Short linear motifs, 3 to 15 amino acids, usually in disordered regions.</dd>
    <dt>BHF</dt><dd>Botryllus histocompatibility factor, a tunicate protein with no known relatives by sequence search.</dd>
    <dt>dark set</dt><dd>Proteins of an invertebrate proteome (here Botryllus) that phmmer, jackhmmer and MMseqs2 all fail to match.</dd>
  </dl>
</details>
<div id="claimlist"></div>

</section>
<section class="panel">
<h2 id="table">Every experiment</h2>
<p>One row per notebook or pipeline run. Click a row to see every number with the output line it was copied from.</p>
<h3>What the verdict colours and the tag mean</h3>
<div class="legend" id="legend"></div>

<div class="controls">
  <input type="search" id="q" placeholder="Search every column" aria-label="Search every column">
  <div class="chips" id="vchips"><span class="label">Show verdict:</span></div>
  <div class="chips" id="cchips"><span class="label">Show paper claim:</span></div>
  <div><button class="clear" id="clear" type="button">Clear search and filters</button> <span class="count" id="count"></span></div>
</div>

<div class="tablewrap">
<table class="main">
<thead><tr id="head"></tr></thead>
<tbody id="body"></tbody>
</table>
</div>

</section>
<section class="panel">
<h2 id="prs">Open pull requests</h2>
<p id="prnote"></p>
<h3>Analysis repository, seanome/2024-kmerseek-analysis</h3>
<div class="tablewrap"><table class="prs" id="prs-analysis"></table></div>
<h3>Search engine, seanome/kmerseek</h3>
<p>These change how kmerseek scores and extends a match. A number on this page changes only when a notebook is rerun on the new build.</p>
<div class="tablewrap"><table class="prs" id="prs-engine"></table></div>

</section>
</main>
<footer>
<p class="note">Numbers read from notebook outputs on __DATE__. __HOSTED__ Source: <a href="https://github.com/seanome/2024-kmerseek-analysis/tree/main/data">data/</a> and <a href="https://github.com/seanome/2024-kmerseek-analysis/blob/main/scripts/build_experiments_page.py">scripts/build_experiments_page.py</a> in github.com/seanome/2024-kmerseek-analysis.</p>
<button type="button" id="pause" aria-pressed="false">pause the water</button>
</footer>

<script>
document.addEventListener("DOMContentLoaded", () => {
const REPO_URL = "https://github.com/seanome/2024-kmerseek-analysis";
const VERDICTS = [
  ["alive", "alive", "●", "the outputs show it working"],
  ["dead", "dead", "✕", "the outputs show it does not work, or it was dropped for a measured reason"],
  ["bug", "bug", "▲", "the result came from a code error"],
  ["running", "running", "◐", "the run it reads has not finished"],
  ["not-run", "not run", "○", "no outputs yet"],
];
const STATUS = [
  ["supported", "What the notebooks support"],
  ["partly", "Supported, with a limit"],
  ["not", "What the notebooks do not support"],
  ["untested", "Not tested yet"],
];
const BEARING = { for: "supports it", against: "argues against it", limit: "sets a limit", open: "no number yet" };
const COLS = [
  ["id", "id"], ["title", "title"], ["question", "what it tested"], ["dataset", "queries and targets"],
  ["comparison_tools", "compared with"], ["result", "what happened"], ["verdict", "verdict"],
  ["claim", "which paper claim"], ["date", "last changed"],
];
const DATA = JSON.parse(document.getElementById("experiments-data").textContent);
const CLAIMS = JSON.parse(document.getElementById("claims-data").textContent);
const PRS = JSON.parse(document.getElementById("prs-data").textContent);
const BYID = Object.fromEntries(DATA.map(r => [String(r.id), r]));

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const vinfo = v => VERDICTS.find(x => x[0] === v) || [v, v, "?", ""];
const badge = v => { const [k, label, mark] = vinfo(v); return `<span class="badge ${esc(k)}"><span aria-hidden="true">${mark}</span>${esc(label)}</span>`; };
const tools = r => (r.comparison_tools || []).join(", ");
const lab = x => x.label || x.name;
const resultText = r => (r.result || []).map(x => `${lab(x)}: ${x.value}`).join("; ");
const rowButton = id => `<button type="button" class="rowlink" data-open="${esc(id)}" title="Open this experiment in the table">${esc(/^\d/.test(id) ? "nb " + id : id)}</button>`;

// Residues: name column padded so the match line sits under the residues.
function residueBlock(p) {
  const split = s => { const parts = s.trim().split(/\s+/); return [parts.slice(0, -1).join(" "), parts[parts.length - 1]]; };
  const [ql, qs] = split(p.query), [tl, ts] = split(p.target);
  const w = Math.max(ql.length, tl.length);
  const text = [ql.padEnd(w) + " " + qs, " ".repeat(w + 1) + (p.match ?? p.match_line ?? ""), tl.padEnd(w) + " " + ts].join("\n")
    + (p.encoded ? "\n\n" + p.encoded : "");
  const where = p.source_cell >= 0 ? ", cell " + p.source_cell : "";
  return `<p><strong>${esc(p.pair)}</strong>: ${esc(p.identity)}${where}.${p.match_note ? " " + esc(p.match_note) : ""}${p.note ? " " + esc(p.note) : ""}</p><pre>${esc(text)}</pre>`;
}


// Claims, grouped by status.
(() => {
  const box = document.getElementById("claimlist");
  box.innerHTML = STATUS.map(([st, heading]) => {
    const cs = CLAIMS.claims.filter(c => c.status === st);
    if (!cs.length) return "";
    return `<section class="claimgroup"><h3>${esc(heading)} (${cs.length})</h3>` + cs.map(c => {
      const rows = c.evidence.map(e => {
        const r = BYID[e.row];
        const x = r && e.name ? r.result.find(y => y.name === e.name) : null;
        const v = x ? x.value : (!r ? "row missing" : e.name ? "not found" : "none yet");
        return `<tr><td>${esc(e.say)}</td><td class="v">${esc(v)}</td><td class="b">${esc(BEARING[e.bearing] || e.bearing)}</td><td>${rowButton(e.row)}</td></tr>`;
      }).join("");
      const prs = (c.open_prs || []).map(n => `<a href="${REPO_URL}/pull/${n}" target="_blank" rel="noopener">#${n}</a>`).join(", ");
      return `<article class="claim" id="claim-${esc(c.id)}"><p class="cname">${esc(c.name)}</p><p class="stmt">${esc(c.statement)}</p>` +
        `<details><summary>The ${c.evidence.length} numbers behind this claim, and whether each supports it</summary>` +
        `<div class="tablewrap"><table class="ev"><thead><tr><th>number</th><th>value as printed</th><th>bearing on the claim</th><th>experiment</th></tr></thead><tbody>${rows}</tbody></table></div></details>` +
        (c.caveats ? `<p class="cav">${esc(c.caveats)}</p>` : "") +
        (prs ? `<p class="prs">Open pull requests that could change this: ${prs}</p>` : "") + `</article>`;
    }).join("") + `</section>`;
  }).join("");
})();

// Sort keys: ids sort by their leading number, then text.
function key(r, col) {
  if (col === "id") { const m = String(r.id).match(/^(\d+)/); return m ? [0, +m[1], String(r.id)] : [1, 0, String(r.id)]; }
  if (col === "comparison_tools") return [0, 0, tools(r)];
  if (col === "result") return [0, 0, resultText(r)];
  if (col === "verdict") return [0, VERDICTS.findIndex(x => x[0] === r.verdict), ""];
  return [0, 0, String(r[col] ?? "")];
}
function cmp(a, b) { for (let i = 0; i < 3; i++) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; } return 0; }

const state = { sort: "id", dir: 1, q: "", verdicts: new Set(), claims: new Set(), open: new Set() };

document.getElementById("legend").innerHTML = VERDICTS.map(([k]) =>
  `<span>${badge(k)} <span class="what">${esc(vinfo(k)[3])}</span></span>`).join("") +
  `<span><span class="tag">changed since read</span> <span class="what">the notebook has commits after its numbers were copied; re-read it before quoting them</span></span>`;

function chip(container, value, label, set) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = label; b.setAttribute("aria-pressed", "false");
  b.addEventListener("click", () => {
    if (set.has(value)) set.delete(value); else set.add(value);
    b.setAttribute("aria-pressed", set.has(value) ? "true" : "false");
    render();
  });
  container.appendChild(b);
}
VERDICTS.forEach(([k, label]) => chip(document.getElementById("vchips"), k, label, state.verdicts));
[...new Set(DATA.map(r => r.claim))].sort().forEach(c => chip(document.getElementById("cchips"), c, c, state.claims));

document.getElementById("head").innerHTML = COLS.map(([k, label]) =>
  `<th scope="col" aria-sort="none" data-col="${k}"><button type="button" title="Sort by ${esc(label)}">${esc(label)}<span class="arrow" aria-hidden="true"></span></button></th>`).join("");
document.querySelectorAll("#head th button").forEach(b => b.addEventListener("click", () => {
  const col = b.parentElement.dataset.col;
  if (state.sort === col) state.dir = -state.dir; else { state.sort = col; state.dir = 1; }
  render();
}));

function clearFilters() {
  state.q = ""; document.getElementById("q").value = "";
  state.verdicts.clear(); state.claims.clear();
  document.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", "false"));
}
document.getElementById("q").addEventListener("input", e => { state.q = e.target.value.trim().toLowerCase(); render(); });
document.getElementById("clear").addEventListener("click", () => { clearFilters(); render(); });

function haystack(r) {
  return [r.id, r.title, r.question, r.dataset, tools(r), resultText(r), vinfo(r.verdict)[1], r.claim, r.date,
    (r.residues || []).map(p => p.pair).join(" "), r.source && r.source.path, r.changed_since_read ? "changed since read" : ""]
    .join(" \u0001 ").toLowerCase();
}

function ghLink(src) {
  if (!src || !src.path) return "";
  const url = `${REPO_URL}/blob/${encodeURI(src.branch || "main")}/${encodeURI(src.path)}`;
  return `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(src.path)} on ${esc(src.branch || "main")}</a>`;
}

function numberRows(list) {
  return list.map(x =>
    `<tr><td>${esc(lab(x))}</td><td class="val">${esc(x.value)}</td><td>${x.source_cell >= 0 ? "cell " + x.source_cell : "results file"}</td><td><code>${esc(x.output_line)}</code></td></tr>`).join("");
}

function detail(r) {
  const thead = `<tr><th>number</th><th>value as printed</th><th>where</th><th>printed line</th></tr>`;
  const rows = numberRows(r.result || []), est = numberRows(r.estimates || []);
  const res = (r.residues || []).map(residueBlock).join("");
  const claims = CLAIMS.claims.filter(c => c.evidence.some(e => e.row === String(r.id)));
  return `<h3>What happened: every number, with the output it was copied from</h3>` +
    (rows ? `<div class="tablewrap"><table class="inner">${thead}${rows}</table></div>` : `<p class="muted">No numbers: the notebook has no outputs.</p>`) +
    (est ? `<h3>Estimates (modelled, not measured)</h3><div class="tablewrap"><table class="inner">${thead}${est}</table></div>` : "") +
    (res ? `<h3>Aligned residues</h3>${res}` : "") +
    `<h3>Why this verdict</h3><p>${esc(r.verdict_basis)}</p>` +
    (claims.length ? `<h3>Claims this experiment is evidence for or against</h3><p>${claims.map(c => `<a href="#claim-${esc(c.id)}">${esc(c.name)}</a>`).join(", ")}</p>` : "") +
    (r.changed_since_read ? `<h3>Changed since read</h3><p>The numbers above were read at commit <code>${esc(r.source.commit)}</code>. ${esc(r.changed_since_read.ref)} now has commit <code>${esc(r.changed_since_read.commit)}</code> (${esc(r.changed_since_read.date)}) touching this file. Re-read the notebook before quoting these numbers.</p>` : "") +
    `<h3>Source</h3><p>${ghLink(r.source)}${r.source && r.source.commit ? ` at commit <code>${esc(r.source.commit)}</code>` : ""}` +
    `${r.pr ? `, <a href="${REPO_URL}/pull/${esc(r.pr)}" target="_blank" rel="noopener">pull request ${esc(r.pr)}</a>` : ""}</p>` +
    (r.notes ? `<h3>Notes</h3><p>${esc(r.notes)}</p>` : "");
}

function render() {
  let rows = DATA.filter(r =>
    (!state.verdicts.size || state.verdicts.has(r.verdict)) &&
    (!state.claims.size || state.claims.has(r.claim)) &&
    (!state.q || haystack(r).includes(state.q)));
  rows.sort((a, b) => state.dir * cmp(key(a, state.sort), key(b, state.sort)));
  document.querySelectorAll("#head th").forEach(th => {
    const on = th.dataset.col === state.sort;
    th.setAttribute("aria-sort", on ? (state.dir > 0 ? "ascending" : "descending") : "none");
    th.querySelector(".arrow").textContent = on ? (state.dir > 0 ? "▲" : "▼") : "";
  });
  document.getElementById("count").textContent = `${rows.length} of ${DATA.length} shown`;
  const body = document.getElementById("body");
  body.innerHTML = rows.map(r => {
    const id = String(r.id), open = state.open.has(id);
    const res = (r.result || []).length
      ? `<ul class="rlist">${r.result.slice(0, open ? r.result.length : 3).map(x => `<li>${esc(lab(x))}: <span class="val">${esc(x.value)}</span></li>`).join("")}` +
        (!open && r.result.length > 3 ? `<li class="muted">and ${r.result.length - 3} more: click the row to see them</li>` : "") + `</ul>`
      : `<span class="muted">no outputs</span>`;
    return `<tr class="row${open ? " open" : ""}" tabindex="0" aria-expanded="${open}" data-id="${esc(id)}">` +
      `<td class="id">${esc(id)}${r.changed_since_read ? '<br><span class="tag">changed since read</span>' : ""}</td><td class="title">${esc(r.title)}</td><td class="question">${esc(r.question)}</td>` +
      `<td>${esc(r.dataset)}</td><td>${esc(tools(r))}</td><td class="result">${res}</td>` +
      `<td>${badge(r.verdict)}</td><td>${esc(r.claim)}</td><td class="date">${esc(r.date)}</td></tr>` +
      (open ? `<tr class="detail"><td colspan="${COLS.length}"><div class="detail">${detail(r)}</div></td></tr>` : "");
  }).join("");
  body.querySelectorAll("tr.row").forEach(tr => {
    const toggle = () => { const id = tr.dataset.id; if (state.open.has(id)) state.open.delete(id); else state.open.add(id); render(); };
    tr.addEventListener("click", e => { if (!e.target.closest("a, button")) toggle(); });
    tr.addEventListener("keydown", e => { if (e.target === tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(); } });
  });
}

// A row button anywhere on the page opens that row in the table and scrolls to it.
function openRow(id) {
  clearFilters(); state.open.add(id); render();
  const tr = document.querySelector(`tr.row[data-id="${CSS.escape(id)}"]`);
  if (tr) { tr.scrollIntoView({ block: "start", behavior: "smooth" }); tr.classList.add("flash"); tr.focus({ preventScroll: true }); setTimeout(() => tr.classList.remove("flash"), 1600); }
}
document.addEventListener("click", e => { const b = e.target.closest("[data-open]"); if (b) { e.preventDefault(); openRow(b.dataset.open); } });

// Open pull requests.
(() => {
  const claimsFor = n => CLAIMS.claims.filter(c => (c.open_prs || []).includes(n));
  const rowsFor = n => DATA.filter(r => r.pr === n).map(r => String(r.id));
  document.getElementById("prnote").textContent = `Open on ${PRS.fetched}: ${PRS.prs.filter(p => p.repo.endsWith("analysis")).length} in the analysis repository and ${PRS.prs.filter(p => p.repo === "seanome/kmerseek").length} in the search engine. "Draft" means not ready for review.`;
  const link = p => `<a href="https://github.com/${p.repo}/pull/${p.n}" target="_blank" rel="noopener">#${p.n}</a>`;
  const an = PRS.prs.filter(p => p.repo.endsWith("analysis")).sort((a, b) => b.n - a.n);
  document.getElementById("prs-analysis").innerHTML = `<thead><tr><th>pull request</th><th>what it does</th><th>claim it could change</th><th>experiments</th><th>last updated</th></tr></thead><tbody>` +
    an.map(p => `<tr><td class="n">${link(p)}${p.draft ? " draft" : ""}</td><td>${esc(p.title)}</td>` +
      `<td>${claimsFor(p.n).map(c => `<a href="#claim-${esc(c.id)}">${esc(c.name)}</a>`).join(", ")}</td>` +
      `<td>${rowsFor(p.n).map(rowButton).join(" ")}</td><td class="n">${esc(p.updated)}</td></tr>`).join("") + `</tbody>`;
  const en = PRS.prs.filter(p => p.repo === "seanome/kmerseek").sort((a, b) => b.n - a.n);
  document.getElementById("prs-engine").innerHTML = `<thead><tr><th>pull request</th><th>what it does</th><th>last updated</th></tr></thead><tbody>` +
    en.map(p => `<tr><td class="n">${link(p)}${p.draft ? " draft" : ""}</td><td>${esc(p.title)}</td><td class="n">${esc(p.updated)}</td></tr>`).join("") + `</tbody>`;
})();

render();
});
</script>
<script type="application/json" id="experiments-data">
__DATA__
</script>
<script type="application/json" id="claims-data">
__CLAIMS__
</script>
<script type="application/json" id="prs-data">
__PRS__
</script>
<script>
__WATER__
</script>
</body>
</html>
"""


def hero_figure(records):
    """The mechanism on one real pair, drawn as inline SVG at build time.

    Uses notebook 254's printout of the notebook 244 main example (exact seeds), because it
    prints the residues and the H/P classes of the same alphabet the call used.
    """
    import html as _html
    import re as _re
    by_id = {str(r["id"]): r for r in records}
    r254, r244 = by_id.get("254"), by_id.get("244")
    if not r254:
        return ""
    p = next((x for x in r254.get("residues", []) if x["pair"].startswith("COL9A1") and "exact seeds" in x["pair"]), None)
    if p is None:
        return ""
    feat_name, f0, f1 = _re.search(r"^COL9A1 (.+?) \(\w+ (\d+)-(\d+)\)", p["pair"]).groups()
    f0, f1 = int(f0), int(f1)
    alpha, k = _re.search(r"(hp_\w+?)_k(\d+)", p["pair"]).groups()
    k = int(k)
    qname, qrange, qres = p["query"].rsplit(" ", 2)
    tname, trange, tres = p["target"].rsplit(" ", 2)
    q0, q1 = map(int, qrange.split("-"))
    t0, t1 = map(int, trange.split("-"))
    enc = p["encoded"].splitlines()
    classes_def = _re.search(r"\((H = .+?)\)", enc[0]).group(1)
    qcls = _re.search(r"\b%d ([HP]+)" % q0, enc[1]).group(1)
    tcls = _re.search(r"\b%d ([HP]+)" % t0, enc[3]).group(1)
    L = len(qres)
    assert len(tres) == len(qcls) == len(tcls) == L
    seeds = [i for i in range(L - k + 1) if qcls[i:i + k] == tcls[i:i + k]]
    def iou(name):
        x = next((y for y in (r244 or {}).get("result", []) if y["name"] == name), None)
        return x["value"] if x else "not found"
    e = _html.escape

    W, X0, X1 = 960, 150, 930
    out = [f'<svg class="mech" viewBox="0 0 {W} 440" role="img" aria-label="Human COL9A1 carries the Swiss-Prot feature {e(feat_name)} at {f0}-{f1}. '
           f'Residues {q0}-{q1} and chicken P12106 {t0}-{t1} have the same hydrophobic or polar class at all {L} positions, so all '
           f'{len(seeds)} stretches of {k} letters match exactly; kmerseek joins them into one region and places the label on chicken {t0}-{t1}.">']
    out.append('<defs><marker id="mech-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               '<path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>')
    T = lambda x, y, t, a="start", size=12, extra="": out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{a}" fill="currentColor" {extra}>{e(t)}</text>')
    # Top: both proteins to scale over a window around the feature (aa = residue number).
    w0, w1 = f0 - 12, f1 + 12
    sx = (X1 - X0) / (w1 - w0)
    xq = lambda aa: X0 + (aa - w0) * sx
    off = t0 - q0                       # chicken numbering runs this far from human over the region
    xt = lambda aa: xq(aa - off)
    yq, yt = 70, 150
    T(10, yq + 4, "human COL9A1", size=12, extra='font-weight="600"')
    T(10, yt + 4, "chicken P12106", size=12, extra='font-weight="600"')
    for y in (yq, yt):
        out.append(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" stroke="currentColor" stroke-width="1.5"/>')
    out.append(f'<rect x="{xq(f0):.1f}" y="{yq - 10}" width="{xq(f1) - xq(f0):.1f}" height="20" rx="3" fill="none" stroke="currentColor" stroke-width="1.5"/>')
    T((xq(f0) + xq(f1)) / 2, yq - 16, f"{feat_name}, {f0}-{f1} aa: the Swiss-Prot feature", "middle")
    out.append(f'<rect x="{xq(q0):.1f}" y="{yq + 14}" width="{xq(q1) - xq(q0):.1f}" height="7" fill="currentColor"/>')
    T(xq(q0) - 6, yq + 22, f"region {q0}-{q1}", "end", 11)
    out.append(f'<rect x="{xt(t0):.1f}" y="{yt - 21}" width="{xt(t1) - xt(t0):.1f}" height="7" fill="currentColor"/>')
    T(xt(t0) - 6, yt - 14, f"region {t0}-{t1}", "end", 11)
    T((xt(t0) + xt(t1)) / 2, yt + 22, f"label placed here: {feat_name}", "middle")
    for a, b in ((q0, t0), (q1, t1)):
        out.append(f'<line x1="{xq(a):.1f}" y1="{yq + 23}" x2="{xt(b):.1f}" y2="{yt - 23}" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3"/>')
    out.append(f'<line x1="{(xq(q0) + xq(q1)) / 2:.1f}" y1="{yq + 25}" x2="{(xt(t0) + xt(t1)) / 2:.1f}" y2="{yt - 25}" stroke="currentColor" stroke-width="1.5" marker-end="url(#mech-arrow)"/>')
    T((xq(q1) + 10), (yq + yt) / 2 + 4, f"same {L} positions, no gaps: the label moves across", "start", 11)
    for aa in (w0, w1):   # residue numbers at the ends of the drawn window: above the human line, below the chicken one
        T(xq(aa), yq - 8, f"{aa}", "middle", 10)
        T(xt(aa + off), yt + 16, f"{aa + off}", "middle", 10)
    # Bottom: the region zoomed to one column per residue.
    cw = 22
    zx = lambda i: X0 + 10 + i * cw + cw / 2
    rows = [(225, f"human residues from {q0}", qres, ""), (247, "H/P classes", qcls, ""), (282, "H/P classes", tcls, ""), (304, f"chicken residues from {t0}", tres, "")]
    T(10, 206, f"Zoom on {L} positions; {alpha}, k = {k}; {classes_def}", "start", 11, 'font-style="italic"')
    for y, lab, seq, _ in rows:
        T(X0 - 8, y, lab, "end", 11)
        for i, ch in enumerate(seq):
            T(zx(i), y, ch, "middle", 13, 'style="font-family:var(--mono)"')
    T(X0 - 8, 266, "same class", "end", 11)
    for i in range(L):
        if qcls[i] == tcls[i]:
            out.append(f'<line x1="{zx(i):.1f}" y1="{254}" x2="{zx(i):.1f}" y2="{270}" stroke="currentColor" stroke-width="1.5"/>')
    T(zx(L - 1) + cw / 2 + 4, 225, f"{q1}", "start", 10)
    T(zx(L - 1) + cw / 2 + 4, 304, f"{t1}", "start", 10)
    # Seeds: first and last shared k-letter stretch, then the region they join into.
    def bracket(i, y, text):
        a, b = zx(i) - cw / 2 + 2, zx(i + k - 1) + cw / 2 - 2
        out.append(f'<path d="M{a:.1f} {y - 6} V{y} H{b:.1f} V{y - 6}" fill="none" stroke="currentColor" stroke-width="1.5"/>')
        T((a + b) / 2, y + 14, text, "middle", 11)
    if seeds:
        bracket(seeds[0], 330, f"first shared {k}-letter seed")
        if seeds[-1] != seeds[0]:
            bracket(seeds[-1], 370, f"last of {len(seeds)} shared seeds")
        a, b = zx(seeds[0]) - cw / 2, zx(seeds[-1] + k - 1) + cw / 2
        out.append(f'<rect x="{a:.1f}" y="400" width="{b - a:.1f}" height="7" fill="currentColor"/>')
        T(X0 - 8, 408, "region", "end", 11)
        T(a, 426, f"the seeds joined into one region: {q0}-{q1} on human, {t0}-{t1} on chicken", "start", 11)
    out.append("</svg>")
    legend = ('<div class="legend">'
              '<span><svg width="28" height="10" aria-hidden="true"><line x1="0" y1="5" x2="28" y2="5" stroke="currentColor" stroke-width="1.5"/></svg>protein</span>'
              '<span><svg width="28" height="14" aria-hidden="true"><rect x="1" y="1" width="26" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>annotated feature</span>'
              '<span><svg width="28" height="10" aria-hidden="true"><rect x="0" y="2" width="28" height="6" fill="currentColor"/></svg>kmerseek region</span>'
              f'<span><svg width="28" height="10" aria-hidden="true"><path d="M1 2V8H27V2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>one shared seed of {k} letters</span>'
              '<span><svg width="10" height="16" aria-hidden="true"><line x1="5" y1="0" x2="5" y2="16" stroke="currentColor" stroke-width="1.5"/></svg>same H/P class</span>'
              '</div>')
    cap = (f"The main example of notebook 244: the human feature {feat_name} placed on chicken P12106. "
           f"All {L} positions keep their class, so every one of the {len(seeds)} stretches of {k} letters is shared at the same offset; "
           f"the region is their union and has no gaps. Of the residues themselves, {p['identity'].replace(' identical', '')} are the same amino acid. "
           f"Boundary overlap with the feature (IoU, the overlap divided by the span both cover): kmerseek {iou('hero_kmerseek_call')}, "
           f"Foldseek {iou('hero_foldseek_call_iou')}, Reseek {iou('hero_reseek_call_iou')} (notebook 244). "
           f"Residues and classes as printed in notebook 254, cell {p['source_cell']}.")
    return (f'<figure class="mech">{legend}<div class="tablewrap" style="background:var(--surface);padding:6px">{"".join(out)}</div>'
            f'<figcaption>{e(cap)} <button type="button" class="rowlink" data-open="244">nb 244</button> '
            f'<button type="button" class="rowlink" data-open="254">nb 254</button></figcaption></figure>')


def check_claims(records, claims):
    by_id = {str(r["id"]): r for r in records}
    missing = []
    for c in claims["claims"]:
        for e in c["evidence"]:
            r = by_id.get(e["row"])
            if r is None:
                missing.append(f"{c['id']}: no row {e['row']}")
            elif e["name"] is not None and not any(x["name"] == e["name"] for x in r["result"]):
                missing.append(f"{c['id']}: row {e['row']} has no number named {e['name']!r}")
    if missing:
        sys.exit("claims.json names numbers that are not in experiments.json:\n" + "\n".join(missing))


def embed(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1).replace("</", "<\\/")


def git(*args):
    """Run git in the repo; return stdout, or None if git or the object is missing."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def page_dates():
    added = git("log", "--diff-filter=A", "--format=%cs", "--", "notebooks/experiments.html")
    return {
        "first": added.splitlines()[-1] if added else None,
        "numbers": git("log", "-1", "--format=%cs", "--", "data/experiments.json"),
        "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def mark_changed_since_read(records):
    """Set r["changed_since_read"] when the file a row was read from differs today.

    Compares the file at the recorded commit with the file on the same branch now
    (origin first, then a local branch, then origin/main for a merged and deleted
    branch). Returns how many rows could be checked.
    """
    checked = 0
    for r in records:
        src = r.get("source") or {}
        path, commit, branch = src.get("path"), src.get("commit"), src.get("branch") or "main"
        r.pop("changed_since_read", None)
        if not path or not commit:
            continue
        then = git("rev-parse", f"{commit}:{path}")
        if then is None:
            continue
        for ref in (f"origin/{branch}", branch, "origin/main"):
            now = git("rev-parse", f"{ref}:{path}")
            if now is not None:
                break
        else:
            continue
        checked += 1
        if now != then:
            last = git("log", "-1", "--format=%h %cs", ref, "--", path) or "? ?"
            sha, date = last.split(" ", 1)
            r["changed_since_read"] = {"ref": ref.removeprefix("origin/"), "commit": sha, "date": date}
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, help="write the GitHub Pages copy into this directory")
    ap.add_argument("--artifact", type=Path, help="write a body-only copy for the claude.ai artifact shelf")
    args = ap.parse_args()

    records = json.loads(DATA.read_text())
    claims = json.loads(CLAIMS.read_text())
    prs = json.loads(PRS.read_text())
    check_claims(records, claims)
    checked = mark_changed_since_read(records)
    n_changed = sum(1 for r in records if r.get("changed_since_read"))
    d = page_dates()

    meta = [f"First published {d['first']}." if d["first"] else "",
            f"Numbers last changed {d['numbers']}." if d["numbers"] else "",
            f"Page built {d['built']}.",
            f"Open pull requests as of {prs['fetched']}."]
    if checked:
        meta.append(f"{n_changed} of {checked} experiments were read from a notebook that has changed since; "
                    "they carry the tag \"changed since read\".")
    hosted = ("This copy is rebuilt every day and whenever the data changes on main, so the dates, the open "
              "pull requests and the \"changed since read\" tags are current; the numbers change only when "
              "data/experiments.json does.") if args.site else ""
    here = Path(__file__).resolve().parent
    water = (here / "experiments_page_water.js").read_text()
    fonts = (here / "experiments_page_fonts.css").read_text()
    html = (TEMPLATE.replace("__HERO__", hero_figure(records)).replace("__WATER__", water).replace("__FONTS__", fonts).replace("__META__", " ".join(m for m in meta if m))
            .replace("__DATE__", d["numbers"] or "an unknown date").replace("__HOSTED__", hosted)
            .replace("__DATA__", embed(records)).replace("__CLAIMS__", embed(claims)).replace("__PRS__", embed(prs)))

    if args.site:
        args.site.mkdir(parents=True, exist_ok=True)
        for name in ("index.html", "experiments.html"):
            (args.site / name).write_text(html)
        print(f"wrote {args.site}/index.html and experiments.html")
    else:
        PAGE.write_text(html)
        print(f"wrote {PAGE.relative_to(REPO)}")
    print(f"{len(records)} records, {len(claims['claims'])} claims, {len(prs['prs'])} open pull requests; "
          f"{n_changed} of {checked} checked rows changed since read; dates {d}")
    if args.artifact:
        title = html[html.index("<title>"): html.index("</title>") + len("</title>")]
        style = html[html.index("<style>"): html.index("</style>") + len("</style>")]
        body = html[html.index("<body>") + len("<body>"): html.index("</body>")]
        args.artifact.write_text(f"{title}\n{style}\n{body}")
        print(f"wrote artifact copy {args.artifact}")


if __name__ == "__main__":
    main()
