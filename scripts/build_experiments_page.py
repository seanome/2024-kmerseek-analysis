"""Write notebooks/experiments.html from data/experiments.json, data/claims.json and data/open_prs.json.

The page is one file with the data embedded at the bottom, so it opens offline.
Every number a claim shows is looked up from experiments.json by row id and
number name; the build stops if a claim names a number that is not there.
Run from the repo root:

    python scripts/build_experiments_page.py [artifact_copy.html]

With the argument it also writes a body-only copy for the claude.ai artifact
shelf, which supplies its own html/head/body.
"""

import datetime
import json
import os
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
<style>
:root {
  --bg: #ffffff; --ink: #1b1f24; --line: #d5d9de; --row-hover: #f4f6f8; --panel: #f7f8fa;
  --alive: #1e7b34; --dead: #7a7f86; --bug: #c85a00; --running: #1f5fbf; --notrun: #1b1f24;
  --chip-on-bg: #1b1f24; --chip-on-ink: #ffffff;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15181c; --ink: #e8eaed; --line: #3a4048; --row-hover: #1f242a; --panel: #1b1f24;
    --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #e8eaed;
    --chip-on-bg: #e8eaed; --chip-on-ink: #15181c; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #15181c; --ink: #e8eaed; --line: #3a4048; --row-hover: #1f242a; --panel: #1b1f24;
  --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #e8eaed;
  --chip-on-bg: #e8eaed; --chip-on-ink: #15181c; color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { margin: 0; overflow-x: hidden; }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; text-wrap: balance; }
h3 { font-size: 1rem; margin: 20px 0 6px; }
p { max-width: 75ch; }
.lede { margin: 0 0 8px; font-size: 1.02rem; }
.aside { font-style: italic; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.9rem; margin: 8px 0 0; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 0 0 8px; }
.legend span.what { font-style: italic; font-size: 0.85rem; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px; border-radius: 4px;
  font-size: 0.8rem; font-weight: 600; white-space: nowrap; border: 1.5px solid transparent; color: #fff; }
.badge.alive { background: var(--alive); }
.badge.dead { background: var(--dead); }
.badge.bug { background: var(--bug); }
.badge.running { background: var(--running); }
.badge.not-run { background: transparent; color: var(--notrun); border-color: var(--notrun); }
pre { font-family: var(--mono); font-size: 0.8rem; line-height: 1.35; margin: 4px 0 8px; padding: 8px; overflow-x: auto;
  border: 1px solid var(--line); border-radius: 4px; background: var(--bg); white-space: pre; }
.steps { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin: 10px 0; padding: 0; list-style: none; counter-reset: s; }
.steps li { border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: var(--panel); counter-increment: s; }
.steps li::before { content: counter(s) ". "; font-weight: 700; }
.claimgroup { margin: 18px 0 0; }
.claim { border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; margin: 10px 0; background: var(--panel); }
.claim .cname { font-weight: 700; font-size: 1rem; margin: 0 0 4px; }
.claim .stmt { margin: 0 0 8px; max-width: 90ch; }
.claim .cav { margin: 6px 0 0; font-style: italic; max-width: 90ch; }
.claim .prs { margin: 6px 0 0; font-size: 0.88rem; }
.ev { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.ev td, .ev th { text-align: left; vertical-align: top; padding: 4px 6px; border-top: 1px solid var(--line); }
.ev th { font-weight: 600; }
.ev td.v { font-family: var(--mono); white-space: nowrap; }
.ev td.b { white-space: nowrap; }
.rowlink { font: inherit; font-family: var(--mono); background: none; border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; color: var(--ink); cursor: pointer; white-space: nowrap; }
.rowlink:hover, .rowlink:focus-visible { border-color: var(--ink); }
details.gloss { border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; margin: 14px 0; }
details.gloss summary { cursor: pointer; font-weight: 600; }
dl.terms { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 10px 0 4px; }
dl.terms dt { font-weight: 600; }
dl.terms dd { margin: 0; max-width: 80ch; }
@media (max-width: 600px) { dl.terms { grid-template-columns: 1fr; } dl.terms dd { margin-bottom: 6px; } }
.controls { display: grid; gap: 8px; margin: 12px 0; }
input[type=search] { width: 100%; max-width: 480px; padding: 8px 10px; font: inherit; color: var(--ink);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chips .label { font-size: 0.85rem; font-style: italic; margin-right: 4px; }
.chip { font: inherit; font-size: 0.82rem; padding: 3px 10px; border-radius: 999px; cursor: pointer;
  background: transparent; color: var(--ink); border: 1px dashed var(--line); }
.chip[aria-pressed="true"] { background: var(--chip-on-bg); color: var(--chip-on-ink); border-style: solid; border-color: var(--chip-on-bg); }
.chip[aria-pressed="true"]::before { content: "\2713\00a0"; }
.clear { font: inherit; font-size: 0.82rem; background: none; border: none; color: var(--ink); text-decoration: underline; cursor: pointer; }
.count { font-size: 0.85rem; font-style: italic; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; position: relative; z-index: 0; }
table.main { border-collapse: collapse; width: 100%; min-width: 980px; font-size: 0.86rem; }
table.main > thead th, table.main > tbody > tr > td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }
table.main > thead th { background: var(--bg); }
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
.val { font-family: var(--mono); }
tr.detail > td { background: var(--row-hover); padding: 12px 14px 16px; }
.detail h3 { font-size: 0.85rem; margin: 10px 0 4px; }
.detail h3:first-child { margin-top: 0; }
table.inner { border-collapse: collapse; font-size: 0.82rem; }
table.inner td, table.inner th { text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); padding: 3px 8px; }
table.prs { border-collapse: collapse; width: 100%; min-width: 720px; font-size: 0.86rem; }
table.prs td, table.prs th { text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid var(--line); }
table.prs td.n { font-family: var(--mono); white-space: nowrap; }
a { color: var(--ink); }
.muted { font-style: italic; }
footer { margin: 24px 0; font-size: 0.85rem; font-style: italic; }
button:focus-visible, a:focus-visible, summary:focus-visible, tr.row:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
@media (max-width: 700px) { h1 { font-size: 1.25rem; } }
</style>
</head>
<body>
<main>
<h1>kmerseek experiments</h1>
<p class="lede">What kmerseek can and cannot do, and every experiment behind each answer. Every number on this page is copied from a notebook output or a results file. Open an experiment's row to see the cell or file each number came from.</p>
<nav class="toc" aria-label="Sections">
  <a href="#what">What kmerseek does</a>
  <a href="#claims">What we can and cannot claim</a>
  <a href="#table">Every experiment</a>
  <a href="#prs">Open pull requests</a>
</nav>

<h2 id="what">What kmerseek does</h2>
<ol class="steps">
  <li>Rewrite every residue of both proteins in a reduced alphabet. The main one has two letters: hydrophobic (H) or polar (P).</li>
  <li>Find a stretch of k letters that is identical in the query and in a target protein. This exact shared stretch is the seed.</li>
  <li>Report the shared stretch as a region, without gaps, and copy the target's annotated feature onto the matching stretch of the query.</li>
</ol>
<p>Because the region ends where the shared pattern ends, a call can be as short as the feature it lands on. Below is the pair notebook 244 uses as its main example, as printed there, with each tool's IoU on this feature.</p>
<div id="hero"></div>

<h2 id="claims">What we can and cannot claim</h2>
<p>Each claim below is one sentence, the numbers behind it, and whether each number argues for it, against it, or sets a limit on it. The row button opens that experiment in the table below.</p>
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

<h2 id="table">Every experiment</h2>
<p>One row per notebook or pipeline run. Click a row to see every number with the output line it was copied from.</p>
<h3>What the verdict colours mean</h3>
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

<h2 id="prs">Open pull requests</h2>
<p id="prnote"></p>
<h3>Analysis repository, seanome/2024-kmerseek-analysis</h3>
<div class="tablewrap"><table class="prs" id="prs-analysis"></table></div>
<h3>Search engine, seanome/kmerseek</h3>
<p>These change how kmerseek scores and extends a match. A number on this page changes only when a notebook is rerun on the new build.</p>
<div class="tablewrap"><table class="prs" id="prs-engine"></table></div>

<footer>Numbers read from notebook outputs on __DATE__.</footer>
</main>

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

// What kmerseek does: the notebook 244 main example.
(() => {
  const r = BYID["244"]; if (!r) return;
  const p = (r.residues || []).find(x => /P12106/.test(x.pair)) || (r.residues || [])[0];
  if (!p) return;
  const iou = n => (r.result.find(x => x.name === n) || {}).value;
  document.getElementById("hero").innerHTML = residueBlock(p) +
    `<p>kmerseek's call on this feature: ${esc(iou("hero_kmerseek_call"))}. Foldseek: ${esc(iou("hero_foldseek_call_iou"))}. Reseek: ${esc(iou("hero_reseek_call_iou"))}. ${rowButton("244")}</p>`;
})();

// Claims, grouped by status.
(() => {
  const box = document.getElementById("claimlist");
  box.innerHTML = STATUS.map(([st, heading]) => {
    const cs = CLAIMS.claims.filter(c => c.status === st);
    if (!cs.length) return "";
    return `<section class="claimgroup"><h3>${esc(heading)}</h3>` + cs.map(c => {
      const rows = c.evidence.map(e => {
        const r = BYID[e.row];
        const x = r && e.name ? r.result.find(y => y.name === e.name) : null;
        const v = x ? x.value : (!r ? "row missing" : e.name ? "not found" : "none yet");
        return `<tr><td>${esc(e.say)}</td><td class="v">${esc(v)}</td><td class="b">${esc(BEARING[e.bearing] || e.bearing)}</td><td>${rowButton(e.row)}</td></tr>`;
      }).join("");
      const prs = (c.open_prs || []).map(n => `<a href="${REPO_URL}/pull/${n}" target="_blank" rel="noopener">#${n}</a>`).join(", ");
      return `<article class="claim" id="claim-${esc(c.id)}"><p class="cname">${esc(c.name)}</p><p class="stmt">${esc(c.statement)}</p>` +
        `<div class="tablewrap"><table class="ev"><thead><tr><th>number</th><th>value as printed</th><th>bearing on the claim</th><th>experiment</th></tr></thead><tbody>${rows}</tbody></table></div>` +
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
  `<span>${badge(k)} <span class="what">${esc(vinfo(k)[3])}</span></span>`).join("");

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
    (r.residues || []).map(p => p.pair).join(" "), r.source && r.source.path]
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
      `<td class="id">${esc(id)}</td><td class="title">${esc(r.title)}</td><td class="question">${esc(r.question)}</td>` +
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
</body>
</html>
"""


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


def main():
    records = json.loads(DATA.read_text())
    claims = json.loads(CLAIMS.read_text())
    prs = json.loads(PRS.read_text())
    check_claims(records, claims)
    date = datetime.date.fromtimestamp(os.path.getmtime(DATA)).isoformat()
    html = (TEMPLATE.replace("__DATE__", date).replace("__DATA__", embed(records))
            .replace("__CLAIMS__", embed(claims)).replace("__PRS__", embed(prs)))
    PAGE.write_text(html)
    print(f"wrote {PAGE.relative_to(REPO)}: {len(records)} records, {len(claims['claims'])} claims, "
          f"{len(prs['prs'])} open pull requests, data dated {date}")
    if len(sys.argv) > 1:
        title = html[html.index("<title>"): html.index("</title>") + len("</title>")]
        style = html[html.index("<style>"): html.index("</style>") + len("</style>")]
        body = html[html.index("<body>") + len("<body>"): html.index("</body>")]
        Path(sys.argv[1]).write_text(f"{title}\n{style}\n{body}")
        print(f"wrote artifact copy {sys.argv[1]}")


if __name__ == "__main__":
    main()
