"""Write notebooks/experiments.html from data/experiments.json, claims.json, open_prs.json, paper_map.json and pr_review.json.

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
PAPER_MAP = REPO / "data" / "paper_map.json"
REVIEW = REPO / "data" / "pr_review.json"
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
[hidden] { display: none !important; }
html, body { margin: 0; overflow-x: clip; }
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
/* The plan: step bar, first step, overview tiles, the whole plan. */
#stepbar { position: sticky; top: 0; z-index: 5; background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 10px; padding: 8px 10px 6px;
  box-shadow: var(--shadow); backdrop-filter: blur(12px) saturate(1.15); -webkit-backdrop-filter: blur(12px) saturate(1.15); min-width: 0; }
.chevrons { display: flex; gap: 2px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: thin; }
.chev { font: inherit; font-size: .84rem; flex: 1 0 auto; min-width: 104px; padding: 6px 18px 6px 22px; cursor: pointer; text-align: left; line-height: 1.25;
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%, 12px 50%);
  background: var(--surface); color: var(--ink); border: none; box-shadow: inset 0 0 0 1.5px var(--ink); }
.chev:first-child { clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%); padding-left: 12px; }
.chev .num { font-weight: 700; margin-right: 4px; }
.chev .cnt { display: block; font-size: .76rem; }
.chev.done { background: var(--hair); color: var(--ink); box-shadow: none; }
.chev.current { background: var(--chip-on-bg); color: var(--chip-on-ink); box-shadow: none; }
.chev.viewing .nm { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 3px; }
.stepkey { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .8rem; margin: -10px 0 0; padding: 4px 10px; background: var(--glass); border-radius: 8px; }
main [id] { scroll-margin-top: 76px; }
@media (max-width: 820px) { #stepbar { position: static; } main [id] { scroll-margin-top: 8px; } }
.stepkey i { display: inline-block; width: 14px; height: 10px; vertical-align: -1px; margin-right: 4px; }
.firstcard { border: 1.5px solid var(--ink); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.firstcard .when { font-size: 1.05rem; margin: 2px 0 10px; max-width: 80ch; }
.progress { display: grid; grid-template-columns: max-content 1fr max-content; gap: 8px; align-items: center; font-size: .85rem; max-width: 560px; }
.progress .track { height: 8px; border: 1px solid var(--ink); border-radius: 4px; overflow: hidden; }
.progress .fill { height: 100%; background: var(--ink); }
.btnrow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.btn { font: inherit; font-size: .9rem; padding: 5px 12px; border-radius: 6px; cursor: pointer; background: var(--surface); color: var(--ink); border: 1.5px solid var(--ink); }
.btn.primary { background: var(--chip-on-bg); color: var(--chip-on-ink); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile { font: inherit; text-align: left; cursor: pointer; border: 1px solid var(--hair); border-radius: 10px; padding: 10px 12px; background: var(--surface); color: var(--ink); display: grid; gap: 4px; }
.tile:hover, .tile[aria-pressed="true"] { border-color: var(--ink); }
.tile[aria-pressed="true"] { box-shadow: inset 0 0 0 1px var(--ink); }
.tile svg { width: 44px; height: 32px; }
.tile .tn { font-weight: 700; font-size: .92rem; line-height: 1.2; }
.tile .tc { font-size: .8rem; }
.tile .thin { height: 4px; border-radius: 2px; background: var(--hair); overflow: hidden; }
.tile .thin span { display: block; height: 100%; background: var(--ink); }
ol.plan { list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 4px; }
ol.plan > li { border: 1px solid var(--hair); border-radius: 8px; background: var(--surface); }
ol.plan > li > button { font: inherit; width: 100%; text-align: left; background: none; border: none; color: var(--ink); padding: 7px 10px; cursor: pointer; display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline; }
ol.plan > li.open { border-color: var(--ink); }
ol.plan .pn { font-weight: 700; }
ol.plan .after { margin-left: auto; font-size: .82rem; font-style: italic; }
ol.plan .body { padding: 0 12px 10px; font-size: .9rem; }
ol.plan .body p { margin: 0 0 6px; }
.prline { display: flex; gap: 8px; flex-wrap: wrap; align-items: baseline; padding: 3px 0; border-top: 1px solid var(--hair); }
.prline .why { font-size: .82rem; font-style: italic; }
/* The map. */
.maplegend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: .84rem; margin: 6px 0 10px; }
.maplegend .node { pointer-events: none; }
.mapbar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin: 0 0 10px; font-size: .9rem; }
.mapbar label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.cols { display: grid; grid-template-columns: 1.25fr 1.25fr .9fr 1fr; gap: 12px; align-items: start; }
.col { border: 1px solid var(--hair); border-radius: 10px; background: var(--surface); padding: 8px 10px; min-width: 0; }
.col h3 { margin: 0 0 6px; font-size: .95rem; display: flex; justify-content: space-between; gap: 6px; }
.col h3 .k { font-weight: 400; font-size: .8rem; font-style: italic; }
.col h4 { margin: 10px 0 4px; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.col .list { display: flex; flex-direction: column; gap: 4px; }
.col .list.chips2 { flex-direction: row; flex-wrap: wrap; }
.node { font: inherit; font-size: .84rem; text-align: left; cursor: pointer; color: var(--ink); background: var(--surface); border: 1px dashed var(--hair);
  border-radius: 6px; padding: 3px 7px; line-height: 1.3; overflow-wrap: anywhere; }
.node .sub { display: block; font-size: .76rem; font-style: italic; }
.node.lit { border: 1.5px solid var(--ink); }
.node.sel { background: var(--chip-on-bg); color: var(--chip-on-ink); border: 1.5px solid var(--chip-on-bg); }
.node.dim { opacity: .38; }
.cols.only .node.dim, .cols.only h4.dim { display: none; }
.fold { font-size: .84rem; font-style: italic; margin: 2px 0; }
.fold button { font: inherit; font-style: normal; background: none; border: none; text-decoration: underline; cursor: pointer; color: var(--ink); padding: 0; }
#mapdetail { border: 1.5px solid var(--ink); border-radius: 10px; background: var(--surface); padding: 12px 14px; margin: 0 0 12px; }
#mapdetail h3 { margin: 0 0 4px; font-size: 1.05rem; }
#mapdetail p { margin: 4px 0; max-width: 90ch; }
#mapdetail dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; margin: 8px 0 0; font-size: .9rem; }
#mapdetail dt { font-weight: 600; }
#mapdetail dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
#mapdetail .figs a { margin-right: 10px; font-family: var(--mono); font-size: .82rem; }
@media (max-width: 860px) { .cols { grid-template-columns: 1fr; } #mapdetail dl { grid-template-columns: 1fr; } #mapdetail dd { margin-bottom: 6px; } }
@media (max-width: 700px) { h1 { font-size: 1.45rem; } main { padding-top: 24px; } .panel { padding: 14px 14px; } }
</style>
</head>
<body>
<canvas id="water" aria-hidden="true"></canvas>
<main>
<section class="panel">
<h1>kmerseek experiments</h1>
<p class="lede">How each pull request, notebook and Sherlock run feeds the paper, what to review first, and every experiment behind each claim. Every number on this page is copied from a notebook output, a results file, a pull request, or the progress summary of 2026-10-13 (docs/progress_2026-10-13.md, PR 127).</p>
<p class="meta">__META__</p>
<nav class="toc" aria-label="Sections">
  <a href="#first">Do this first</a>
  <a href="#map">What feeds the paper</a>
  <a href="#what">What kmerseek does</a>
  <a href="#claims">What we can and cannot claim</a>
  <a href="#table">Every experiment</a>
  <a href="#prs">Open pull requests</a>
</nav>

</section>
<nav id="stepbar" aria-label="Review plan steps">
  <div class="chevrons" id="chevrons"></div>
</nav>
<div class="stepkey"><span><i style="background:var(--hair)"></i>done (you marked it)</span><span><i style="background:var(--chip-on-bg)"></i>current step</span><span><i style="box-shadow:inset 0 0 0 1.5px var(--ink)"></i>to do</span><span><u>underlined</u>: the step shown below</span><span>The number is how many pull requests are in the step.</span></div>
<section class="panel">
<h2 id="first">Do this first</h2>
<div id="firstcard" class="firstcard"></div>
<h3>The parts of the paper</h3>
<p class="meta">One tile per part. The bar is the share of its pull requests you have reviewed. Click a tile to light up everything behind it in the map below.</p>
<div class="tiles" id="tiles"></div>
<h3>The whole plan</h3>
<p class="meta">Every open pull request is in exactly one step: the first step whose rule it meets, in this order. Only the step you are looking at is open.</p>
<ol class="plan" id="plan"></ol>
</section>
<section class="panel">
<h2 id="map">What feeds the paper</h2>
<p>Click anything: a figure, a claim, a pull request, a notebook or a Sherlock run. Everything linked to it lights up in the other columns, and the box above the columns says what it is and where it stands.</p>
<div class="maplegend"><span class="node sel">picked</span><span class="node lit">linked to what you picked</span><span class="node">not linked</span><span class="badge running"><span aria-hidden="true">◐</span>in the queue</span><span>a Sherlock run that has not finished: running or waiting</span></div>
<div class="mapbar"><label><input type="checkbox" id="onlylinked" checked> hide what is not linked</label><button type="button" class="clear" id="mapclear">clear the pick</button><span class="count" id="mapnote"></span></div>
<div id="mapdetail"></div>
<div class="cols only" id="cols">
  <div class="col"><h3>Paper <span class="k">figures, examples, benchmarks, claims</span></h3><div id="col-paper"></div></div>
  <div class="col"><h3>Pull requests <span class="k" id="k-prs"></span></h3><div id="col-prs"></div></div>
  <div class="col"><h3>Notebooks <span class="k" id="k-nbs"></span></h3><div id="col-nbs"></div></div>
  <div class="col"><h3>Sherlock runs</h3><p class="fold" id="k-runs"></p><div id="col-runs"></div></div>
</div>
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
<section class="panel">
<h2 id="how">How this page decides</h2>
<ul>
<li><strong>Review status</strong> comes from GitHub, read on __REVIEWREAD__ with pr_status.py from the weekly-pr-status skill. "Not reviewed": no review and no review comment from Olga. "Look again": she reviewed, and new commits came after. "Reviewed, no commits since": nothing changed after her review. What stops a pull request merging is listed in this order: base branch already in main, base pull request closed, the pull request it is built on, merge conflicts, failing checks.</li>
<li><strong>Notebooks and figures of a pull request</strong> are the .ipynb and image files it changes, read from GitHub on __PRDATE__. A pull request built on another one lists only the files the lower one does not have. A pull request with no notebook file of its own takes the notebook named in its title ("Notebook 284"). Two different notebooks are numbered 245; the P66/CD47 one is shown as nb 245-p66.</li>
<li><strong>Which part of the paper a pull request feeds</strong> is written by hand in data/paper_map.json, from docs/progress_2026-10-13.md (PR 127). Pull requests 126 and 128 to 131 were placed from their titles.</li>
<li><strong>Sherlock runs</strong> are from squeue and each run's Nextflow trace file, read __RUNSDATE__. A run's link to a figure is marked when it is inferred rather than read by a notebook.</li>
<li><strong>Steps.</strong> A pull request goes in the first step whose rule it meets. "Mark done" is saved in this page's shared store, so it shows on every device; it only appears in the claude.ai viewer.</li>
</ul>
<details class="gloss"><summary>Why the page opens on one step (research)</summary>
<ul>
<li>People hold about four separate things in mind at once (Cowan 2001), so the page opens on one step and folds the rest.</li>
<li>Making a specific plan for unfinished tasks stops them pulling at attention (Masicampo and Baumeister 2011), so every pull request is in a step.</li>
<li>Plans written as "when this happens, I will do that" are followed more often than goals alone (Gollwitzer and Sheeran 2006), so each step is written that way.</li>
<li>Seeing progress on meaningful work motivates (Amabile and Kramer 2011), so the first card shows the count going down.</li>
<li>More choices do not always overwhelm: a meta-analysis found a mean effect near zero (Scheibehenne, Greifeneder and Todd 2010). So nothing is hidden, only folded.</li>
</ul>
<p class="meta">Amabile, T., and S. Kramer. <em>The Progress Principle.</em> Harvard Business Review Press, 2011.
Cowan, N. "The Magical Number 4 in Short-Term Memory." <em>Behavioral and Brain Sciences</em> 24 (2001): 87-114. https://doi.org/10.1017/s0140525x01003922.
Gollwitzer, P. M., and P. Sheeran. "Implementation Intentions and Goal Achievement: A Meta-Analysis." <em>Advances in Experimental Social Psychology</em> 38 (2006): 69-119. https://doi.org/10.1016/S0065-2601(06)38002-1.
Masicampo, E. J., and R. F. Baumeister. "Consider It Done!" <em>Journal of Personality and Social Psychology</em> 101 (2011): 667-83. https://doi.org/10.1037/a0024192.
Scheibehenne, B., R. Greifeneder, and P. M. Todd. "Can There Ever Be Too Many Options?" <em>Journal of Consumer Research</em> 37 (2010): 409-25. https://doi.org/10.1086/651235.</p>
</details>
</section>
</main>
<footer>
<p class="note">Experiment numbers read from notebook outputs on __DATE__. __HOSTED__ Source: <a href="https://github.com/seanome/2024-kmerseek-analysis/tree/main/data">data/</a> and <a href="https://github.com/seanome/2024-kmerseek-analysis/blob/main/scripts/build_experiments_page.py">scripts/build_experiments_page.py</a> in github.com/seanome/2024-kmerseek-analysis.</p>
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
      return `<article class="claim" id="claim-${esc(c.id)}" data-comment-target><p class="cname">${esc(c.name)} <button type="button" class="rowlink" data-comment-btn hidden title="Opens the claude.ai comment box on this claim">Comment on this claim</button></p><p class="stmt">${esc(c.statement)}</p>` +
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
  const REVIEW = JSON.parse(document.getElementById("review-data").textContent);
  const rev = p => { const r = (REVIEW.repos[p.repo] || {})[String(p.n)]; return r ? esc([r.review].concat(r.blockers).join("; ")) : ""; };
  document.getElementById("prs-analysis").innerHTML = `<thead><tr><th>pull request</th><th>what it does</th><th>your review, and what stops it merging</th><th>claim it could change</th><th>experiments</th><th>last updated</th></tr></thead><tbody>` +
    an.map(p => `<tr><td class="n">${link(p)}${p.draft ? " draft" : ""}</td><td>${esc(p.title)}</td><td>${rev(p)}</td>` +
      `<td>${claimsFor(p.n).map(c => `<a href="#claim-${esc(c.id)}">${esc(c.name)}</a>`).join(", ")}</td>` +
      `<td>${rowsFor(p.n).map(rowButton).join(" ")}</td><td class="n">${esc(p.updated)}</td></tr>`).join("") + `</tbody>`;
  const en = PRS.prs.filter(p => p.repo === "seanome/kmerseek").sort((a, b) => b.n - a.n);
  document.getElementById("prs-engine").innerHTML = `<thead><tr><th>pull request</th><th>what it does</th><th>your review, and what stops it merging</th><th>last updated</th></tr></thead><tbody>` +
    en.map(p => `<tr><td class="n">${link(p)}${p.draft ? " draft" : ""}</td><td>${esc(p.title)}</td><td>${rev(p)}</td><td class="n">${esc(p.updated)}</td></tr>`).join("") + `</tbody>`;
})();

// ---- The plan and the map: paper items, pull requests, notebooks and Sherlock runs, from data/paper_map.json. ----
(() => {
  const MAP = JSON.parse(document.getElementById("map-data").textContent);
  const REV = JSON.parse(document.getElementById("review-data").textContent);
  const AN = "seanome/2024-kmerseek-analysis", ENG = "seanome/kmerseek";
  const prKey = (repo, n) => `pr:${repo === ENG ? "k" : "a"}${n}`;
  const prName = (repo, n) => repo === ENG ? `kmerseek #${n}` : `#${n}`;
  const prUrl = (repo, n) => `https://github.com/${repo}/pull/${n}`;
  const nbOfRow = id => { const m = String(id).match(/^(\d+[a-z]?)/); return m ? m[1] : null; };
  const short = (s, n) => s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;
  const cleanTitle = t => String(t || "").replace(/^#+\s*/, "").replace(/\*+/g, "");

  // Nodes.
  const nodes = new Map();          // key -> {key, kind, label, sub, ...}
  const links = new Map();          // key -> Set of keys (both directions)
  const link = (a, b) => { if (!a || !b || a === b) return; (links.get(a) || links.set(a, new Set()).get(a)).add(b); (links.get(b) || links.set(b, new Set()).get(b)).add(a); };
  const add = n => { if (!nodes.has(n.key)) nodes.set(n.key, n); return nodes.get(n.key); };

  // Pull requests: open ones from GitHub, plus the merged or closed ones the map names.
  const titleNbs = t => [...String(t).matchAll(/\b(?:[Nn]otebook|[Nn]b)\s+(\d{2,3}[a-z]?)\b/g), ...String(t).matchAll(/^(\d{2,3}[a-z]?):/g)].map(m => m[1]);
  // A pull request's notebooks are the .ipynb files it adds; two different notebooks share a number,
  // so paper_map.json's notebook_aliases names one of them. Only a pull request with no notebook file
  // of its own takes the number in its title ("Notebook 284: ...").
  const ALIAS = MAP.notebook_aliases || {};
  const fileNb = f => ALIAS[f] || ((f.split("/").pop().match(/^(\d+[a-z]?)[_\-.]/) || [])[1]);
  const nbsOf = p => { const own = [...new Set((p.notebooks || []).map(fileNb).filter(Boolean))]; return own.length ? own : [...new Set(titleNbs(p.title))]; };
  PRS.prs.forEach(p => {
    const r = (REV.repos[p.repo] || {})[String(p.n)] || null;
    add({ key: prKey(p.repo, p.n), kind: "pr", repo: p.repo, n: p.n, open: true, p, rev: r,
      label: prName(p.repo, p.n), sub: short(p.title, 70), nbs: nbsOf(p) });
  });
  (PRS.not_open || []).forEach(p => add({ key: prKey(p.repo, p.n), kind: "pr", repo: p.repo, n: p.n, open: false, p, rev: null,
    label: prName(p.repo, p.n), sub: short(p.title, 70) + ` (${p.state} ${p.date})`, nbs: titleNbs(p.title) }));
  const prNode = n => nodes.get(prKey(AN, n));

  // Notebooks: every notebook a pull request changes, a paper item or run names, or a claim's evidence comes from.
  const nbTitle = id => { const r = BYID[id]; return r ? cleanTitle(r.title) : null; };
  const nbNode = id => add({ key: `nb:${id}`, kind: "nb", id, label: `nb ${id}`, sub: "" });
  nodes.forEach(n => { if (n.kind === "pr") n.nbs.forEach(id => { nbNode(id); link(n.key, `nb:${id}`); }); });
  DATA.forEach(r => { const id = nbOfRow(r.id); if (id && r.pr && nodes.has(prKey(AN, r.pr)) && nodes.has(`nb:${id}`)) link(prKey(AN, r.pr), `nb:${id}`); });

  // Sherlock runs.
  MAP.runs.forEach(r => {
    add({ key: `run:${r.id}`, kind: "run", r, label: r.name, sub: Object.entries(r.jobs).map(([k, v]) => `${v} ${k}`).join(", ") });
    r.prs.forEach(n => { if (prNode(n)) link(`run:${r.id}`, prKey(AN, n)); });
    r.notebooks.forEach(id => { nbNode(id); link(`run:${r.id}`, `nb:${id}`); });
  });

  // Claims (data/claims.json) and paper items.
  CLAIMS.claims.forEach(c => {
    const k = `claim:${c.id}`; add({ key: k, kind: "claim", c, label: c.name, sub: { supported: "supported", partly: "supported, with a limit", not: "not supported", untested: "not tested yet" }[c.status] || c.status });
    c.evidence.forEach(e => { const id = nbOfRow(e.row); if (id) { nbNode(id); link(k, `nb:${id}`); } const r = BYID[e.row]; if (r && r.pr && prNode(r.pr)) link(k, prKey(AN, r.pr)); });
    (c.open_prs || []).forEach(n => { if (prNode(n)) link(k, prKey(AN, n)); });
  });
  MAP.paper.forEach(it => {
    const k = `paper:${it.id}`; add({ key: k, kind: "paper", it, label: it.name, sub: it.status });
    it.prs.forEach(n => { if (prNode(n)) link(k, prKey(AN, n)); });
    it.notebooks.forEach(id => { nbNode(id); link(k, `nb:${id}`); });
    it.runs.forEach(id => link(k, `run:${id}`));
    it.claims.forEach(id => link(k, `claim:${id}`));
  });
  nodes.forEach(n => { if (n.kind === "nb") { const t = nbTitle(n.id) || (() => { const pr = [...(links.get(n.key) || [])].map(k => nodes.get(k)).find(x => x.kind === "pr"); return pr ? pr.p.title : ""; })(); n.sub = short(t, 60); } });

  // What lights up when something is picked: its links, plus one more step out for the kinds that need it
  // (a figure's notebooks come through its pull requests; a pull request's claims come through its paper items).
  const of = (k, kind) => [...(links.get(k) || [])].filter(x => nodes.get(x) && nodes.get(x).kind === kind);
  function linked(key) {
    const out = new Set(links.get(key) || []);
    const n = nodes.get(key);
    const more = ks => ks.forEach(x => out.add(x));
    if (!n) return out;
    if (n.kind === "paper" || n.kind === "claim") of(key, "pr").forEach(p => { more(of(p, "nb")); more(of(p, "run")); });
    if (n.kind === "claim") of(key, "paper").forEach(p => more(of(p, "pr")));
    if (n.kind === "pr" || n.kind === "nb" || n.kind === "run") [...out].filter(x => nodes.get(x).kind === "paper").forEach(p => more(of(p, "claim")));
    if (n.kind === "run") of(key, "pr").forEach(p => more(of(p, "paper")));
    return out;
  }

  // Steps: each open pull request in the first step whose rule it meets.
  const live = new Set(MAP.runs.flatMap(r => r.prs.map(n => prKey(AN, n))));
  const groupsOf = k => of(k, "paper").map(p => nodes.get(p).it.group);
  function stepOf(n) {
    if (n.rev && n.rev.review === "look again") return "look-again";
    if (live.has(n.key)) return "runs";
    const g = groupsOf(n.key);
    if (g.includes("figures")) return "figures";
    if (g.includes("examples")) return "examples";
    if (g.length) return "evidence";
    if (n.repo === ENG) return "engine";
    return "unplaced";
  }
  const openPrs = [...nodes.values()].filter(n => n.kind === "pr" && n.open);
  openPrs.forEach(n => { n.step = stepOf(n); });
  const STEPS = MAP.steps.map((s, i) => ({ ...s, i, prs: openPrs.filter(n => n.step === s.id).sort((a, b) => a.repo.localeCompare(b.repo) || a.n - b.n) }));
  const stepName = id => { const s = STEPS.find(x => x.id === id); return s ? `${s.i + 1}. ${s.name}` : id; };

  const st = { done: {}, viewing: null, pick: null, showAll: { prs: false, nbs: false } };
  const firstOpen = () => STEPS.find(s => !st.done[s.id] && s.prs.length) || null;

  function whyLine(n) {
    if (!n.rev) return "";
    const r = n.rev; const bits = [r.review + (r.last_review ? ` (${r.last_review})` : "")].concat(r.blockers || []);
    return bits.join("; ");
  }
  const nodeBtn = n => `<button type="button" class="node" data-key="${esc(n.key)}">${esc(n.label)}${n.sub ? `<span class="sub">${esc(n.sub)}</span>` : ""}</button>`;

  // Step bar.
  function renderBar() {
    const cur = firstOpen(), view = st.viewing || (cur && cur.id);
    document.getElementById("chevrons").innerHTML = STEPS.map(s => {
      const cls = st.done[s.id] ? "done" : (cur && s.id === cur.id ? "current" : "");
      return `<button type="button" class="chev ${cls}${s.id === view ? " viewing" : ""}" data-step="${esc(s.id)}" aria-current="${s.id === view ? "step" : "false"}">` +
        `<span class="num">${st.done[s.id] ? "✓" : s.i + 1}</span><span class="nm">${esc(s.name)}</span><span class="cnt">${s.prs.length} PRs</span></button>`;
    }).join("");
  }
  // "Do this first".
  let db = null;
  function renderFirst() {
    const cur = firstOpen(), box = document.getElementById("firstcard");
    if (!cur) { box.innerHTML = `<p class="when">Every step is marked done.</p>`; return; }
    const total = openPrs.length;
    const nowOpen = STEPS.filter(s => !st.done[s.id]).reduce((a, s) => a + s.prs.length, 0);
    const after = nowOpen - cur.prs.length;
    box.innerHTML = `<p class="meta">Step ${cur.i + 1} of ${STEPS.length}: ${esc(cur.name)}</p><p class="when">${esc(cur.when)}</p>` +
      `<p>${cur.prs.length} pull requests: ${cur.prs.map(n => `<a href="${prUrl(n.repo, n.n)}" target="_blank" rel="noopener">${esc(n.label)}</a>`).join(", ")}.</p>` +
      `<div class="progress"><span>${nowOpen} open now</span><span class="track" aria-hidden="true"><span class="fill" style="display:block;width:${(100 * (total - after) / total).toFixed(1)}%"></span></span><span>${after} open after this step</span></div>` +
      `<div class="btnrow"><button type="button" class="btn primary" data-showstep="${esc(cur.id)}">Show them in the map</button>` +
      (db ? `<button type="button" class="btn" data-markdone="${esc(cur.id)}">✓ Mark done</button>` : "") + `</div>`;
  }
  // Tiles.
  const ICONS = {
    figures: '<svg viewBox="0 0 44 32" aria-hidden="true"><rect x="2" y="2" width="40" height="28" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M7 25 L16 15 L23 20 L31 9 L38 16" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    examples: '<svg viewBox="0 0 44 32" aria-hidden="true"><line x1="3" y1="9" x2="41" y2="9" stroke="currentColor" stroke-width="1.5"/><rect x="14" y="5" width="12" height="8" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="3" y1="23" x2="41" y2="23" stroke="currentColor" stroke-width="1.5"/><rect x="18" y="19" width="12" height="8" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    alphabets: '<svg viewBox="0 0 44 32" aria-hidden="true"><text x="3" y="22" font-size="17" font-weight="700" fill="currentColor" font-family="monospace">HPPH</text></svg>',
    cases: '<svg viewBox="0 0 44 32" aria-hidden="true"><rect x="3" y="4" width="17" height="11" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="24" y="4" width="17" height="11" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="3" y="18" width="17" height="11" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="24" y="18" width="17" height="11" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    benchmarks: '<svg viewBox="0 0 44 32" aria-hidden="true"><rect x="5" y="16" width="7" height="13" fill="currentColor"/><rect x="18" y="8" width="7" height="21" fill="currentColor"/><rect x="31" y="12" width="7" height="17" fill="currentColor"/></svg>',
    side: '<svg viewBox="0 0 44 32" aria-hidden="true"><path d="M22 4 L22 14 M22 14 L10 26 M22 14 L34 26" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="22" cy="4" r="2.5" fill="currentColor"/><circle cx="10" cy="27" r="2.5" fill="currentColor"/><circle cx="34" cy="27" r="2.5" fill="currentColor"/></svg>',
    runs: '<svg viewBox="0 0 44 32" aria-hidden="true"><rect x="6" y="3" width="32" height="7" rx="1" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="6" y="13" width="32" height="7" rx="1" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="6" y="23" width="32" height="7" rx="1" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="11" cy="6.5" r="1.5" fill="currentColor"/><circle cx="11" cy="16.5" r="1.5" fill="currentColor"/><circle cx="11" cy="26.5" r="1.5" fill="currentColor"/></svg>',
    unplaced: '<svg viewBox="0 0 44 32" aria-hidden="true"><circle cx="22" cy="16" r="12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 3"/><text x="22" y="21" font-size="14" text-anchor="middle" fill="currentColor">?</text></svg>',
  };
  function tileData() {
    const t = MAP.groups.map(g => {
      const items = MAP.paper.filter(p => p.group === g.id);
      const prs = [...new Set(items.flatMap(p => of(`paper:${p.id}`, "pr")))].map(k => nodes.get(k)).filter(n => n.open);
      return { key: `group:${g.id}`, icon: g.id, name: g.name, count: `${items.length} ${items.length === 1 ? "item" : "items"}, ${prs.length} open ${prs.length === 1 ? "PR" : "PRs"}`, prs };
    });
    t.push({ key: "group:runs", icon: "runs", name: "Sherlock runs", count: `${MAP.runs.length} runs, ${MAP.runs.reduce((a, r) => a + Object.values(r.jobs).reduce((x, y) => x + y, 0), 0)} jobs`, prs: [...live].map(k => nodes.get(k)).filter(Boolean) });
    const un = STEPS.find(s => s.id === "unplaced");
    t.push({ key: "step:unplaced", icon: "unplaced", name: "Not in the story yet", count: `${un ? un.prs.length : 0} open ${un && un.prs.length === 1 ? "PR" : "PRs"}`, prs: un ? un.prs : [] });
    return t;
  }
  function renderTiles() {
    document.getElementById("tiles").innerHTML = tileData().map(t => {
      const rev = t.prs.filter(n => n.rev && n.rev.review !== "not reviewed").length;
      const share = t.prs.length ? rev / t.prs.length : 0;
      return `<button type="button" class="tile" data-key="${esc(t.key)}" aria-pressed="${st.pick === t.key}">${ICONS[t.icon] || ""}<span class="tn">${esc(t.name)}</span><span class="tc">${esc(t.count)}</span>` +
        `<span class="thin" title="${rev} of ${t.prs.length} reviewed"><span style="width:${(100 * share).toFixed(0)}%"></span></span><span class="tc">${rev} of ${t.prs.length} reviewed</span></button>`;
    }).join("");
  }
  // The whole plan.
  function renderPlan() {
    const cur = firstOpen(), view = st.viewing || (cur && cur.id);
    let left = STEPS.filter(s => !st.done[s.id]).reduce((a, s) => a + s.prs.length, 0);
    document.getElementById("plan").innerHTML = STEPS.map(s => {
      if (!st.done[s.id]) left -= s.prs.length;
      const open = s.id === view;
      return `<li class="${open ? "open" : ""}"><button type="button" data-step="${esc(s.id)}" aria-expanded="${open}"><span class="pn">${st.done[s.id] ? "✓" : s.i + 1}. ${esc(s.name)}</span><span>${s.prs.length} PRs</span><span class="after">${st.done[s.id] ? "marked done" : `${left} open after`}</span></button>` +
        (open ? `<div class="body"><p>${esc(s.when)}</p><p class="meta">Rule: ${esc(s.rule)}.${st.done[s.id] && db ? ` <button type="button" class="clear" data-undone="${esc(s.id)}">mark not done</button>` : ""}</p>` + (s.prs.length ? s.prs.map(n =>
          `<div class="prline"><button type="button" class="rowlink" data-key="${esc(n.key)}">${esc(n.label)}</button><span>${esc(short(n.p.title, 90))}</span><span class="why">${esc(whyLine(n))}</span></div>`).join("") : `<p class="muted">None.</p>`) + `</div>` : "") + `</li>`;
    }).join("");
  }

  // The map.
  function pickSet() {
    if (!st.pick) return null;
    if (st.pick.startsWith("step:")) { const s = STEPS.find(x => x.id === st.pick.slice(5)); const ks = new Set(); (s ? s.prs : []).forEach(n => { ks.add(n.key); linked(n.key).forEach(x => ks.add(x)); }); return ks; }
    if (st.pick.startsWith("group:")) {
      const g = st.pick.slice(6), ks = new Set();
      const seeds = g === "runs" ? MAP.runs.map(r => `run:${r.id}`) : MAP.paper.filter(p => p.group === g).map(p => `paper:${p.id}`);
      seeds.forEach(k => { ks.add(k); linked(k).forEach(x => ks.add(x)); });
      return ks;
    }
    const ks = linked(st.pick); ks.add(st.pick); return ks;
  }
  function renderCols() {
    const lit = pickSet(), cols = document.getElementById("cols");
    cols.classList.toggle("only", document.getElementById("onlylinked").checked && !!lit);
    const cls = k => !lit ? "" : (k === st.pick ? " sel" : lit.has(k) ? " lit" : " dim");
    const btn = n => nodeBtn(n).replace('class="node"', `class="node${cls(n.key)}"`);
    // Paper column, by group, then claims.
    const groups = MAP.groups.map(g => { const ns = MAP.paper.filter(p => p.group === g.id).map(p => nodes.get(`paper:${p.id}`)); const any = !lit || ns.some(n => lit.has(n.key)); return `<h4 class="${any ? "" : "dim"}">${esc(g.name)}</h4><div class="list">${ns.map(btn).join("")}</div>`; }).join("");
    const claims = CLAIMS.claims.map(c => nodes.get(`claim:${c.id}`)); const anyC = !lit || claims.some(n => lit.has(n.key));
    document.getElementById("col-paper").innerHTML = groups + `<h4 class="${anyC ? "" : "dim"}">Claims</h4><div class="list">${claims.map(btn).join("")}</div>`;
    // Pull requests: folded until something is picked.
    const prs = [...nodes.values()].filter(n => n.kind === "pr").sort((a, b) => (a.repo === ENG) - (b.repo === ENG) || b.n - a.n);
    const showPr = lit ? prs.filter(n => lit.has(n.key)) : prs;
    const nOpen = prs.filter(n => n.open).length;
    document.getElementById("k-prs").textContent = (lit ? `${showPr.length} linked of ` : "") + `${nOpen} open and ${prs.length - nOpen} merged`;
    document.getElementById("col-prs").innerHTML = (!lit && !st.showAll.prs)
      ? `<p class="fold">Pick something to see its pull requests, or <button type="button" data-showall="prs">show all ${prs.length}</button>.</p>`
      : `<div class="list">${(lit && document.getElementById("onlylinked").checked ? showPr : prs).map(n => btn({ ...n, sub: n.sub + (n.rev ? ` · ${n.rev.review}` : "") })).join("")}</div>`;
    // Notebooks.
    const nbs = [...nodes.values()].filter(n => n.kind === "nb").sort((a, b) => parseInt(a.id) - parseInt(b.id) || a.id.localeCompare(b.id));
    const showNb = lit ? nbs.filter(n => lit.has(n.key)) : nbs;
    document.getElementById("k-nbs").textContent = lit ? `${showNb.length} linked of ${nbs.length}` : `${nbs.length}`;
    document.getElementById("col-nbs").innerHTML = (!lit && !st.showAll.nbs)
      ? `<p class="fold">Pick something to see its notebooks, or <button type="button" data-showall="nbs">show all ${nbs.length}</button>.</p>`
      : `<div class="list">${(lit && document.getElementById("onlylinked").checked ? showNb : nbs).map(btn).join("")}</div>`;
    // Runs.
    const runs = MAP.runs.map(r => nodes.get(`run:${r.id}`));
    const nLit = lit ? runs.filter(n => lit.has(n.key)).length : runs.length;
    document.getElementById("k-runs").textContent = `The queue on ${MAP.runs_read}.` + (lit ? ` ${nLit} linked of ${runs.length}.` : "");
    document.getElementById("col-runs").innerHTML = `<div class="list">${runs.map(n => btn(n).replace("</button>", ` <span class="badge running"><span aria-hidden="true">◐</span>in the queue</span></button>`)).join("")}</div>`;
    const note = document.getElementById("mapnote");
    note.textContent = lit ? `${lit.size} items linked.` : "Nothing picked.";
  }

  const figLink = (branch, path) => `<a href="https://github.com/${AN}/blob/${encodeURI(branch)}/${encodeURI(path)}" target="_blank" rel="noopener">${esc(path.split("/").pop())}</a>`;
  function figList(items, max = 12) { return items.length ? `<span class="figs">${items.slice(0, max).join(" ")}${items.length > max ? ` <span class="muted">and ${items.length - max} more</span>` : ""}</span>` : `<span class="muted">none</span>`; }
  const btnsOf = (key, kind) => { const ks = [...linked(key)].filter(x => nodes.get(x).kind === kind).map(x => nodes.get(x)); return ks.length ? ks.map(n => `<button type="button" class="rowlink" data-key="${esc(n.key)}">${esc(n.label)}</button>`).join(" ") : `<span class="muted">none</span>`; };
  const runLine = n => `${esc(n.r.name)}: ${esc(Object.entries(n.r.jobs).map(([k, v]) => `${v} ${k}`).join(", "))}${n.r.waiting_why ? ` (waiting on ${esc(n.r.waiting_why)})` : ""}. ${esc(n.r.progress)}.`;
  function figuresOfPaper(it) {
    // Link each named figure file to the pull request branch that has it.
    return it.figures.map(f => { for (const n of it.prs) { const p = prNode(n); const hit = p && p.open && (p.p.figures || []).find(x => x.endsWith("/" + f) || x === f); if (hit) return figLink(p.p.branch, hit); } return `<code>${esc(f)}</code>`; });
  }
  function renderDetail() {
    const box = document.getElementById("mapdetail"), k = st.pick;
    if (!k) { box.innerHTML = `<p>Nothing picked yet. Start with <button type="button" class="rowlink" data-key="paper:fig1">Figure 1</button> or a tile above.</p>`; return; }
    if (k.startsWith("step:") || k.startsWith("group:")) {
      const t = tileData().find(x => x.key === k); const s = STEPS.find(x => `step:${x.id}` === k);
      box.innerHTML = `<h3>${esc(s ? `Step ${stepName(s.id)}` : t ? t.name : k)}</h3>${s ? `<p>${esc(s.when)}</p>` : ""}<p>Everything linked is lit below.</p>`; return;
    }
    const n = nodes.get(k); let h = "";
    if (n.kind === "pr") {
      const p = n.p, r = n.rev;
      const nbFigs = (p.figures || []).map(f => figLink(p.branch, f));
      const below = p.base_pr ? ` Built on <button type="button" class="rowlink" data-key="${esc(prKey(n.repo, p.base_pr))}">${esc(prName(n.repo, p.base_pr))}</button>.` : "";
      h = `<h3><a href="${prUrl(n.repo, n.n)}" target="_blank" rel="noopener">${esc(n.label)}</a> ${esc(p.title)}</h3>` +
        `<p>${n.open ? (p.draft ? "Open, draft." : "Open.") : `${esc(p.state)} ${esc(p.date)}.`}${below}${n.open ? ` In step ${esc(stepName(n.step))}.` : ""}</p><dl>` +
        (r ? `<dt>Your review</dt><dd>${esc(r.review)}${r.last_review ? ` (last ${esc(r.last_review)})` : ""}</dd><dt>What stops it merging</dt><dd>${r.blockers.length ? esc(r.blockers.join("; ")) : "nothing found"}</dd>` +
          `<dt>${esc(r.skill || "review skill")}</dt><dd>${r.skill_review.mark === "none" ? "no review comment" : `${esc(r.skill_review.mark)} (${esc(r.skill_review.date)})`}</dd>` : "") +
        `<dt>Paper</dt><dd>${btnsOf(k, "paper")}</dd><dt>Claims</dt><dd>${btnsOf(k, "claim")}</dd><dt>Notebooks</dt><dd>${btnsOf(k, "nb")}</dd>` +
        `<dt>Figures it adds</dt><dd>${figList(nbFigs)}</dd>` +
        `<dt>Sherlock</dt><dd>${of(k, "run").map(x => runLine(nodes.get(x))).join("<br>") || "no run in the queue"}</dd></dl>`;
    } else if (n.kind === "paper") {
      const it = n.it;
      h = `<h3>${esc(it.name)}</h3><p>${esc(it.what)}</p>${it.number ? `<p><strong>Number:</strong> ${esc(it.number)}</p>` : ""}<p class="meta">Status: ${esc(it.status)}.</p><dl>` +
        `<dt>Pull requests</dt><dd>${it.prs.map(x => { const p = prNode(x); return p ? `<button type="button" class="rowlink" data-key="${esc(p.key)}">${esc(p.label)}</button>${p.rev ? ` <span class="muted">${esc(p.rev.review)}</span>` : p.open ? "" : ` <span class="muted">${esc(p.p.state)}</span>`}` : `#${x}`; }).join(", ")}</dd>` +
        `<dt>Notebooks</dt><dd>${btnsOf(k, "nb")}</dd><dt>Figures</dt><dd>${figList(figuresOfPaper(it))}</dd>` +
        `<dt>Sherlock</dt><dd>${[...linked(k)].filter(x => nodes.get(x).kind === "run").map(x => runLine(nodes.get(x))).join("<br>") || "no run in the queue"}</dd>` +
        `<dt>Claims</dt><dd>${btnsOf(k, "claim")}</dd></dl>`;
    } else if (n.kind === "nb") {
      const r = BYID[n.id];
      const prs = of(k, "pr").map(x => nodes.get(x)).filter(p => p.open);
      const stem = n.id.split("-")[0] + "_";
      const figs = prs.flatMap(p => (p.p.figures || []).filter(f => f.split("/").pop().startsWith(stem)).map(f => figLink(p.p.branch, f)));
      const files = prs.flatMap(p => (p.p.notebooks || []).filter(f => fileNb(f) === n.id).map(f => figLink(p.p.branch, f)));
      if (!files.length && r && r.source && r.source.path) files.push(figLink(r.source.branch || "main", r.source.path));
      h = `<h3>Notebook ${esc(n.id)}</h3><p>${esc(n.sub)}</p><p class="figs">${files.join(" ")}</p>` + (r ? `<p>${badge(r.verdict)} ${esc((r.result || []).slice(0, 2).map(x => `${lab(x)}: ${x.value}`).join("; "))} ${rowButton(String(r.id))}</p>` : `<p class="muted">Not in the experiment table yet.</p>`) +
        `<dl><dt>Pull requests</dt><dd>${btnsOf(k, "pr")}</dd><dt>Paper</dt><dd>${btnsOf(k, "paper")}</dd><dt>Claims</dt><dd>${btnsOf(k, "claim")}</dd>` +
        `<dt>Figures</dt><dd>${figList(figs)}</dd><dt>Sherlock</dt><dd>${of(k, "run").map(x => runLine(nodes.get(x))).join("<br>") || "no run in the queue"}</dd></dl>`;
    } else if (n.kind === "run") {
      const r = n.r;
      h = `<h3>${esc(r.name)} <span class="badge running"><span aria-hidden="true">◐</span>in the queue</span></h3><p>${esc(Object.entries(r.jobs).map(([a, b]) => `${b} ${a}`).join(", "))} jobs${r.waiting_why ? `; waiting on ${esc(r.waiting_why)}` : ""}. ${esc(r.progress)}. Head job ${esc(r.head_job)}. Read ${esc(MAP.runs_read)}.</p><p>${esc(r.note)}</p><dl>` +
        `<dt>Feeds</dt><dd>${r.feeds.length ? r.feeds.map(f => `<code>${esc(f)}</code>`).join(" ") + (r.feeds_inferred ? " <em>(inferred, not yet read by the notebook)</em>" : "") : "no figure yet"}</dd>` +
        `<dt>Pull requests</dt><dd>${btnsOf(k, "pr")}</dd><dt>Notebooks</dt><dd>${btnsOf(k, "nb")}</dd><dt>Paper</dt><dd>${btnsOf(k, "paper")}</dd></dl>`;
    } else if (n.kind === "claim") {
      const c = n.c;
      h = `<h3>${esc(c.name)}</h3><p>${esc(c.statement)}</p><p class="meta">${esc(n.sub)}. <a href="#claim-${esc(c.id)}">Its numbers</a>.</p><dl>` +
        `<dt>Paper</dt><dd>${btnsOf(k, "paper")}</dd><dt>Pull requests</dt><dd>${btnsOf(k, "pr")}</dd><dt>Notebooks</dt><dd>${btnsOf(k, "nb")}</dd><dt>Sherlock</dt><dd>${btnsOf(k, "run")}</dd></dl>`;
    }
    box.innerHTML = h;
  }
  function renderAll() { renderBar(); renderFirst(); renderTiles(); renderPlan(); renderCols(); renderDetail(); }
  function pick(k, scroll) {
    st.pick = st.pick === k ? null : k; renderTiles(); renderCols(); renderDetail();
    if (scroll) document.getElementById("map").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-key],[data-step],[data-showstep],[data-markdone],[data-undone],[data-showall]");
    if (!t) return;
    if (t.dataset.key) { e.preventDefault(); const inMap = !!t.closest("#map"); pick(t.dataset.key, !inMap); }
    else if (t.dataset.step) { st.viewing = t.dataset.step; renderBar(); renderPlan(); if (t.closest("#stepbar")) document.getElementById("plan").scrollIntoView({ behavior: "smooth", block: "center" }); }
    else if (t.dataset.showstep) { st.pick = null; pick(`step:${t.dataset.showstep}`, true); }
    else if (t.dataset.showall) { st.showAll[t.dataset.showall] = true; renderCols(); }
    else if ((t.dataset.markdone || t.dataset.undone) && db) {
      const done = { ...st.done }; if (t.dataset.markdone) done[t.dataset.markdone] = true; else delete done[t.dataset.undone];
      db.doc("plan/steps").set({ done }).catch(() => {});
      st.done = done; st.viewing = null; renderAll();
    }
  });
  document.getElementById("onlylinked").addEventListener("change", renderCols);
  document.getElementById("mapclear").addEventListener("click", () => { st.pick = null; renderTiles(); renderCols(); renderDetail(); });
  renderAll();

  // "Mark done" is kept in the page's shared store (the db capability), so it follows Olga to her phone.
  // Outside the claude.ai viewer (the GitHub Pages copy) there is no store and the button stays hidden.
  (async () => {
    const d = window.claude && window.claude.use ? await window.claude.use("db") : null;
    if (!d) return;
    db = d;
    db.doc("plan/steps").onSnapshot(snap => { st.done = (snap.exists && snap.data().done) || {}; renderAll(); }, () => { db = null; renderFirst(); });
    renderFirst();
  })();
})();

render();

// Comments: in the claude.ai viewer, each claim gets a button that opens the shell's own comment box
// anchored to that claim (declared composer-only: no consent, nothing stored by the page). Elsewhere,
// such as the GitHub Pages copy, window.claude is absent and the buttons stay hidden.
(async () => {
  const c = window.claude && window.claude.use ? await window.claude.use("comments") : null;
  if (!c) return;
  const buttons = [...document.querySelectorAll("[data-comment-btn]")];
  const hideAll = () => buttons.forEach(b => { b.hidden = true; });
  buttons.forEach(b => {
    b.hidden = false;
    b.addEventListener("click", async e => {
      e.stopPropagation();
      try { await c.openComposer({ element: b.closest("[data-comment-target]") }); }
      catch (err) {
        const code = err && err.code;
        if (["unavailable", "not_granted", "forbidden", "capability_disabled", "capability_removed"].includes(code)) hideAll();
      }
    });
  });
})();
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
<script type="application/json" id="map-data">
__MAP__
</script>
<script type="application/json" id="review-data">
__REVIEW__
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
    T((xq(q1) + 10), (yq + yt) / 2 + 4, f"same {L} positions, no gaps", "start", 11)
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


def check_map(paper_map, claims, prs, review):
    """Stop if paper_map.json names a claim, run, group, step or pull request the other files do not have."""
    claim_ids = {c["id"] for c in claims["claims"]}
    run_ids = {r["id"] for r in paper_map["runs"]}
    group_ids = {g["id"] for g in paper_map["groups"]}
    known = {(p["repo"], p["n"]) for p in prs["prs"] + prs.get("not_open", [])}
    an = "seanome/2024-kmerseek-analysis"
    bad = []
    for it in paper_map["paper"]:
        bad += [f"{it['id']}: no claim {c}" for c in it["claims"] if c not in claim_ids]
        bad += [f"{it['id']}: no run {r}" for r in it["runs"] if r not in run_ids]
        bad += [f"{it['id']}: no group {it['group']}"] if it["group"] not in group_ids else []
        bad += [f"{it['id']}: PR {n} is neither open nor in not_open (run fetch_open_prs.py)" for n in it["prs"] if (an, n) not in known]
    for r in paper_map["runs"]:
        bad += [f"run {r['id']}: PR {n} unknown" for n in r["prs"] if (an, n) not in known]
    if {s["id"] for s in paper_map["steps"]} != {"look-again", "runs", "figures", "examples", "evidence", "engine", "unplaced"}:
        bad.append("steps must be exactly the seven the page assigns pull requests to")
    if bad:
        sys.exit("paper_map.json does not match the other data files:\n" + "\n".join(bad))


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
    paper_map = json.loads(PAPER_MAP.read_text())
    review = json.loads(REVIEW.read_text())
    check_claims(records, claims)
    check_map(paper_map, claims, prs, review)
    checked = mark_changed_since_read(records)
    n_changed = sum(1 for r in records if r.get("changed_since_read"))
    d = page_dates()

    meta = [f"First published {d['first']}." if d["first"] else "",
            f"Experiment numbers last changed {d['numbers']}." if d["numbers"] else "",
            f"Page built {d['built']}.",
            f"Open pull requests as of {prs['fetched']}.",
            f"Sherlock queue read {paper_map['runs_read']}."]
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
            .replace("__REVIEWREAD__", review["read"]).replace("__PRDATE__", prs["fetched"]).replace("__RUNSDATE__", paper_map["runs_read"])
            .replace("__MAP__", embed(paper_map)).replace("__REVIEW__", embed(review))
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
        # The shelf supplies its own html/head/body, so carry over everything in our <head>
        # (title, font links, both style blocks) except the charset and viewport tags.
        head = html[html.index("<head>") + len("<head>"): html.index("</head>")]
        head = "\n".join(line for line in head.splitlines() if not line.lstrip().startswith("<meta"))
        body = html[html.index("<body>") + len("<body>"): html.index("</body>")]
        for needle in ("<title>", 'id="seanome-fonts"', "--glass:", "fonts.googleapis.com"):
            assert needle in head, f"artifact copy would lose {needle}"
        args.artifact.write_text(f"{head.strip()}\n{body}")
        print(f"wrote artifact copy {args.artifact}")


if __name__ == "__main__":
    main()
