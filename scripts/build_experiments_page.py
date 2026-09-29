"""Write notebooks/experiments.html from data/experiments.json.

The page is one file with the records embedded at the bottom, so it opens
offline. Run from the repo root:

    python scripts/build_experiments_page.py [artifact_copy.html]
"""

import datetime
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "experiments.json"
PAGE = REPO / "notebooks" / "experiments.html"

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kmerseek experiments</title>
<style>
:root {
  --bg: #ffffff; --ink: #1b1f24; --line: #d5d9de; --row-hover: #f4f6f8;
  --alive: #1e7b34; --dead: #7a7f86; --bug: #c85a00; --running: #1f5fbf; --notrun: #1b1f24;
  --chip-on-bg: #1b1f24; --chip-on-ink: #ffffff;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15181c; --ink: #e8eaed; --line: #3a4048; --row-hover: #1f242a;
    --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #e8eaed;
    --chip-on-bg: #e8eaed; --chip-on-ink: #15181c; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #15181c; --ink: #e8eaed; --line: #3a4048; --row-hover: #1f242a;
  --alive: #3fa55a; --dead: #8d939a; --bug: #e07a1f; --running: #4b8ae6; --notrun: #e8eaed;
  --chip-on-bg: #e8eaed; --chip-on-ink: #15181c; color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { margin: 0; overflow-x: hidden; }
body { background: var(--bg); color: var(--ink); font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.4rem; margin: 0 0 4px; }
.lede { margin: 0 0 16px; color: var(--ink); font-style: italic; }
h2 { font-size: 0.95rem; margin: 16px 0 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 0 0 8px; }
.legend span.what { color: var(--ink); font-style: italic; font-size: 0.85rem; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px; border-radius: 4px;
  font-size: 0.8rem; font-weight: 600; white-space: nowrap; border: 1.5px solid transparent; color: #fff; }
.badge.alive { background: var(--alive); }
.badge.dead { background: var(--dead); }
.badge.bug { background: var(--bug); }
.badge.running { background: var(--running); }
.badge.not-run { background: transparent; color: var(--notrun); border-color: var(--notrun); }
.controls { display: grid; gap: 8px; margin: 12px 0; }
input[type=search] { width: 100%; max-width: 480px; padding: 8px 10px; font: inherit; color: var(--ink);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chips .label { font-size: 0.85rem; color: var(--ink); font-style: italic; margin-right: 4px; }
.chip { font: inherit; font-size: 0.82rem; padding: 3px 10px; border-radius: 999px; cursor: pointer;
  background: transparent; color: var(--ink); border: 1px dashed var(--line); }
.chip[aria-pressed="true"] { background: var(--chip-on-bg); color: var(--chip-on-ink); border-style: solid; border-color: var(--chip-on-bg); }
.chip[aria-pressed="true"]::before { content: "\2713\00a0"; }
.clear { font: inherit; font-size: 0.82rem; background: none; border: none; color: var(--ink); text-decoration: underline; cursor: pointer; }
.count { font-size: 0.85rem; color: var(--ink); font-style: italic; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; position: relative; z-index: 0; }
table { border-collapse: collapse; width: 100%; min-width: 980px; font-size: 0.86rem; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { position: sticky; top: 0; background: var(--bg); z-index: 1; }
th button { font: inherit; font-weight: 600; color: var(--ink); background: none; border: none; padding: 0; cursor: pointer; text-align: left; }
th button .arrow { display: inline-block; width: 1em; }
tr.row { cursor: pointer; }
tr.row:hover, tr.row:focus { background: var(--row-hover); outline: none; }
tr.row.open { background: var(--row-hover); }
td.id { font-family: var(--mono); white-space: nowrap; }
td.title { min-width: 180px; }
td.question { min-width: 220px; }
td.result { min-width: 240px; max-width: 340px; overflow-wrap: anywhere; }
td.date { white-space: nowrap; font-family: var(--mono); }
.rlist { margin: 0; padding-left: 16px; }
.rlist li { margin: 0 0 2px; }
.val { font-family: var(--mono); }
tr.detail td { background: var(--row-hover); padding: 12px 14px 16px; }
.detail h3 { font-size: 0.85rem; margin: 10px 0 4px; }
.detail h3:first-child { margin-top: 0; }
.detail table.inner { min-width: 0; width: auto; font-size: 0.82rem; }
.detail table.inner td, .detail table.inner th { border-bottom: 1px solid var(--line); padding: 3px 8px; position: static; background: none; }
pre { font-family: var(--mono); font-size: 0.8rem; line-height: 1.35; margin: 4px 0 8px; padding: 8px; overflow-x: auto;
  border: 1px solid var(--line); border-radius: 4px; background: var(--bg); white-space: pre; }
a { color: var(--ink); }
.muted { color: var(--ink); font-style: italic; }
footer { margin: 16px 0 24px; font-size: 0.85rem; color: var(--ink); font-style: italic; }
@media (max-width: 700px) { main { padding: 16px; } h1 { font-size: 1.2rem; } }
</style>
</head>
<body>
<main>
<h1>kmerseek experiments</h1>
<p class="lede">One row per notebook or pipeline run. Every number is copied from a notebook output or a results file. Click a row to see where each number comes from.</p>

<h2>What the verdict colours mean</h2>
<div class="legend" id="legend"></div>

<div class="controls">
  <input type="search" id="q" placeholder="Search every column" aria-label="Search every column">
  <div class="chips" id="vchips"><span class="label">Show verdict:</span></div>
  <div class="chips" id="cchips"><span class="label">Show paper claim:</span></div>
  <div><button class="clear" id="clear" type="button">Clear search and filters</button> <span class="count" id="count"></span></div>
</div>

<div class="tablewrap">
<table>
<thead><tr id="head"></tr></thead>
<tbody id="body"></tbody>
</table>
</div>

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
const COLS = [
  ["id", "id"], ["title", "title"], ["question", "what it tested"], ["dataset", "queries and targets"],
  ["comparison_tools", "compared with"], ["result", "what happened"], ["verdict", "verdict"],
  ["claim", "which paper claim"], ["date", "last changed"],
];
const DATA = JSON.parse(document.getElementById("experiments-data").textContent);

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const vinfo = v => VERDICTS.find(x => x[0] === v) || [v, v, "?", ""];
const badge = v => { const [k, label, mark] = vinfo(v); return `<span class="badge ${esc(k)}"><span aria-hidden="true">${mark}</span>${esc(label)}</span>`; };
const tools = r => (r.comparison_tools || []).join(", ");
const resultText = r => (r.result || []).map(x => `${x.name}: ${x.value}`).join("; ");

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
document.querySelectorAll("th button").forEach(b => b.addEventListener("click", () => {
  const col = b.parentElement.dataset.col;
  if (state.sort === col) state.dir = -state.dir; else { state.sort = col; state.dir = 1; }
  render();
}));

document.getElementById("q").addEventListener("input", e => { state.q = e.target.value.trim().toLowerCase(); render(); });
document.getElementById("clear").addEventListener("click", () => {
  state.q = ""; document.getElementById("q").value = "";
  state.verdicts.clear(); state.claims.clear();
  document.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", "false"));
  render();
});

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

function detail(r) {
  const rows = (r.result || []).map(x =>
    `<tr><td>${esc(x.name)}</td><td class="val">${esc(x.value)}</td><td>${x.source_cell >= 0 ? "cell " + x.source_cell : "results file"}</td><td><code>${esc(x.output_line)}</code></td></tr>`).join("");
  const est = (r.estimates || []).map(x =>
    `<tr><td>${esc(x.name)}</td><td class="val">${esc(x.value)}</td><td>${x.source_cell >= 0 ? "cell " + x.source_cell : "results file"}</td><td><code>${esc(x.output_line)}</code></td></tr>`).join("");
  const res = (r.residues || []).map(p => {
    const lines = [p.query, p.match_line, p.target].join("\n") + (p.encoded ? "\n\n" + p.encoded : "");
    return `<p><strong>${esc(p.pair)}</strong>: ${esc(p.identity)}${p.source_cell >= 0 ? ", cell " + p.source_cell : ""}</p><pre>${esc(lines)}</pre>`;
  }).join("");
  const thead = `<tr><th>number</th><th>value as printed</th><th>where</th><th>printed line</th></tr>`;
  return `<h3>What happened: every number, with the output it was copied from</h3>` +
    (rows ? `<div class="tablewrap"><table class="inner">${thead}${rows}</table></div>` : `<p class="muted">No numbers: the notebook has no outputs.</p>`) +
    (est ? `<h3>Estimates (modelled, not measured)</h3><div class="tablewrap"><table class="inner">${thead}${est}</table></div>` : "") +
    (res ? `<h3>Aligned residues</h3>${res}` : "") +
    `<h3>Why this verdict</h3><p>${esc(r.verdict_basis)}</p>` +
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
  document.querySelectorAll("th").forEach(th => {
    const on = th.dataset.col === state.sort;
    th.setAttribute("aria-sort", on ? (state.dir > 0 ? "ascending" : "descending") : "none");
    th.querySelector(".arrow").textContent = on ? (state.dir > 0 ? "▲" : "▼") : "";
  });
  document.getElementById("count").textContent = `${rows.length} of ${DATA.length} shown`;
  const body = document.getElementById("body");
  body.innerHTML = rows.map(r => {
    const id = String(r.id), open = state.open.has(id);
    const res = (r.result || []).length
      ? `<ul class="rlist">${r.result.slice(0, 3).map(x => `<li>${esc(x.name)}: <span class="val">${esc(x.value)}</span></li>`).join("")}` +
        (r.result.length > 3 ? `<li class="muted">and ${r.result.length - 3} more: open the row to see them</li>` : "") + `</ul>`
      : `<span class="muted">no outputs</span>`;
    return `<tr class="row${open ? " open" : ""}" tabindex="0" aria-expanded="${open}" data-id="${esc(id)}">` +
      `<td class="id">${esc(id)}</td><td class="title">${esc(r.title)}</td><td class="question">${esc(r.question)}</td>` +
      `<td>${esc(r.dataset)}</td><td>${esc(tools(r))}</td><td class="result">${res}</td>` +
      `<td>${badge(r.verdict)}</td><td>${esc(r.claim)}</td><td class="date">${esc(r.date)}</td></tr>` +
      (open ? `<tr class="detail"><td colspan="${COLS.length}"><div class="detail">${detail(r)}</div></td></tr>` : "");
  }).join("");
  body.querySelectorAll("tr.row").forEach(tr => {
    const toggle = () => { const id = tr.dataset.id; if (state.open.has(id)) state.open.delete(id); else state.open.add(id); render(); };
    tr.addEventListener("click", e => { if (!e.target.closest("a")) toggle(); });
    tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  });
}
render();
});
</script>
<script type="application/json" id="experiments-data">
__DATA__
</script>
</body>
</html>
"""


def main():
    records = json.loads(DATA.read_text())
    date = datetime.date.fromtimestamp(os.path.getmtime(DATA)).isoformat()
    payload = json.dumps(records, ensure_ascii=False, indent=1).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATE__", date).replace("__DATA__", payload)
    PAGE.write_text(html)
    print(f"wrote {PAGE.relative_to(REPO)}: {len(records)} records, data dated {date}")
    if len(sys.argv) > 1:
        # Body-only copy for the claude.ai artifact shelf, which supplies its own html/head/body.
        title = html[html.index("<title>"): html.index("</title>") + len("</title>")]
        style = html[html.index("<style>"): html.index("</style>") + len("</style>")]
        body = html[html.index("<body>") + len("<body>"): html.index("</body>")]
        Path(sys.argv[1]).write_text(f"{title}\n{style}\n{body}")
        print(f"wrote artifact copy {sys.argv[1]}")


if __name__ == "__main__":
    main()
