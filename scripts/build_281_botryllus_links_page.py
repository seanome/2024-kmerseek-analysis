#!/usr/bin/env python3
"""Build the interactive page of every Botryllus protein linked to an MHC, immune-recognition
or self/non-self protein, for notebook 281.

    python3 scripts/build_281_botryllus_links_page.py [out.html]

Reads the notebook's inputs (/Users/olga/data/botryllus-mhc/) through
notebooks/botryllus_mhc_utils.py, so links mean what they mean in the notebook: a kmerseek
link is a (Botryllus protein, Swiss-Prot target) pair with a kept region (not half low
complexity or helix, target residues in the named target) with E < 10; a tool link is a
phmmer, jackhmmer or mmseqs2 hit on the pair with E <= 10. The page filters these down to
any lower cutoff. The control proteins are left out; the category-level numbers against the
control come from the notebook's decision table in tables/.

Each link also carries every kept kmerseek region on the pair (best E for each alphabet and
position) and every tool hit with its target coordinates, for the pair view. The tool hit
files keep only target coordinates (computeDarkSet's 6-column hits), so tool hits are drawn on
the target alone. Botryllus protein lengths come from botryllus_query_lengths.tsv, written by
scripts/fasta_lengths.py from the run's botryllus.query.fasta.

Named Botryllus genes the search box knows (BHF and others, mapped to their gene models)
come from data/botryllus_named_genes.tsv; add a row there to make a name searchable.

Writes figures/281_botryllus_immune_links_page.html by default. Run with the
2025-kmerseek-analysis interpreter.
"""

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import botryllus_mhc_utils as bmu  # noqa: E402

DATA = Path("/Users/olga/data/botryllus-mhc")
MAX_BIASED_SHARE = 0.5  # same as notebook 281
LINK_E_MAX = 10
SOURCE = "botryllus_vs_swissprot_minus_ascidiacea"
DECISION = ROOT / "tables" / f"281_kmerseek_only_targets_by_category_vs_control_{SOURCE}.csv"
FONTS = ROOT / "scripts" / "experiments_page_fonts.css"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "281_botryllus_immune_links_page.html"


def r3(x):
    return None if x is None else float(f"{x:.3g}")


def main():
    targets = bmu.load_targets(DATA)
    regions_all = bmu.load_regions(DATA, targets, MAX_BIASED_SHARE)
    regions = regions_all.filter(pl.col("kept"))
    tool_hits = bmu.load_tool_hits(DATA, targets)
    tool_pairs = tool_hits.select("query_name", "target_name").unique()
    letter_maps = bmu.alphabet_letter_maps(regions_all)

    links = bmu.with_identity(bmu.links_at(regions, tool_pairs, LINK_E_MAX), letter_maps) \
        .filter(pl.col("category") != "control")
    tools = (tool_hits.filter(pl.col("category") != "control")
             .group_by("query_name", "target_name", "tool").agg(pl.col("evalue").min())
             .pivot(on="tool", index=["query_name", "target_name"], values="evalue"))
    for t in bmu.TOOLS:
        if t not in tools.columns:
            tools = tools.with_columns(pl.lit(None, dtype=pl.Float64).alias(t))
    pairs = (links.select("query_name", "target_name")
             .join(tools.select("query_name", "target_name"), on=["query_name", "target_name"], how="full", coalesce=True))
    pairs = pairs.join(links, on=["query_name", "target_name"], how="left").join(tools, on=["query_name", "target_name"], how="left")

    alphabets = sorted(regions["alphabet"].unique())
    a_index = {a: i for i, a in enumerate(alphabets)}
    region_sets = (regions.filter((pl.col("category") != "control") & (pl.col("region_evalue") < LINK_E_MAX))
                   .group_by("query_name", "target_name", "alphabet", "region_start", "region_end", "target_start", "target_end")
                   .agg(pl.col("region_evalue").min())
                   .sort("region_evalue")
                   .group_by("query_name", "target_name", maintain_order=True)
                   .agg(pl.struct("alphabet", "region_start", "region_end", "target_start", "target_end", "region_evalue").alias("rs")))
    pair_regions = {(q, t): [[a_index[x["alphabet"]], x["region_start"] + 1, x["region_end"], x["target_start"] + 1,
                              x["target_end"], r3(x["region_evalue"])] for x in rs]
                    for q, t, rs in region_sets.iter_rows()}
    hit_sets = (tool_hits.filter(pl.col("category") != "control").sort("evalue")
                .group_by("query_name", "target_name", maintain_order=True)
                .agg(pl.struct("tool", "tstart", "tend", "evalue").alias("hs")))
    t_tool = {t: i for i, t in enumerate(bmu.TOOLS)}
    pair_hits = {(q, t): [[t_tool[x["tool"]], x["tstart"], x["tend"], r3(x["evalue"])] for x in hs]
                 for q, t, hs in hit_sets.iter_rows()}
    qlen = dict(pl.read_csv(DATA / "botryllus_query_lengths.tsv", separator="\t", has_header=False,
                            new_columns=["name", "length"]).iter_rows())

    cat_targets = targets.filter(pl.col("category") != "control").sort("accession")
    t_index = {a: i for i, a in enumerate(cat_targets["accession"])}
    # Target names come from the annotate step (Swiss-Prot RecName); fall back to the target table.
    desc = dict(zip(cat_targets["accession"], cat_targets["description"]))
    desc.update({a: d for a, d in zip(links["target_name"], links["target_description"]) if d})
    genes = sorted(pairs["query_name"].unique())
    g_index = {g: i for i, g in enumerate(genes)}

    rows = []
    for r in pairs.iter_rows(named=True):
        k = None
        if r["region_evalue"] is not None:
            k = [r3(r["region_evalue"]), r["alphabet"], r["ksize"], r["region_start"] + 1, r["region_end"],
                 r["target_start"] + 1, r["target_end"], r["region_subseq"], r["target_subseq"],
                 r["query_letters"], r["target_letters"],
                 None if r["identity"] is None else round(r["identity"] * 100, 1),
                 None if r["chance_identity"] is None else round(r["chance_identity"] * 100, 1),
                 None if r["letter_identity"] is None else round(r["letter_identity"] * 100, 1),
                 None if r["chance_letter_identity"] is None else round(r["chance_letter_identity"] * 100, 1),
                 r["n_alphabets"]]
        key = (r["query_name"], r["target_name"])
        rows.append([g_index[r["query_name"]], t_index[r["target_name"]], k] + [r3(r[t]) for t in bmu.TOOLS]
                    + [pair_regions.get(key, []), pair_hits.get(key, [])])

    decision = pl.read_csv(DECISION)
    expected = {f"{c:g}": {row["category"]: [row["n_reached"], row["n_targets"], round(row["expected_share"] * 100, 1),
                                             round(row["fold"], 2), row["p_bonferroni"]]
                           for row in decision.filter(pl.col("e_cutoff") == c).iter_rows(named=True)}
                for c in decision["e_cutoff"].unique()}

    data = {
        "categories": bmu.CATEGORIES,
        "category_name": {c: bmu.CATEGORY_NAME[c] for c in bmu.ORDER},
        "tools": bmu.TOOLS,
        "genes": genes,
        "gene_length": [qlen[g] for g in genes],
        "named": [[n, m, src, qlen.get(m)] for n, m, src in
                  pl.read_csv(ROOT / "data" / "botryllus_named_genes.tsv", separator="\t").iter_rows()],
        "alphabet_list": alphabets,
        "targets": [[a, c, desc.get(a, ""), n] for a, c, n in
                    zip(cat_targets["accession"], cat_targets["category"], cat_targets["length"])],
        "links": rows,
        "expected": expected,
        "alphabets": {a: {letter: "".join(sorted(aa for aa, l in m.items() if l == letter and aa != "X"))
                          for letter in sorted(set(m.values()))} for a, m in sorted(letter_maps.items())},
        "n_control": int(targets.filter(pl.col("category") == "control").height),
    }
    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.replace("/*FONTS*/", FONTS.read_text()).replace("__DATA__", payload)
    OUT.write_text(html)
    n_k = sum(1 for r in rows if r[2] is not None)
    print(f"{OUT}: {len(html) / 1e6:.1f} MB; {len(genes):_} Botryllus proteins, {len(rows):_} links "
          f"({n_k:_} with a kmerseek region), {len(cat_targets):_} targets")


TEMPLATE = r"""<title>Botryllus immune links</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/*FONTS*/
/* Layout: one column. The answer first, then filters, then one row per Botryllus protein that opens to its links and residues. */
:root {
  --bg: #F6F7F5; --panel: #FFFFFF; --fg: #1C211F; --muted: #5B6661; --line: #DCE2DE; --soft: #EDF1EE;
  --kmerseek: #0072B2; --phmmer: #B86E00; --jackhmmer: #007A5E; --mmseqs2: #A8457F;
  --match: #1C211F; --letter: #0072B2;
  --display: "Fraunces", Georgia, serif; --text: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  --mono: "Fantasque Sans Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --panel: #1C2320; --fg: #E6ECE8; --muted: #9AA7A0; --line: #2F3A35; --soft: #232B27;
  --kmerseek: #56B4E9; --phmmer: #E69F00; --jackhmmer: #2BC79A; --mmseqs2: #E08BC0; --match: #E6ECE8; --letter: #56B4E9;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --panel: #1C2320; --fg: #E6ECE8; --muted: #9AA7A0; --line: #2F3A35; --soft: #232B27;
  --kmerseek: #56B4E9; --phmmer: #E69F00; --jackhmmer: #2BC79A; --mmseqs2: #E08BC0; --match: #E6ECE8; --letter: #56B4E9;
  color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--fg); font-family: var(--text); font-size: 15px; line-height: 1.5; margin: 0 }
.wrap { max-width: 1100px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 60px; display: grid; gap: 22px }
h1 { font-family: var(--display); font-weight: 650; font-size: clamp(26px, 4vw, 38px); line-height: 1.1; margin: 0; text-wrap: balance }
h2 { font-family: var(--display); font-weight: 500; font-size: 20px; margin: 0 0 8px }
p { margin: 0; max-width: 70ch }
.lede { font-size: 17px }
.muted { color: var(--muted) }
.mono, code { font-family: var(--mono) }
details.facts { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px }
details.facts summary { cursor: pointer; font-weight: 600 }
details.facts ul { margin: 8px 0 0; padding-left: 20px; max-width: 80ch }
.controls { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
  padding-block: 10px; display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden }
.seg button { font: inherit; font-size: 14px; border: 0; background: var(--panel); color: var(--fg); padding: 4px 10px; cursor: pointer }
.seg button + button { border-left: 1px solid var(--line) }
.seg button[aria-pressed="true"] { background: var(--fg); color: var(--bg) }
.ctl { display: flex; gap: 6px; align-items: center; font-size: 14px }
.ctl > span { color: var(--muted) }
.named { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; font-size: 14px; max-width: 90ch }
input[type=search] { font: inherit; font-size: 14px; padding: 4px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--fg); width: 16em; max-width: 100% }
button:focus-visible, input:focus-visible, summary:focus-visible, .gene:focus-visible { outline: 2px solid var(--kmerseek); outline-offset: 2px }
.cats { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px }
.cat { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; align-items: start; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 8px 10px; cursor: pointer; text-align: left; font: inherit; color: var(--fg) }
.cat[aria-pressed="false"] { opacity: .45 }
.cat .box { width: 14px; height: 14px; border: 1.5px solid var(--fg); border-radius: 3px; margin-top: 3px; display: grid; place-items: center; font-size: 11px; line-height: 1 }
.cat .name { font-weight: 600 }
.cat .nums { grid-column: 2; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums }
.bar { grid-column: 2; position: relative; height: 8px; background: var(--soft); border-radius: 4px; margin-top: 4px }
.bar i { position: absolute; top: 0; bottom: 0; left: 0; background: var(--kmerseek); border-radius: 4px }
.bar b { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--fg) }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); align-items: center }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px }
.table { overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; white-space: nowrap }
th { font-weight: 600; font-size: 13px; color: var(--muted); position: relative }
th[data-sort] { cursor: pointer } th[data-sort]:hover { color: var(--fg) }
th .arrow { font-size: 11px }
td.catcell { white-space: normal; min-width: 14em }
tr.gene { cursor: pointer } tr.gene:hover td { background: var(--soft) }
tr.gene[aria-expanded="true"] td { background: var(--soft); border-bottom-color: transparent }
.ev { font-family: var(--mono); font-size: 13px }
.none { color: var(--muted) }
.tag { display: inline-block; font-size: 12px; border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; margin: 1px 3px 1px 0; white-space: nowrap }
.only { display: inline-block; font-size: 12px; font-weight: 700; color: var(--kmerseek); border: 1.5px solid var(--kmerseek); border-radius: 4px; padding: 0 5px }
.detail td { background: var(--soft); white-space: normal; padding: 4px 10px 14px }
.link { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-top: 8px; display: grid; gap: 6px; min-width: 0 }
.link h3 { font-size: 15px; margin: 0; font-weight: 600 }
.link h3 a { color: inherit }
.meta { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; color: var(--muted) }
.meta b { color: var(--fg); font-weight: 600 }
pre.res { font-family: var(--mono); font-size: 13px; line-height: 1.35; margin: 0; overflow-x: auto; background: var(--bg); padding: 8px; border-radius: 6px }
pre.res .lab { color: var(--muted) } pre.res .let { color: var(--letter) }
svg.pair { width: 100%; height: auto; display: block; background: var(--bg); border-radius: 6px }
svg.pair text { font-family: var(--text); font-size: 11px; fill: var(--fg) }
svg.pair .axis { font-size: 10px; fill: var(--muted) }
.more { justify-self: start; font: inherit; border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 6px; padding: 6px 14px; cursor: pointer }
footer { font-size: 13px; color: var(--muted); display: grid; gap: 6px }
footer p { max-width: 90ch }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto } }
</style>

<div class="wrap">
  <header style="display:grid;gap:10px">
    <h1>Botryllus immune links</h1>
    <p class="lede">Every <i>Botryllus schlosseri</i> protein that kmerseek, phmmer, jackhmmer or mmseqs2 links to a Swiss-Prot protein of the MHC system, immune recognition or self/non-self recognition. Click a protein to see each link and its residues.</p>
    <p><b>Read this first:</b> kmerseek links none of these categories more often than random Swiss-Prot proteins of the same length, and its links that no other tool finds match at about chance amino-acid identity, as its links to random proteins do (<a href="https://github.com/seanome/2024-kmerseek-analysis/pull/126">notebook 281, PR 126</a>). Treat a link that only kmerseek finds as a pattern to look at, not a homolog.</p>
    <details class="facts"><summary>The numbers behind this (E &lt; 1)</summary><ul id="facts"></ul></details>
  </header>

  <div class="controls" role="group" aria-label="Filters">
    <div class="ctl"><span>E-value below</span><div class="seg" id="ecut"></div></div>
    <div class="ctl"><span>Found by</span><div class="seg" id="found"></div></div>
    <div class="ctl"><label for="q" class="muted">Search</label><input id="q" type="search" placeholder="BHF, ABCB1, FUN020551, HLA, P15979"></div>
  </div>

  <section>
    <h2>Categories</h2>
    <div class="key" style="margin-bottom:8px">
      <span><span class="sw" style="background:var(--kmerseek)"></span>targets with a kmerseek-only link</span>
      <span><span class="sw" style="background:var(--fg);width:2px"></span>expected for random proteins of the same lengths</span>
      <span>Click a category to hide or show it. Minor histocompatibility antigens (ordinary cell proteins whose peptides MHC presents) start hidden.</span>
    </div>
    <div class="cats" id="cats"></div>
  </section>

  <section style="display:grid;gap:8px">
    <h2 id="count"></h2>
    <p class="named" id="named" hidden></p>
    <div class="key">
      <span><span class="only">kmerseek only</span> no tool hit on the pair, even at E = 10</span>
      <span>Each row's E-values are the best over all of that protein's targets, so one target can be kmerseek-only while the protein's other targets have tool hits; open the row to see each target.</span>
      <span>E-values: <span class="ev" style="color:var(--kmerseek)">kmerseek</span>, <span class="ev" style="color:var(--phmmer)">phmmer</span>, <span class="ev" style="color:var(--jackhmmer)">jackhmmer</span>, <span class="ev" style="color:var(--mmseqs2)">mmseqs2</span>; <span class="none">·</span> no hit below the cutoff</span>
    </div>
    <div class="key" aria-label="Pair view key">
      <span><b>Pair view</b> (in each opened row):</span>
      <span><svg width="26" height="10" aria-hidden="true"><line x1="0" y1="5" x2="26" y2="5" stroke="currentColor" stroke-width="1.5"/></svg>protein, drawn to scale</span>
      <span><span class="sw" style="background:var(--kmerseek);opacity:.55"></span>kmerseek region, with a band joining its two ends (darker = more alphabets agree)</span>
      <span><span class="sw" style="background:var(--phmmer)"></span>phmmer</span>
      <span><span class="sw" style="background:var(--jackhmmer)"></span>jackhmmer</span>
      <span><span class="sw" style="background:var(--mmseqs2)"></span>mmseqs2 hit, on the target only: the run kept no Botryllus-side position for tool hits</span>
    </div>
    <div class="table"><table>
      <thead><tr>
        <th data-sort="gene">Botryllus protein <span class="arrow"></span></th>
        <th data-sort="n">Targets <span class="arrow"></span></th>
        <th>Categories</th>
        <th data-sort="kmerseek" style="color:var(--kmerseek)" title="Best E-value over all of this protein's targets">kmerseek best E <span class="arrow"></span></th>
        <th data-sort="phmmer" style="color:var(--phmmer)">phmmer <span class="arrow"></span></th>
        <th data-sort="jackhmmer" style="color:var(--jackhmmer)">jackhmmer <span class="arrow"></span></th>
        <th data-sort="mmseqs2" style="color:var(--mmseqs2)">mmseqs2 <span class="arrow"></span></th>
        <th data-sort="only">Links only kmerseek finds <span class="arrow"></span></th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
    <button class="more" id="more" hidden>Show 50 more</button>
  </section>

  <footer>
    <p><b>Data.</b> Query: <i>Botryllus schlosseri</i> gene models (45_339 proteins). Target: reviewed Swiss-Prot without Ascidiacea. kmerseek 0.4.0-rc5 (invertebrate dark-set midi run, PR 57), 174 alphabet and k-size pairs, scaled 1, low-complexity mask off; each link shows its best region. Regions at least half low complexity or transmembrane helix (on either protein), and regions whose target residues are not in the named target, are left out. phmmer, jackhmmer (3 iterations) and mmseqs2 hits from the same run, E &le; 10. Categories from HGNC gene groups and UniProt (scripts/mhc_target_sets.py).</p>
    <p><b>Residues.</b> Five lines per block: Botryllus residues (Q), the same in the alphabet's letters (Q a), a match line (<code>|</code> same amino acid, <code>:</code> same letter only), the target in letters (T a), the target (T). Coordinates are 1-based, inclusive. A region chained across an insertion or deletion has strings of different lengths and no match line. <i>Chance</i> identity is what two strings of these compositions would share at random.</p>
    <p>Built by scripts/build_281_botryllus_links_page.py in seanome/2024-kmerseek-analysis.</p>
  </footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const TOOLS = D.tools, CUTS = ["0.01", "1", "10"];
const state = { cut: "1", found: "any", cats: new Set(D.categories.filter(c => c !== "minor_histocompatibility")), q: "", sort: "kmerseek", asc: true, shown: 50, open: new Set() };
try { const s = JSON.parse(localStorage.getItem("bml") || "{}"); if (CUTS.includes(s.cut)) state.cut = s.cut; if (s.found) state.found = s.found; } catch (e) {}
const save = () => { try { localStorage.setItem("bml", JSON.stringify({ cut: state.cut, found: state.found })); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtE = e => e == null ? "" : e === 0 ? "0" : e >= 0.01 ? (+e.toPrecision(2)).toString() : e.toExponential(1).replace("e-", "e-");
const geneLabel = g => { const [fun, rest] = [g.split("_")[0], g.split("_").slice(1).join("_")]; return rest && !rest.startsWith("FUN") ? `${fun} <span class="muted">${esc(rest)}</span>` : esc(g); };

// Facts: the decision numbers at E < 1.
(function () {
  const e = D.expected["1"], c = e.control, ul = document.getElementById("facts");
  const li = t => { const x = document.createElement("li"); x.innerHTML = t; ul.append(x); };
  li(`Random control proteins (${c[1]}, length-matched) with a kmerseek-only link: ${c[0]} (${(100 * c[0] / c[1]).toFixed(0)}%).`);
  for (const k of D.categories) { const v = e[k]; li(`${esc(D.category_name[k])}: ${v[0]} of ${v[1]} targets have a kmerseek-only link, against ${v[2]}% expected; ${v[3]} times the control (Bonferroni p = ${v[4] == null ? "-" : (+v[4]).toPrecision(2)}).`); }
  li("Rule set before the tests: a category counts when it is at least 1.5 times the control with Bonferroni p &lt; 0.05. None does at E &lt; 1, 0.01 or 10.");
})();

function seg(id, items, key) {
  const el = document.getElementById(id); el.innerHTML = "";
  for (const [v, label] of items) {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.setAttribute("aria-pressed", state[key] === v);
    b.onclick = () => { state[key] = v; state.shown = 50; save(); render(); }; el.append(b);
  }
}

function linkOK(L) {
  const cut = +state.cut, k = L[2];
  const kOK = k && k[0] < cut, t = TOOLS.map((_, i) => L[3 + i]), tOK = t.some(e => e != null && e < cut);
  const only = k && kOK && t.every(e => e == null);  // no tool hit at E <= 10
  if (!kOK && !tOK) return null;
  if (!state.cats.has(D.targets[L[1]][1])) return null;
  if (state.found === "only" && !only) return null;
  if (state.found === "both" && !(kOK && tOK)) return null;
  if (state.found === "tools" && !(tOK && !kOK)) return null;
  return { kOK, tOK, only, t };
}

function namedMatch() {
  const q = state.q.trim().toLowerCase();
  return q ? D.named.filter(n => n[0].toLowerCase() === q) : [];
}

function genesNow() {
  const q = state.q.trim().toLowerCase(), by = new Map(), models = namedMatch().map(n => n[1].toLowerCase());
  for (const L of D.links) {
    const ok = linkOK(L); if (!ok) continue;
    const T = D.targets[L[1]];
    const gname = D.genes[L[0]].toLowerCase();
    if (q && !(gname.includes(q) || models.includes(gname) || T[0].toLowerCase().includes(q) || T[2].toLowerCase().includes(q))) continue;
    let g = by.get(L[0]); if (!g) by.set(L[0], g = { gi: L[0], links: [], cats: new Set(), best: { kmerseek: null }, only: 0 });
    g.links.push([L, ok]); g.cats.add(T[1]); if (ok.only) g.only++;
    if (ok.kOK && (g.best.kmerseek == null || L[2][0] < g.best.kmerseek)) g.best.kmerseek = L[2][0];
    TOOLS.forEach((t, i) => { const e = ok.t[i]; if (e != null && e < +state.cut && (g.best[t] == null || e < g.best[t])) g.best[t] = e; });
  }
  const arr = [...by.values()], s = state.sort, dir = state.asc ? 1 : -1;
  const key = g => s === "gene" ? D.genes[g.gi] : s === "n" ? g.links.length : s === "only" ? g.only : g.best[s];
  arr.sort((a, b) => { const x = key(a), y = key(b);
    if (x == null && y == null) return D.genes[a.gi] < D.genes[b.gi] ? -1 : 1; if (x == null) return 1; if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * dir || (D.genes[a.gi] < D.genes[b.gi] ? -1 : 1); });
  return arr;
}

function renderCats() {
  const e = D.expected[state.cut], el = document.getElementById("cats"); el.innerHTML = "";
  for (const k of D.categories) {
    const v = e[k], on = state.cats.has(k), b = document.createElement("button");
    b.type = "button"; b.className = "cat"; b.setAttribute("aria-pressed", on);
    const share = 100 * v[0] / v[1];
    b.innerHTML = `<span class="box">${on ? "✓" : ""}</span><span class="name">${esc(D.category_name[k])}</span>
      <span class="nums">${v[0]} of ${v[1]} targets kmerseek-only (${share.toFixed(0)}%); ${v[2]}% expected</span>
      <span class="bar" aria-hidden="true"><i style="width:${share}%"></i><b style="left:calc(${v[2]}% - 1px)"></b></span>`;
    b.onclick = () => { on ? state.cats.delete(k) : state.cats.add(k); state.shown = 50; render(); };
    el.append(b);
  }
}

function residues(k) {
  const [q, t, qa, ta] = [k[7], k[8], k[9], k[10]], same = q.length === t.length, W = 60, out = [];
  for (let i = 0; i < Math.max(q.length, t.length); i += W) {
    const Q = q.slice(i, i + W), T = t.slice(i, i + W), QA = qa.slice(i, i + W), TA = ta.slice(i, i + W);
    out.push(`<span class="lab">Q   </span>${esc(Q)}`, `<span class="lab">Q a </span><span class="let">${esc(QA)}</span>`);
    if (same) out.push("    " + [...Q].map((x, j) => x === T[j] ? "|" : QA[j] === TA[j] ? ":" : " ").join(""));
    out.push(`<span class="lab">T a </span><span class="let">${esc(TA)}</span>`, `<span class="lab">T   </span>${esc(T)}`, "");
  }
  return out.join("\n");
}

function geneLines(g) {
  // FUN020551_FUN020551.1 -> ["FUN020551.1"]; FUN025982_ABCB1.1 -> ["FUN025982", "ABCB1.1"]
  const i = g.indexOf("_"), fun = i < 0 ? g : g.slice(0, i), rest = i < 0 ? "" : g.slice(i + 1);
  return !rest ? [fun] : rest.startsWith(fun) ? [rest] : [fun, rest];
}

function pairView(L, T) {
  const cut = +state.cut, qlen = D.gene_length[L[0]], tlen = T[3];
  const regs = L[3 + TOOLS.length].filter(x => x[5] < cut), hits = L[4 + TOOLS.length].filter(x => x[3] < cut);
  if (!regs.length && !hits.length) return "";
  const W = 760, X0 = 130, span = Math.max(qlen, tlen), sx = v => X0 + (v / span) * (W - X0 - 10);
  const yQ = 22, yT = 92, rowH = 14, yTool = i => yT + 18 + i * rowH, H = yTool(TOOLS.length) + 22;
  // How many alphabets put a region at the same position: shade by it.
  const byPos = new Map();
  for (const x of regs) { const k = x.slice(1, 5).join(","); byPos.set(k, (byPos.get(k) || []).concat([x])); }
  let s = `<svg class="pair" viewBox="0 0 ${W} ${H}" role="img" aria-label="Pair view of ${esc(D.genes[L[0]])} and ${esc(T[0])}">`;
  for (const [, xs] of byPos) {
    const [, qs, qe, ts, te] = xs[0], op = Math.min(0.15 + 0.12 * xs.length, 0.8);
    const tip = xs.map(x => `${D.alphabet_list[x[0]]} E=${fmtE(x[5])}`).join("; ");
    s += `<g><title>Botryllus ${qs}-${qe}, target ${ts}-${te}: ${esc(tip)}</title>
      <polygon points="${sx(qs - 1)},${yQ + 5} ${sx(qe)},${yQ + 5} ${sx(te)},${yT - 5} ${sx(ts - 1)},${yT - 5}" fill="var(--kmerseek)" opacity="${op * 0.45}"/>
      <rect x="${sx(qs - 1)}" y="${yQ - 5}" width="${Math.max(sx(qe) - sx(qs - 1), 1.5)}" height="10" fill="var(--kmerseek)" opacity="${op}"/>
      <rect x="${sx(ts - 1)}" y="${yT - 5}" width="${Math.max(sx(te) - sx(ts - 1), 1.5)}" height="10" fill="var(--kmerseek)" opacity="${op}"/></g>`;
  }
  s += `<line x1="${sx(0)}" y1="${yQ}" x2="${sx(qlen)}" y2="${yQ}" stroke="var(--fg)" stroke-width="1.5"/>
    <line x1="${sx(0)}" y1="${yT}" x2="${sx(tlen)}" y2="${yT}" stroke="var(--fg)" stroke-width="1.5"/>
    ${geneLines(D.genes[L[0]]).map((t, i) => `<text x="4" y="${yQ + 4 + 13 * i}"${i ? ' class="axis"' : ""}>${esc(t)}</text>`).join("")}
    <text x="4" y="${yQ + 4 + 13 * geneLines(D.genes[L[0]]).length}" class="axis">Botryllus, ${qlen} aa</text>
    <text x="4" y="${yT + 4}">${esc(T[0])}</text><text x="4" y="${yT + 17}" class="axis">${tlen} aa</text>`;
  TOOLS.forEach((t, i) => {
    const y = yTool(i) + 6;
    s += `<text x="${X0 - 6}" y="${y + 4}" text-anchor="end" class="axis">${t}</text>
      <line x1="${sx(0)}" y1="${y}" x2="${sx(tlen)}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;
    for (const h of hits.filter(h => h[0] === i))
      s += `<rect x="${sx(h[1] - 1)}" y="${y - 4}" width="${Math.max(sx(h[2]) - sx(h[1] - 1), 1.5)}" height="8" fill="var(--${t})"><title>${t} target ${h[1]}-${h[2]}, E=${fmtE(h[3])}</title></rect>`;
    if (!hits.some(h => h[0] === i)) s += `<text x="${sx(0) + 4}" y="${y + 4}" class="axis">no hit with E &lt; ${state.cut}</text>`;
  });
  const step = [50, 100, 200, 500, 1000, 2000].find(v => span / v <= 8) || 5000;
  for (let v = 0; v <= span; v += step) s += `<text x="${sx(v)}" y="${H - 4}" text-anchor="middle" class="axis">${v}</text>`;
  return s + `</svg>`;
}

function detail(g) {
  const items = g.links.slice().sort((a, b) => (b[1].only - a[1].only) || ((a[0][2] ? a[0][2][0] : 99) - (b[0][2] ? b[0][2][0] : 99)));
  return items.map(([L, ok]) => {
    const T = D.targets[L[1]], k = L[2];
    const tools = TOOLS.map((t, i) => L[3 + i] == null ? "" : `<span>${t} <b class="ev" style="color:var(--${t})">${fmtE(L[3 + i])}</b></span>`).join("");
    let kpart = `<span class="none">No kept kmerseek region on this pair.</span>`, res = "";
    if (k) {
      const alph = D.alphabets[k[1]], classes = Object.values(alph).filter(x => x).join(" · ");
      kpart = `<span>kmerseek <b class="ev" style="color:var(--kmerseek)">${fmtE(k[0])}</b></span>
        <span><b>${esc(k[1])}</b> k=${k[2]} <span class="mono" title="residue classes">(${esc(classes)})</span></span>
        <span>Q ${k[3]}-${k[4]}, T ${k[5]}-${k[6]}</span>
        <span>identity <b>${k[11] == null ? "-" : k[11] + "%"}</b> (chance ${k[12] == null ? "-" : k[12] + "%"})</span>
        <span>letter identity <b>${k[13] == null ? "-" : k[13] + "%"}</b> (chance ${k[14] == null ? "-" : k[14] + "%"})</span>
        <span>alphabets with a region: <b>${k[15]}</b></span>`;
      res = `<pre class="res">${residues(k)}</pre>`;
    }
    return `<div class="link"><h3>${ok.only ? '<span class="only">kmerseek only</span> ' : ""}<a href="https://www.uniprot.org/uniprotkb/${esc(T[0])}/entry" target="_blank" rel="noopener">${esc(T[0])}</a> ${esc(T[2])}
      <span class="tag">${esc(D.category_name[T[1]])}</span> <span class="muted" style="font-weight:400">${T[3]} aa</span></h3>
      <div class="meta">${kpart}${tools}</div>${pairView(L, T)}${res}</div>`;
  }).join("");
}

function render() {
  seg("ecut", CUTS.map(c => [c, c]), "cut");
  seg("found", [["any", "Any"], ["only", "kmerseek only"], ["both", "kmerseek and a tool"], ["tools", "Tools only"]], "found");
  renderCats();
  const arr = genesNow(), body = document.getElementById("rows"); body.innerHTML = "";
  const nl = arr.reduce((s, g) => s + g.links.length, 0);
  document.getElementById("count").textContent = `${arr.length.toLocaleString()} Botryllus proteins, ${nl.toLocaleString()} links`;
  const nm = namedMatch(), box = document.getElementById("named");
  box.hidden = !nm.length;
  box.innerHTML = nm.map(([name, model, src, len]) => {
    const gi = D.genes.indexOf(model), shown = arr.some(g => g.gi === gi);
    const status = gi < 0
      ? `has <b>no link</b> to any protein in these categories: no kmerseek region with E &lt; 10 that passes the filters, and no phmmer, jackhmmer or mmseqs2 hit with E &le; 10.`
      : shown ? `is listed below.` : `has links, but none with the current E-value cutoff, tool filter and categories; try E-value below 10 and Found by: Any.`;
    return `<b>${esc(name)}</b> is gene model <span class="mono">${esc(model)}</span>${len ? ` (${len} aa)` : ""}. It ${status} <span class="muted">${esc(src)}.</span>`;
  }).join("<br>");
  document.querySelectorAll("th[data-sort]").forEach(th => th.querySelector(".arrow").textContent = th.dataset.sort === state.sort ? (state.asc ? "▲" : "▼") : "");
  for (const g of arr.slice(0, state.shown)) {
    const tr = document.createElement("tr"); tr.className = "gene"; tr.tabIndex = 0; const open = state.open.has(g.gi);
    tr.setAttribute("aria-expanded", open);
    const ev = t => g.best[t] == null ? '<span class="none">·</span>' : `<span class="ev" style="color:var(--${t})">${fmtE(g.best[t])}</span>`;
    const cats = [...g.cats].sort((a, b) => D.categories.indexOf(a) - D.categories.indexOf(b)).map(c => `<span class="tag">${esc(D.category_name[c])}</span>`).join("");
    tr.innerHTML = `<td class="mono">${geneLabel(D.genes[g.gi])}</td><td>${g.links.length}</td><td class="catcell">${cats}</td>
      <td>${ev("kmerseek")}</td>${TOOLS.map(t => `<td>${ev(t)}</td>`).join("")}<td>${g.only ? `${g.only} of ${g.links.length}` : '<span class="none">0</span>'}</td>`;
    const toggle = () => { open ? state.open.delete(g.gi) : state.open.add(g.gi); render(); };
    tr.onclick = toggle; tr.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } };
    body.append(tr);
    if (open) { const d = document.createElement("tr"); d.className = "detail"; d.innerHTML = `<td colspan="8">${detail(g)}</td>`; body.append(d); }
  }
  const more = document.getElementById("more"); more.hidden = arr.length <= state.shown;
  more.textContent = `Show 50 more (${(arr.length - state.shown).toLocaleString()} left)`;
}
document.getElementById("more").onclick = () => { state.shown += 50; render(); };
document.getElementById("q").oninput = e => { state.q = e.target.value; state.shown = 50; render(); };
document.querySelectorAll("th[data-sort]").forEach(th => th.onclick = () => {
  const s = th.dataset.sort; if (state.sort === s) state.asc = !state.asc; else { state.sort = s; state.asc = !["n", "only"].includes(s); } render(); });
render();
</script>
"""

if __name__ == "__main__":
    main()
