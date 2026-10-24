// Checks the explainer page's sums (notebooks/284_explainer_compute.js) against the tables the
// Python scripts wrote. Run by test_284_explainer_compute.py, or: node scripts/tests/284_explainer_compute_check.mjs
// 1. With every alphabet counted, each query x E-value cut x counting rule must reproduce
//    notebooks/284_agreement_by_evalue_cut.csv exactly (42 rows).
// 2. All regions, alphabets counted: the histogram must equal 284_alphabet_agreement_data.json.
// 3. Removing alphabets is a shortcut on the page (see 284_explainer_removal_check.py); print
//    how far it is from a fresh count, and fail only if the partner's value moves by more than 1.
import fs from "node:fs";
import path from "node:path";
import {createRequire} from "node:module";
import {fileURLToPath} from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const NB = path.join(ROOT, "notebooks");
const A = createRequire(import.meta.url)(path.join(NB, "284_explainer_compute.js"));
const man = JSON.parse(fs.readFileSync(path.join(NB, "284_explainer_manifest.json")));

function csv(file) {
  const [head, ...lines] = fs.readFileSync(path.join(NB, file), "utf8").trim().split("\n");
  const cols = head.split(",");
  return lines.map(l => {
    const vals = l.match(/("[^"]*"|[^,]*)(,|$)/g).map(v => v.replace(/,$/, "").replace(/^"|"$/g, ""));
    return Object.fromEntries(cols.map((c, i) => [c, vals[i]]));
  });
}
const cache = new Map();
function load(q, ci) {
  const name = `spots.${q}.${ci}.bin`;
  if (!cache.has(name)) {
    const b = fs.readFileSync(path.join(NB, "284_explainer_files", name));
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    cache.set(name, A.parseSpots(buf, man.files[name].layout));
  }
  return cache.get(name);
}
const ALL = (1 << man.alphabets.length) - 1;
const bit = a => 1 << man.alphabets.indexOf(a);
const p20 = man.alphabets.indexOf("protein20");
const run = (q, ci, rule, keep) => {
  const S = load(q, ci), {best} = A.bestPerTarget(S, A.makeCounter(rule, man.hp.length, p20), keep);
  return A.summarise(best, man.genes.indexOf(man.partners[q]));
};
const cutLabel = c => c === null ? "all regions" : `E < ${c}`;

let fails = 0, n = 0;
const FIELDS = ["proteins_with_a_call", "proteome_max", "proteins_at_max", "partner_value", "proteins_with_more"];
for (const r of csv("284_agreement_by_evalue_cut.csv").filter(r => r.query in man.partners)) {
  const ci = man.cuts.findIndex(c => cutLabel(c) === r.e_cut);
  if (ci < 0) throw new Error("no cut for " + r.e_cut);
  const s = run(r.query, ci, r.count, ALL);
  n++;
  for (const f of FIELDS) if (String(s[f]) !== r[f]) { fails++; console.log("MISMATCH", r.query, r.e_cut, r.count, f, s[f], "csv", r[f]); }
}
console.log(`check 1: ${n} rows of 284_agreement_by_evalue_cut.csv, ${fails} mismatches`);

const data = JSON.parse(fs.readFileSync(path.join(NB, "284_alphabet_agreement_data.json")));
for (const q of Object.keys(man.partners)) {
  const s = run(q, 0, "n_alphabets", ALL);
  for (const [k, v] of Object.entries(data.histogram[q].counts))
    if (s.hist.get(+k) !== v) { fails++; console.log("HIST MISMATCH", q, k, s.hist.get(+k), v); }
}
console.log(`check 2: histograms, ${fails} mismatches so far`);

let worst = 0, worstMore = 0, nDiffer = 0;
const rows = csv("284_explainer_removal_check.csv");
for (const r of rows) {
  const keep = r.removed_alphabets.split(",").reduce((k, a) => k & ~bit(a), ALL);
  const ci = man.cuts.findIndex(c => (c === null ? -1 : c) === Number(r.e_cut));
  const s = run(r.query, ci, r.count, keep);
  const dv = Math.abs(s.partner_value - r.partner_value), dm = s.proteins_with_more - r.proteins_with_more;
  worst = Math.max(worst, dv); worstMore = Math.max(worstMore, Math.abs(dm));
  if (dv || dm || s.proteins_with_a_call !== +r.proteins_with_a_call) nDiffer++;
  if (dv || dm || s.proteins_with_a_call !== +r.proteins_with_a_call)
    console.log("removal", r.query, r.removed, r.e_cut, r.count, "page", s.partner_value, s.proteins_with_more,
                "fresh", r.partner_value, r.proteins_with_more);
  if (dv > 1) fails++;
}
console.log(`check 3: ${rows.length} removal rows, largest partner-value gap ${worst}, largest gap in proteins with more ${worstMore}`);
// the page prints this, so the sentence about the shortcut stays tied to the check
fs.writeFileSync(path.join(NB, "284_explainer_removal_gap.json"), JSON.stringify({
  n_rows: rows.length, removals: [...new Set(rows.map(r => r.removed))],
  largest_partner_value_gap: worst, largest_proteins_with_more_gap: worstMore,
  rows_that_differ: nDiffer}, null, 1) + "\n");
process.exit(fails ? 1 : 0);
