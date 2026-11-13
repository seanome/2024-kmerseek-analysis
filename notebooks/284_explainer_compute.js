// Sums for the notebook 284 explainer page. The spots themselves are built in Python
// (scripts/284_explainer_data.py, with alphabet_agreement_utils.assign_spots); this file only
// reads the stored tables, applies the reader's choice of counting rule and alphabets, and
// counts proteins. The three rules match alphabet_agreement_utils.spots:
//   n_alphabets                 alphabets in the spot;
//   n_families                  the same, with the seven H/P alphabets counted as one;
//   n_families_identity_as_one  n_families, but 1 for any spot protein20 also calls.
// tests/284_explainer_compute.test.mjs checks every result against the committed CSVs.
(function (root) {
  "use strict";

  function popcount(x) {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }

  // nHp: the H/P alphabets are bits 0..nHp-1; p20Bit: protein20's bit.
  function makeCounter(rule, nHp, p20Bit) {
    const hp = (1 << nHp) - 1, p20 = 1 << p20Bit;
    if (rule === "n_alphabets") return m => popcount(m);
    const fam = m => popcount(m & ~hp) + ((m & hp) ? 1 : 0);
    if (rule === "n_families") return fam;
    if (rule === "n_families_identity_as_one") return m => (m & p20) ? 1 : fam(m);
    throw new Error("unknown rule " + rule);
  }

  // One stored file: typed views over the buffer, and where each spot's alphabet entries start.
  function parseSpots(buf, layout) {
    const v = (Type, key) => new Type(buf, layout[key][0], layout[key][1]);
    const S = {
      t: v(Uint16Array, "t"), qs: v(Uint16Array, "qs"), qe: v(Uint16Array, "qe"),
      diag: v(Int32Array, "diag"), mask: v(Uint32Array, "mask"),
      a_qs: v(Uint16Array, "a_qs"), a_qe: v(Uint16Array, "a_qe"),
      a_e: v(Float32Array, "a_e"), a_k: v(Uint8Array, "a_k"), union: v(Uint32Array, "union"),
    };
    S.n = S.t.length;
    S.start = new Uint32Array(S.n + 1);
    for (let i = 0; i < S.n; i++) S.start[i + 1] = S.start[i] + popcount(S.mask[i]);
    if (S.start[S.n] !== S.a_qs.length) throw new Error("entry count does not match the masks");
    return S;
  }

  // Lowest E-value among the counted alphabets of stored spot i.
  function minE(S, i, keep) {
    let e = Infinity, k = S.start[i];
    for (let bit = 0; bit < 32; bit++) if (S.mask[i] & (1 << bit)) { if ((keep & (1 << bit)) && S.a_e[k] < e) e = S.a_e[k]; k++; }
    return e;
  }

  // Each target's highest count over its spots, and which stored spot gives it (-1 when the
  // best is a spot one alphabet backs, which is not stored one by one).
  function bestPerTarget(S, count, keep) {
    const nT = S.union.length, best = new Int32Array(nT), spot = new Int32Array(nT).fill(-1);
    for (let t = 0; t < nT; t++) best[t] = (S.union[t] & keep) ? 1 : 0;
    for (let i = 0; i < S.n; i++) {
      const m = S.mask[i] & keep, c = count(m), t = S.t[i];
      if (c === 0) continue;
      // ties: the stored spot with more alphabets, then the lower E-value, then the first
      let take = c > best[t] || (c === best[t] && spot[t] < 0);
      if (!take && c === best[t]) {
        const n = popcount(m), n0 = popcount(S.mask[spot[t]] & keep);
        take = n > n0 || (n === n0 && minE(S, i, keep) < minE(S, spot[t], keep));
      }
      if (take) { best[t] = c; spot[t] = i; }
    }
    return {best, spot};
  }

  // The numbers the page prints for one setting.
  function summarise(best, partner) {
    let withCall = 0, max = 0, atMax = 0, more = 0;
    const v = partner >= 0 ? best[partner] : 0, hist = new Map();
    for (let t = 0; t < best.length; t++) {
      const b = best[t];
      if (b === 0) continue;
      withCall++;
      hist.set(b, (hist.get(b) || 0) + 1);
      if (b > max) { max = b; atMax = 0; }
      if (b === max) atMax++;
      if (b > v) more++;
    }
    return {partner_value: v, proteins_with_more: more, proteins_with_a_call: withCall,
            proteome_max: max, proteins_at_max: atMax, hist};
  }

  const api = {popcount, makeCounter, parseSpots, bestPerTarget, summarise};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Agree = api;
})(this);
