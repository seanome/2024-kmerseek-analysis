#!/usr/bin/env python3
"""What does kmerseek reach that no sequence arm could?

The only number this pipeline exists for. The dark set is the proteins phmmer, jackhmmer and
mmseqs2 all failed to place; this counts how many of those kmerseek puts a region on, split
by alphabet, ksize, and -- the part that decides whether any of it is real -- whether the
low-complexity mask was on.

Read the mask pair, never one side alone. BHF's flagship matches included polar-biased
low-complexity segments, so a rescue present with the mask off and absent with it on is an
artifact of composition, not a homology signal. This reports both and the difference; a
large mask-off-only number is a negative result wearing a positive one's clothes.

This is REACH, not accuracy. A region on a dark protein says kmerseek found something
there, not that the family label is right -- that needs the structural key. A number here
is necessary for the claim and nowhere near sufficient.

The PLACED set is counted next to the dark one, for the same combos, because reach alone
saturates. On Botryllus, hp_thomas_dill2 k23 put a region on 20_448 of 20_448 dark
proteins -- and on all 45_339 proteins of the proteome, which is not a rescue, it is an arm
that reports something for everything. An arm that reaches 95% of the placed proteins and
10% of the dark ones is discriminating; one that reaches 100% of both is saturated, and
its dark number means nothing. The two columns side by side are what tell those apart,
and across a ksize sweep the k where the placed fraction starts to fall is the k where
the dark fraction starts to mean something.

Two more controls, both optional because they need products an older run did not write:

  --scores   per-query maximum region score (<chunk>.<combo>.query_scores.tsv, written
             by kmerseekSearch since 2026-09-20). With these, reach is also counted at
             stricter cutoffs than the run's own (--thresholds), so a 100% at the run
             cutoff can be read against what survives a tighter one.
  --shuffled the same query lists for the dark proteins with their residues shuffled
             (same length, same composition, no homology). What kmerseek reaches on those
             is what it reaches by composition alone; a dark reach that the shuffled
             sequences match is not a homology signal.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl


def query_accessions(fasta: Path) -> set[str]:
    """First whitespace token of every header, verbatim -- the key every arm reports."""
    accs = set()
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                accs.add(line[1:].strip().split()[0])
    return accs

# <chunk>.<alphabet>.k<ksize>.lc<true|false>.queries.txt, or .query_scores.tsv
NAME = re.compile(r"^(?P<chunk>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.lc(?P<lc>true|false)"
                  r"\.(?P<kind>queries\.txt|query_scores\.tsv)$")


def paired_wins(real_best: dict, shuf_best: dict, accs, deltas) -> dict:
    """Each protein against ITS OWN shuffled copy, same length and composition.

    This is the readout that survives saturation. A per-combo fraction is unreadable once
    the cutoff reaches every protein -- 20_448 of 20_448 against a control that is also
    20_448 of 20_448 says nothing. The paired difference still separates them, because it
    asks whether THIS protein scored better than its own shuffle, not whether some
    threshold was cleared.

    A protein neither side reached carries no information, so it is out of the
    denominator; n_compared reports how many were left. A protein only one side reached
    counts, with the missing side as minus infinity.
    """
    out = {"n_compared": 0, "wins_at": {}}
    pairs = []
    for a in accs:
        r = real_best.get(a)
        h = shuf_best.get(a)
        if r is None and h is None:
            continue
        pairs.append((r if r is not None else float("-inf"),
                      h if h is not None else float("-inf")))
    out["n_compared"] = len(pairs)
    for d in deltas:
        n = sum(1 for r, h in pairs if r - h >= d and r != float("-inf"))
        out["wins_at"][str(d)] = n
        out.setdefault("fraction_at", {})[str(d)] = (round(n / len(pairs), 4)
                                                     if pairs else None)
    return out


def read_lists(files, what: str) -> tuple[dict, dict]:
    """combo -> set of query accessions with any region, and combo -> {accession: max
    region score} where the file carries scores. Unioned across chunks."""
    hits: dict[tuple, set] = {}
    scores: dict[tuple, dict] = {}
    unparsed = []
    for f in files:
        m = NAME.match(f.name)
        if not m:
            unparsed.append(f.name)
            continue
        key = (m["alphabet"], int(m["ksize"]), m["lc"] == "true")
        if m["kind"] == "queries.txt":
            got = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
            hits.setdefault(key, set()).update(got)
        else:
            best = scores.setdefault(key, {})
            for ln in f.read_text().splitlines():
                if not ln.strip():
                    continue
                cols = ln.rstrip("\n").split("\t")
                acc, v = cols[0], float(cols[1])
                # Columns 3 and 4, where present, are the best region's start and end.
                region = (int(cols[2]), int(cols[3])) if len(cols) >= 4 else None
                if v > best.get(acc, (float("-inf"), None))[0]:
                    best[acc] = (v, region)
            hits.setdefault(key, set()).update(best)
    if unparsed:
        raise SystemExit(f"could not parse combo from {what}: {', '.join(unparsed[:5])}")
    return hits, scores


def focus_proteins(registry: Path | None, species: str) -> dict[str, str]:
    """accession -> what it is, from the registry row's focus_proteins; {} without one."""
    if registry is None or not registry.exists():
        return {}
    row = json.loads(registry.read_text()).get(species) or {}
    return dict(row.get("focus_proteins") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True,
                    help="the proteome every arm searched; the placed set is this minus --dark")
    ap.add_argument("--species", required=True)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--scores", type=Path, nargs="*", default=[],
                    help="per-query maximum region score files, one per chunk and combo; "
                         "optional, and absent from runs before 2026-09-20")
    ap.add_argument("--run-cutoff", type=float, default=None,
                    help="the --min-region-score the search itself ran at, so the output "
                         "records which threshold was the run's own rather than leaving it "
                         "to be inferred. main.nf passes params.min_region_score; the "
                         "default here is only for running this by hand.")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.3, 3.0, 10.0],
                    help="region-score cutoffs to count reach at when --scores is given; "
                         "the first should be the run's own --min-region-score")
    ap.add_argument("--shuffled", type=Path, nargs="*", default=[],
                    help="query lists from the same combos run on the dark proteins with "
                         "their residues shuffled; optional")
    ap.add_argument("--registry", type=Path, default=None,
                    help="species_metadata.json; its focus_proteins for --species get "
                         "their reach, best region score and best region written per "
                         "combo into the summary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    dark = set(pl.read_parquet(args.dark)["accession"].to_list())
    proteome = query_accessions(args.query)
    stray = dark - proteome
    if stray:
        raise SystemExit(
            f"{len(stray)} dark accessions are not in {args.query} (e.g. "
            f"{sorted(stray)[:3]}): the dark set and the proteome disagree on the key.")
    placed = proteome - dark
    if not dark:
        raise SystemExit(
            f"the dark set for '{args.species}' is empty, so there is nothing for kmerseek "
            f"to be measured on. Either every protein was placed by a sequence arm, or "
            f"computeDarkSet ran against a broken reference."
        )

    # combo -> set of query accessions with any region, unioned across chunks
    combos, _ = read_lists(args.queries, "--queries")
    _, scores = read_lists(args.scores, "--scores")
    shuffled, shuffled_scores = read_lists(args.shuffled, "--shuffled")
    # The shuffled lists carry the dark accessions with a suffix or not at all, depending
    # on how the shuffled FASTA named them; either way only the dark set can be in them.
    shuffled = {k: {a.removesuffix("_shuffled") for a in v} & dark for k, v in shuffled.items()}
    # The scores were being dropped on the floor, which left the control alive only at the
    # run's own cutoff -- the cutoff at which everything is reached, so the comparison said
    # nothing. Same accession rule as the hit lists above.
    shuffled_scores = {k: {a.removesuffix("_shuffled"): v for a, v in d.items()}
                       for k, d in shuffled_scores.items()}
    # The run's own cutoff is always among the thresholds, whatever else was asked for:
    # a control that exists only at cutoffs the run never used cannot speak about the run.
    thresholds = sorted(set(args.thresholds) | ({args.run_cutoff}
                                                if args.run_cutoff is not None else set()))

    rows = []
    for (alphabet, ksize, lc), hit in sorted(combos.items()):
        rescued = hit & dark
        seen_placed = hit & placed
        row = {
            "species": args.species, "alphabet": alphabet, "ksize": ksize,
            "low_complexity_mask": lc,
            "dark_proteins": len(dark),
            "dark_reached": len(rescued),
            "fraction_dark_reached": round(len(rescued) / len(dark), 4),
            "placed_proteins": len(placed),
            "placed_reached": len(seen_placed),
            "fraction_placed_reached": (round(len(seen_placed) / len(placed), 4)
                                        if placed else None),
            "queries_with_any_region": len(hit),
        }
        key = (alphabet, ksize, lc)
        if key in scores:
            best = {a: v for a, (v, _r) in scores[key].items()}
            row["dark_reached_at"] = {
                str(t): sum(1 for a in rescued if best.get(a, float("-inf")) >= t)
                for t in thresholds}
            row["placed_reached_at"] = {
                str(t): sum(1 for a in seen_placed if best.get(a, float("-inf")) >= t)
                for t in thresholds}
        if key in shuffled:
            row["shuffled_dark_reached"] = len(shuffled[key])
            row["fraction_shuffled_dark_reached"] = round(len(shuffled[key]) / len(dark), 4)
        if key in shuffled_scores:
            shuf_best = {a: v for a, (v, _r) in shuffled_scores[key].items()}
            row["shuffled_dark_reached_at"] = {
                str(t): sum(1 for a in dark if shuf_best.get(a, float("-inf")) >= t)
                for t in thresholds}
            if key in scores:
                real_best = {a: v for a, (v, _r) in scores[key].items()}
                # Delta 0 is the sign test: did this protein beat its own shuffle at all.
                deltas = [0.0] + thresholds
                row["dark_vs_own_shuffle"] = paired_wins(real_best, shuf_best, dark, deltas)
                # The positive control: if the PLACED proteins do not beat their
                # shuffles either, the arm is reading composition and the dark number
                # means nothing. Only computable when placed proteins were actually
                # shuffled -- shuffle_dark_queries.py does the dark set by default, and
                # --also-placed adds a sample. Without them every placed protein would
                # face a missing shuffle, score an automatic win, and the control would
                # read 100% while measuring nothing.
                placed_shuffled = placed & set(shuf_best)
                if placed_shuffled:
                    row["placed_vs_own_shuffle"] = paired_wins(
                        real_best, shuf_best, placed_shuffled, deltas)
                else:
                    row["placed_vs_own_shuffle"] = None
                    row["placed_control_absent"] = (
                        "no placed protein was shuffled; rerun shuffle_dark_queries.py "
                        "with --also-placed N")
        rows.append(row)
    # The parquet keeps the flat columns; the nested per-threshold counts are in the JSON.
    df = pl.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    df.write_parquet(args.out, compression="zstd")

    # The paired difference is the headline, not either column on its own.
    paired = []
    for alphabet, ksize in sorted({(r["alphabet"], r["ksize"]) for r in rows}):
        on = next((r for r in rows if r["alphabet"] == alphabet and r["ksize"] == ksize
                   and r["low_complexity_mask"]), None)
        off = next((r for r in rows if r["alphabet"] == alphabet and r["ksize"] == ksize
                    and not r["low_complexity_mask"]), None)
        if on and off:
            paired.append({
                "alphabet": alphabet, "ksize": ksize,
                "dark_reached_mask_on": on["dark_reached"],
                "dark_reached_mask_off": off["dark_reached"],
                "lost_to_masking": off["dark_reached"] - on["dark_reached"],
            })

    # The focus proteins, per combo: reached or not, the best region score and where that
    # region sits, and whether the shuffled copy was reached too.
    focus = {}
    for acc, what in focus_proteins(args.registry, args.species).items():
        per = {}
        for (alphabet, ksize, lc), hit in sorted(combos.items()):
            v, region = scores.get((alphabet, ksize, lc), {}).get(acc, (None, None))
            per[f"{alphabet} k{ksize} lc{str(lc).lower()}"] = {
                "alphabet": alphabet, "ksize": ksize, "low_complexity_mask": lc,
                "reached": acc in hit, "best_region_score": v,
                "best_region": list(region) if region else None,
                "shuffled_reached": (acc in shuffled[(alphabet, ksize, lc)]
                                     if (alphabet, ksize, lc) in shuffled else None)}
        focus[acc] = {"what": what, "dark": acc in dark, "per_combo": per}

    summary = {"species": args.species, "dark_proteins": len(dark),
               "placed_proteins": len(placed),
               "thresholds": thresholds if scores else [],
               "run_cutoff": args.run_cutoff,
               "has_shuffled_control": bool(shuffled),
               "by_combo": rows, "mask_pairs": paired,
               "focus_proteins": focus}
    print(json.dumps(summary, indent=2))
    # The paired comparison first, because it is the number that survives saturation.
    # "Reached" counts are a threshold statement and go under it, not above it.
    beat = [r for r in rows if r.get("dark_vs_own_shuffle")]
    if beat:
        print("\neach dark protein against its own shuffled copy (the number that counts):")
        for r in beat:
            d = r["dark_vs_own_shuffle"]
            n = d["n_compared"]
            w = d["wins_at"].get("0.0", d["wins_at"].get("0"))
            frac = f"{100 * w / n:.1f}%" if n else "n/a"
            print(f"  {r['alphabet']} k{r['ksize']} lc{str(r['low_complexity_mask']).lower()}: "
                  f"{w} of {n} dark proteins beat their own shuffle ({frac})")
            pc = r.get("placed_vs_own_shuffle")
            if pc and pc["n_compared"]:
                pw = pc["wins_at"].get("0.0", pc["wins_at"].get("0"))
                print(f"      positive control, placed proteins: {pw} of "
                      f"{pc['n_compared']} beat theirs "
                      f"({100 * pw / pc['n_compared']:.1f}%)")
            elif r.get("placed_control_absent"):
                print(f"      positive control: {r['placed_control_absent']}")
    elif args.shuffled:
        print("\nno paired comparison: the shuffled arm ran but wrote no scores, so the "
              "control exists only at the run cutoff, where everything is reached.")

    if paired:
        print("\nsurviving the low-complexity mask:")
        for q in paired:
            print(f"  {q['alphabet']} k{q['ksize']}: {q['dark_reached_mask_on']} of "
                  f"{len(dark)} dark proteins reached with the mask ON "
                  f"({q['lost_to_masking']} lost when it is applied)")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
