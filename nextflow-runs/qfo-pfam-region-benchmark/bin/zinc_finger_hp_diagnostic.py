#!/usr/bin/env python3
"""Is the C2H2 zinc-finger signature rare in HP space, or generic?

The decision this answers, before any compute is spent: whether zinc fingers lead the
domain-annotation story or are reported as the honest limit.

C2H2 is a Cys/His-anchored motif, and Folddisco's own index statistics make Cys-Trp the
rarest residue pair in AFDB50 (0.034%) -- rare means informative. But every HP alphabet
here merges cysteine into one of two 10-residue classes, so the question is whether the
motif survives that merge.

Rarity alone is the wrong measure: C2H2 occurs in ~763 human proteins, often as tandem
arrays, so its k-mers are COMMON by construction. What matters is SPECIFICITY -- of all
the places a C2H2-derived k-mer occurs in the proteome, what fraction are themselves
inside an annotated C2H2 domain. High specificity means the HP pattern identifies the
motif; low means it is generic background that happens to also occur there.

Run across alphabets that disagree about cysteine, which turns a parameter choice into a
biological result:
    C in the HYDROPHOBIC class: pbotc_1st_ed (production), lehninger_plus_c, thomas_dill
    C in the POLAR class:       lehninger, thomas_dill_no_c
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

# Verbatim from src/rust/hp_alphabets.rs. Each maps to (hydrophobic_residues,) with
# everything else polar; build_hp() there asserts the two sets cover all 20.
HP_ALPHABETS = {
    "hp_lehninger":         "AFGILMPVWY",
    "hp_thomas_dill":       "ACFILMVWY",
    "hp_thomas_dill_no_c":  "AFILMVWY",
    "hp_lehninger_plus_c":  "ACFGILMPVWY",
    "hp_pbotc_1st_ed":      "ACFILMPVWY",
}
C2H2_PFAM = "PF00096"


def encode(seq: str, h_set: frozenset) -> str:
    return "".join("h" if c in h_set else "p" for c in seq)


def read_fasta(path: Path) -> dict[str, str]:
    recs, name, buf = {}, None, []
    opener = open
    with opener(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    recs[name] = "".join(buf)
                parts = line[1:].split("|")
                name = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        recs[name] = "".join(buf)
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--ksize", type=int, default=19)
    p.add_argument("--pfam-id", default=C2H2_PFAM)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    seqs = read_fasta(args.fasta)
    dom = pl.read_parquet(args.annotations / "human_pfam_domains.parquet").filter(
        pl.col("has_position") & (pl.col("pfam_id") == args.pfam_id)
    )
    if dom.height == 0:
        raise SystemExit(f"no {args.pfam_id} domains found")

    # Residue-level mask of "inside a target domain", per protein.
    inside = defaultdict(set)
    for acc, s, e in dom.select("accession", "domain_start", "domain_end").iter_rows():
        if acc in seqs:
            inside[acc].update(range(max(0, s - 1), min(len(seqs[acc]), e)))

    n_prot = len({a for a in inside if a in seqs})
    print(f"{args.pfam_id}: {dom.height} domain instances across {n_prot} human proteins")
    print(f"ksize {args.ksize}\n")

    results = {}
    for name, h_res in HP_ALPHABETS.items():
        h_set = frozenset(h_res)
        total = Counter()      # k-mer -> occurrences anywhere in the proteome
        in_dom = Counter()     # k-mer -> occurrences that sit inside a target domain
        for acc, seq in seqs.items():
            enc = encode(seq, h_set)
            dom_pos = inside.get(acc, ())
            for i in range(len(enc) - args.ksize + 1):
                km = enc[i:i + args.ksize]
                total[km] += 1
                # Count as "in domain" only when the whole k-mer lies inside one.
                if dom_pos and all(j in dom_pos for j in range(i, i + args.ksize)):
                    in_dom[km] += 1

        if not in_dom:
            print(f"{name:22s} no k-mers fully inside a domain")
            continue

        # Specificity, weighted by how often each k-mer appears in the domains: of all
        # proteome occurrences of C2H2-derived k-mers, what share is inside C2H2.
        num = sum(in_dom[k] for k in in_dom)
        den = sum(total[k] for k in in_dom)
        specificity = num / den

        uniq = len(in_dom)
        exclusive = sum(1 for k in in_dom if total[k] == in_dom[k])
        median_occ = sorted(total[k] for k in in_dom)[uniq // 2]

        results[name] = {
            "cysteine_class": "hydrophobic" if "C" in h_res else "polar",
            "n_distinct_kmers_in_domain": uniq,
            "specificity": round(specificity, 4),
            "frac_kmers_exclusive_to_domain": round(exclusive / uniq, 4),
            "median_proteome_occurrences": median_occ,
        }
        print(f"{name:22s} C={results[name]['cysteine_class']:11s} "
              f"specificity={specificity:.4f}  "
              f"exclusive={exclusive / uniq:.3f}  "
              f"median_occ={median_occ}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))

    # The decision the checklist asks for, stated rather than left to interpretation.
    print()
    best = max(results.items(), key=lambda kv: kv[1]["specificity"])
    prod = results.get("hp_pbotc_1st_ed")
    print(f"most specific alphabet: {best[0]} ({best[1]['specificity']:.4f})")
    if prod:
        print(f"production (pbotc_1st_ed): {prod['specificity']:.4f}")
    h = [v["specificity"] for v in results.values() if v["cysteine_class"] == "hydrophobic"]
    pol = [v["specificity"] for v in results.values() if v["cysteine_class"] == "polar"]
    if h and pol:
        print(f"C-hydrophobic mean {sum(h)/len(h):.4f}  vs  C-polar mean {sum(pol)/len(pol):.4f}")


if __name__ == "__main__":
    main()
