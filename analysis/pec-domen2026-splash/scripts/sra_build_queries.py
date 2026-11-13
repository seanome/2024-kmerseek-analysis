"""Build the protein queries for the read search: every isoform of every gene on the paper's lists,
plus a length-matched set of random botznik proteins from genes on no list.

A gene g<N> in `paper_named_lists.tsv` maps to every botznik protein whose original_id is
g<N>.t<M>. A gene can sit on several lists (contrast x direction), so the query map has one row
per (protein, list). For the random set, each list protein gets one partner drawn without
replacement from the non-list proteins whose length is closest to its own (the 25 nearest by
length), so the two sets have nearly the same length distribution.

Usage: python sra_build_queries.py <paper_named_lists.tsv> <botznik.faa> <botznik.id_map.tsv> <out_dir> [n_pilot]
Writes queries.faa, query_map.tsv and (if n_pilot) pilot_queries.faa into <out_dir>.
"""

import bisect
import csv
import random
import sys
from pathlib import Path

SEED = 361


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    for line in path.open():
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def main() -> None:
    lists_tsv, faa, id_map, out = map(Path, sys.argv[1:5])
    n_pilot = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(faa)
    gene_of, orig_of = {}, {}
    for r in csv.DictReader(id_map.open(), delimiter="\t"):
        orig_of[r["stable_id"]] = r["original_id"]
        if r["original_id"].startswith("g") and ".t" in r["original_id"]:
            gene_of[r["stable_id"]] = r["original_id"].split(".t")[0]
    isoforms: dict[str, list[str]] = {}
    for sid, g in gene_of.items():
        isoforms.setdefault(g, []).append(sid)
    lists = list(csv.DictReader(lists_tsv.open(), delimiter="\t"))
    list_genes = {r["gene_id"] for r in lists}
    missing = sorted(g for g in list_genes if g not in isoforms)
    print(
        f"list rows {len(lists)}, distinct genes {len(list_genes)}, genes with no botznik protein {len(missing)}: {missing[:10]}"
    )

    rows = []
    for r in lists:
        for sid in sorted(isoforms.get(r["gene_id"], [])):
            rows.append(
                {
                    "query": sid,
                    "original_id": orig_of[sid],
                    "gene": r["gene_id"],
                    "gene_name": r["gene_name"],
                    "contrast": r["contrast"],
                    "direction": r["direction"],
                    "set": "paper_list",
                    "length": len(seqs[sid]),
                }
            )
    list_prots = sorted({x["query"] for x in rows})
    # random partners: proteins of genes on no list, matched on length
    pool = sorted(
        (len(seqs[s]), s)
        for s, g in gene_of.items()
        if g not in list_genes and s in seqs
    )
    rng = random.Random(SEED)
    used, partner = set(), {}
    for sid in sorted(list_prots, key=lambda s: (len(seqs[s]), s)):
        n = len(seqs[sid])
        i = bisect.bisect_left(pool, (n, ""))
        cand = [p for p in pool[max(0, i - 40) : i + 40] if p[1] not in used]
        cand.sort(key=lambda p: (abs(p[0] - n), p[1]))
        pick = rng.choice(cand[:25])[1]
        used.add(pick)
        partner[sid] = pick
        rows.append(
            {
                "query": pick,
                "original_id": orig_of[pick],
                "gene": gene_of[pick],
                "gene_name": "",
                "contrast": "random_control",
                "direction": "none",
                "set": "random_length_matched",
                "length": len(seqs[pick]),
                "matched_to": sid,
            }
        )
    cols = [
        "query",
        "original_id",
        "gene",
        "gene_name",
        "contrast",
        "direction",
        "set",
        "length",
        "matched_to",
    ]
    with (out / "query_map.tsv").open("w") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)
    all_q = list_prots + sorted(partner.values())
    with (out / "queries.faa").open("w") as f:
        for q in all_q:
            f.write(f">{q}\n{seqs[q]}\n")
    lens_l = sorted(len(seqs[q]) for q in list_prots)
    lens_r = sorted(len(seqs[q]) for q in partner.values())
    print(
        f"list proteins {len(list_prots)} from {len(list_genes) - len(missing)} genes; random proteins {len(partner)}"
    )
    print(
        f"median length list {lens_l[len(lens_l) // 2]} random {lens_r[len(lens_r) // 2]}"
    )
    if n_pilot:
        pick = rng.sample(list_prots, n_pilot // 2)
        with (out / "pilot_queries.faa").open("w") as f:
            for q in pick + [partner[q] for q in pick]:
                f.write(f">{q}\n{seqs[q]}\n")


if __name__ == "__main__":
    main()
