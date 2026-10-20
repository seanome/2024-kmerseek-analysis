#!/usr/bin/env python3
"""Notebook 353, Stage 1 tools: run both workflows on real queries, build the blinded
review sheet, and score the pass rule. Stage 1 was not run (no real work-queue queries were
supplied); these are ready for Olga. The rule is in notebooks/353_prereg.md.

Steps, in order:

  1. order   stage1.py order --query-log query_log.tsv --out order.tsv
             Randomises, per query, which workflow the analyst uses first (seed 353).
  2. search  (Sherlock, inside the kmerseek image)
             stage1.py search --query-log query_log.tsv --records records.tsv \
                 --index <frozen setting's collection.db> --alphabet A --k K --out-dir D
             kmerseek with the frozen setting, and the comparison: blastp (-task blastp-short
             under 30 residues) plus exact substring lookup, all against the same
             collection. Writes D/cards.tsv: one evidence card per (query, method, record).
  3. sheet   stage1.py sheet --cards D/cards.tsv --query-log query_log.tsv --out-dir D
             Drops self-records and duplicate evidence, merges a record both workflows found
             into one card, shuffles cards within a query, and writes D/review_sheet.tsv
             (no method column) and D/review_key.tsv (which workflow found each card; do
             not open until scoring is done).
  4. The analyst times each query in each workflow with a stopwatch into
     stopwatch_log.tsv, and the judge fills the sheet's last three columns.
  5. score   stage1.py score --sheet D/review_sheet.tsv --key D/review_key.tsv \
                 --stopwatch stopwatch_log.tsv
             Prints the three pass-rule numbers and PASS or FAIL.

Card limits, fixed before Stage 1 runs: each workflow shows at most CARDS_PER_METHOD cards
per query; kmerseek regions and BLAST hits at E <= REPORT_E; exact substrings always.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

SEED = 353
CARDS_PER_METHOD = 10
REPORT_E = 1.0
BLASTP_SHORT_BELOW = 30
QUERY_LOG_COLUMNS = ["query_number", "received_at", "work_item", "protein_accession", "start", "end", "sequence", "chosen_before_results", "notes"]
STOPWATCH_COLUMNS = ["query_number", "workflow", "used_first", "start_time", "stop_time", "seconds", "analyst", "notes"]
CARD_COLUMNS = ["query_number", "method", "record_id", "query_start", "query_end", "query_segment", "record_segment", "match_line", "identical", "aligned", "evalue", "record_coordinates", "measurement", "reference", "record_accession"]
JUDGE_COLUMNS = ["analyst_had_seen_it (yes/no)", "kept_for_investigation (yes/no)", "one_line_why"]


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def match_line(a: str, b: str) -> tuple[str, int]:
    line = "".join("|" if x == y else " " for x, y in zip(a, b))
    return line, line.count("|")


def card(q, method, rec, qs, qe, rs, re_, evalue):
    qseg, rseg = q["sequence"][qs:qe], rec["sequence"][rs:re_]
    line, n = match_line(qseg, rseg)
    a = int(rec["record_start"])
    return {
        "query_number": q["query_number"],
        "method": method,
        "record_id": rec["record_id"],
        "query_start": qs + 1,
        "query_end": qe,
        "query_segment": qseg,
        "record_segment": rseg,
        "match_line": line,
        "identical": n,
        "aligned": len(qseg),
        "evalue": evalue,
        "record_coordinates": f"{rec['accession']} {a + rs + 1}-{a + re_}",
        "measurement": rec["measurement"],
        "reference": rec["reference"],
        "record_accession": rec["base_accession"],
    }


def cmd_order(args):
    rows = read_tsv(args.query_log)
    rng = random.Random(SEED)
    out = [{"query_number": r["query_number"], "used_first": rng.choice(["kmerseek", "comparison"])} for r in rows]
    write_tsv(args.out, out, ["query_number", "used_first"])


def cmd_search(args):
    queries = read_tsv(args.query_log)
    records = {r["record_id"]: r for r in read_tsv(args.records)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qfa = out_dir / "stage1_queries.fasta"
    qfa.write_text("".join(f">Q{q['query_number']}\n{q['sequence']}\n" for q in queries))
    byq = {f"Q{q['query_number']}": q for q in queries}
    cards = []

    # kmerseek, frozen setting, same flags as Stage 0.
    res = subprocess.run(
        ["kmerseek", "search", "--query", str(qfa), "--target", args.index, "--alphabet", args.alphabet, "--ksize", str(args.k), "--extend-mismatch-penalty", "2", "--extend-xdrop", "8"],
        check=True, capture_output=True, text=True,
    )
    hits: dict[str, list] = {}
    for r in csv.DictReader(res.stdout.splitlines()):
        if r["region_evalue"] and float(r["region_evalue"]) <= REPORT_E:
            hits.setdefault(r["query_name"], []).append(r)
    for qn, rs in hits.items():
        for r in sorted(rs, key=lambda r: float(r["region_evalue"]))[:CARDS_PER_METHOD]:
            cards.append(card(byq[qn], "kmerseek", records[r["target_name"]], int(r["region_start"]), int(r["region_end"]), int(r["target_start"]), int(r["target_end"]), r["region_evalue"]))

    # BLAST: blastp-short under 30 residues, standard blastp otherwise.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "collection"
        subprocess.run(["makeblastdb", "-in", args.records_fasta, "-dbtype", "prot", "-out", str(db), "-parse_seqids"], check=True, capture_output=True)
        for q in queries:
            task = "blastp-short" if len(q["sequence"]) < BLASTP_SHORT_BELOW else "blastp"
            res = subprocess.run(
                ["blastp", "-task", task, "-db", str(db), "-evalue", str(REPORT_E), "-max_target_seqs", str(CARDS_PER_METHOD), "-outfmt", "6 sseqid qstart qend sstart send evalue"],
                input=f">Q{q['query_number']}\n{q['sequence']}\n", check=True, capture_output=True, text=True,
            )
            seen = 0
            for line in res.stdout.splitlines():
                sid, qs, qe, ss, se, ev = line.split("\t")
                if seen >= CARDS_PER_METHOD:
                    break
                seen += 1
                # Ungapped view of the BLAST hit for the card: the query and record segments
                # at the hit's coordinates; the match line counts identical residues by position.
                cards.append(card(q, "comparison:blastp", records[sid], int(qs) - 1, int(qe), int(ss) - 1, int(se), ev))

    # Exact full-alphabet substring: the query inside a record, or a record inside the query.
    for q in queries:
        s = q["sequence"]
        n = 0
        for rec in records.values():
            if n >= CARDS_PER_METHOD:
                break
            i = rec["sequence"].find(s)
            if i >= 0:
                cards.append(card(q, "comparison:substring", rec, 0, len(s), i, i + len(s), "exact"))
                n += 1
                continue
            j = s.find(rec["sequence"])
            if j >= 0:
                cards.append(card(q, "comparison:substring", rec, j, j + len(rec["sequence"]), 0, len(rec["sequence"]), "exact"))
                n += 1
    write_tsv(out_dir / "cards.tsv", cards, CARD_COLUMNS)
    print(f"{len(cards)} cards for {len(queries)} queries")


def cmd_sheet(args):
    cards = read_tsv(args.cards)
    queries = {q["query_number"]: q for q in read_tsv(args.query_log)}
    out_dir = Path(args.out_dir)
    merged: dict[tuple, dict] = {}
    dropped_self = dropped_dup = 0
    for c in cards:
        q = queries[c["query_number"]]
        acc, rng_ = c["record_coordinates"].split()[0].split("-")[0], c["record_coordinates"].split()[1]
        a, b = (int(x) for x in rng_.split("-"))
        if q["protein_accession"] and acc == q["protein_accession"] and q["start"] and int(q["start"]) <= b and a <= int(q["end"]):
            dropped_self += 1
            continue
        # Duplicate evidence: the same measurement on the same residues, whatever the record id.
        key = (c["query_number"], c["record_segment"], c["measurement"])
        if key in merged:
            merged[key]["methods"].add(c["method"].split(":")[0])
            dropped_dup += 1
            continue
        merged[key] = dict(c, methods={c["method"].split(":")[0]})
    rng = random.Random(SEED)
    sheet, keyrows = [], []
    for qn in queries:
        group = [c for k, c in merged.items() if k[0] == qn]
        rng.shuffle(group)
        for i, c in enumerate(group, 1):
            bid = f"{qn}-{i:02d}"
            sheet.append({"blind_id": bid, "query_sequence": queries[qn]["sequence"], **{k: c[k] for k in CARD_COLUMNS if k not in ("method", "evalue", "record_accession")}, **dict.fromkeys(JUDGE_COLUMNS, "")})
            keyrows.append({"blind_id": bid, "found_by": "+".join(sorted(c["methods"]))})
    cols = ["blind_id", "query_sequence"] + [k for k in CARD_COLUMNS if k not in ("method", "evalue", "record_accession")] + JUDGE_COLUMNS
    write_tsv(out_dir / "review_sheet.tsv", sheet, cols)
    write_tsv(out_dir / "review_key.tsv", keyrows, ["blind_id", "found_by"])
    print(f"{len(sheet)} cards on the sheet; dropped {dropped_self} self-records and merged {dropped_dup} duplicates")


def useful(row) -> bool:
    return row["analyst_had_seen_it (yes/no)"].strip().lower() == "no" and row["kept_for_investigation (yes/no)"].strip().lower() == "yes" and row["one_line_why"].strip() != ""


def cmd_score(args):
    sheet = read_tsv(args.sheet)
    key = {r["blind_id"]: r["found_by"].split("+") for r in read_tsv(args.key)}
    times = read_tsv(args.stopwatch)
    qs = sorted({r["blind_id"].rsplit("-", 1)[0] for r in sheet} | {t["query_number"] for t in times})
    yield_ = {"kmerseek": set(), "comparison": set()}
    for r in sheet:
        if useful(r):
            for m in key[r["blind_id"]]:
                yield_[m].add(r["blind_id"].rsplit("-", 1)[0])
    secs = {m: [float(t["seconds"]) for t in times if t["workflow"] == m] for m in ("kmerseek", "comparison")}
    med = {m: statistics.median(v) if v else float("nan") for m, v in secs.items()}
    n_k, n_c = len(yield_["kmerseek"]), len(yield_["comparison"])
    c1 = n_k >= 20
    c2 = med["kmerseek"] <= 0.5 * med["comparison"]
    c3 = n_k >= n_c
    print(f"queries: {len(qs)}")
    print(f"1. queries with a useful measurement from kmerseek: {n_k} (threshold >= 20): {'holds' if c1 else 'fails'}")
    print(f"2. median handling time, kmerseek {med['kmerseek']:.0f} s vs comparison {med['comparison']:.0f} s; needs kmerseek <= 50% of comparison: {'holds' if c2 else 'fails'}")
    print(f"3. useful yield, kmerseek {n_k} vs comparison {n_c}; needs kmerseek >= comparison: {'holds' if c3 else 'fails'}")
    print("PASS" if (c1 and c2 and c3) else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("order")
    p.add_argument("--query-log", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("search")
    p.add_argument("--query-log", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--records-fasta", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--alphabet", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--cards", required=True)
    p.add_argument("--query-log", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("score")
    p.add_argument("--sheet", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--stopwatch", required=True)
    args = ap.parse_args()
    {"order": cmd_order, "search": cmd_search, "sheet": cmd_sheet, "score": cmd_score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
