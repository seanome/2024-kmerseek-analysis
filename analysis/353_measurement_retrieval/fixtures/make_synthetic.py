#!/usr/bin/env python3
"""SYNTHETIC fixtures for testing the notebook 353 software. Nothing here is data.

Every sequence, query, judgment and timing below is made up by a random number
generator so the pipeline can be run end to end. Every file carries synthetic = yes (or
an id starting with SYN), and `mr353.py score` refuses to give PASS or FAIL on them.

Writes into --work:
  evidence/            8 made-up proteins with 30 made-up measured stretches
  synthetic_queue.csv  160 made-up queries: some are mutated copies of evidence stretches
                       (so searches have something to find), the rest are random
  kmerseek_stub/       CSVs in kmerseek's column layout, made by an exact-match stub on a
                       two-letter encoding. They test the reader, not kmerseek.
with --step dev-labels (after `cards`): dev_labels.csv, made-up usefulness labels;
with --step fill-session (after `session-pack`): session_filled.csv, made-up timings;
with --step fill-review (after `review-pack`): review_filled.csv, made-up judgments.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mr353  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("ACFGHILMVWY")  # a stand-in two-letter split, for the stub only


def rand_seq(rng, n):
    return "".join(rng.choice(AA) for _ in range(n))


def mutate(rng, s, frac):
    s = list(s)
    for i in range(len(s)):
        if rng.random() < frac:
            s[i] = rng.choice(AA)
    return "".join(s)


def make_evidence(work: Path, rng: random.Random):
    prots = {f"SYNP{i:02d}": rand_seq(rng, rng.randint(200, 400)) for i in range(8)}
    rows = []
    for j in range(30):
        acc = rng.choice(sorted(prots))
        seq = prots[acc]
        L = rng.choice([6, 8, 12, 53])
        s0 = rng.randint(0, len(seq) - L)
        rows.append(
            dict(
                evidence_id=f"SYN:E{j:03d}",
                source="SYNTHETIC",
                source_version="synthetic fixture",
                source_record_id=f"E{j:03d}",
                source_url="https://example.invalid/synthetic",
                protein_accession=acc,
                protein_name=f"synthetic protein {acc}",
                organism="none (synthetic)",
                sequence_origin="synthetic",
                protein_md5=mr353.md5(seq),
                uniprot_sequence_version=None,
                sequence_matches_uniprot=None,
                orig_start=s0 + 1,
                orig_end=s0 + L,
                orig_coordinates="1-based, inclusive (synthetic)",
                start0=s0,
                end0=s0 + L,
                stretch=seq[s0 : s0 + L],
                measurement="synthetic measurement",
                outcome=rng.choice(["active", "weak", "inactive"]),
                value=rng.random() * 10,
                value_units="synthetic units",
                z_score=rng.gauss(0, 2),
                n_cells=100.0,
                assay_context="SYNTHETIC: no assay",
                references="none",
                endpoint_note="SYNTHETIC: endpoints are made up.",
            )
        )
    seq_by_md5 = {mr353.md5(s): s for s in prots.values()}
    m = mr353.write_evidence(rows, seq_by_md5, work / "evidence", None)
    import json

    (work / "evidence" / "manifest.json").write_text(
        json.dumps(m, indent=1, default=str)
    )
    return prots, rows


def make_queue(work: Path, rng: random.Random, prots, rows):
    out = work / "synthetic_queue.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(mr353.QUEUE_COLUMNS)
        for i in range(160):
            L = rng.randint(15, 60)
            if i % 3 == 0:
                r = rng.choice(rows)
                seq = prots[r["protein_accession"]]
                mid = (r["start0"] + r["end0"]) // 2
                s0 = max(0, min(len(seq) - L, mid - L // 2))
                q = mutate(rng, seq[s0 : s0 + L], 0.15)
            else:
                q = rand_seq(rng, L)
            w.writerow(
                [
                    f"SYNQ{i:03d}",
                    f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                    q,
                    "",
                    "",
                    "SYNTHETIC",
                    "synthetic query",
                    "yes",
                ]
            )
        # one query copied letter for letter from a stretch: must be excluded
        r = rows[0]
        seq = prots[r["protein_accession"]]
        w.writerow(
            [
                "SYNQ_COPY",
                "2026-01-01T09:00:00",
                seq[r["start0"] : r["start0"] + 20],
                "",
                "",
                "SYNTHETIC",
                "copied",
                "yes",
            ]
        )
    return out


def encode(s):
    return "".join("h" if c in HYDROPHOBIC else "p" for c in s)


def make_kmerseek_stub(work: Path):
    """Exact runs of >= k identical encoded letters between query and target, written in
    kmerseek's v0.4.0 column layout (only the columns mr353 reads carry meaning)."""
    header = (
        open(
            Path(__file__).resolve().parents[1] / "tests" / "kmerseek_v0.4.0_sample.csv"
        )
        .readline()
        .strip()
        .split(",")
    )
    qs = []
    for part in ("development", "evaluation"):
        qs += mr353.read_fasta(work / "queries" / f"{part}.fasta")
    qs += mr353.read_fasta(work / "controls" / "controls.fasta")
    ts = mr353.read_fasta(work / "evidence" / "evidence_proteins.fasta")
    out = work / "kmerseek_stub"
    out.mkdir(exist_ok=True)
    for setting, k in (("hp_lehninger2_k10", 10), ("hp_lehninger2_k12", 12)):
        with open(out / f"{setting}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for qn, q in qs:
                eq = encode(q)
                for tn, t in ts:
                    et = encode(t)
                    seen = set()
                    for i in range(len(eq) - k + 1):
                        j = et.find(eq[i : i + k])
                        while j >= 0:
                            d = j - i
                            if d not in seen:
                                seen.add(d)
                                a, b = i, i + k
                                while (
                                    a > 0
                                    and j - (i - a) > 0
                                    and eq[a - 1] == et[a - 1 + d]
                                ):
                                    a -= 1
                                while (
                                    b < len(eq)
                                    and b + d < len(et)
                                    and eq[b] == et[b + d]
                                ):
                                    b += 1
                                row = {c: "" for c in header}
                                row.update(
                                    query_name=qn,
                                    target_name=tn,
                                    ksize=k,
                                    moltype="STUB",
                                    region_start=a,
                                    region_end=b,
                                    region_subseq=q[a:b],
                                    target_start=a + d,
                                    target_end=b + d,
                                    target_subseq=t[a + d : b + d],
                                    region_ka_bits=b - a,
                                    region_evalue="nan",
                                )
                                w.writerow(row)
                            j = et.find(eq[i : i + k], j + 1)


def dev_labels(work: Path, rng: random.Random):
    """Made-up usefulness labels on development cards, to exercise `tune`."""
    import polars as pl

    cards = pl.read_parquet(work / "hits" / "cards.parquet")
    with open(work / "dev_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "evidence_id", "useful", "reason", "synthetic"])
        for qid, eid in (
            cards.filter(pl.col("split") == "development")
            .select("query_id", "evidence_id")
            .unique()
            .iter_rows()
        ):
            u = rng.random() < 0.5
            w.writerow(
                [qid, eid, "yes" if u else "no", "synthetic reason" if u else "", "yes"]
            )


def fill_review(work: Path, rng: random.Random):
    """Made-up judgments, to exercise `score`."""
    rv = list(csv.DictReader(open(work / "review" / "blind_review.csv")))
    with open(work / "review_filled.csv", "w", newline="") as f:
        w = (
            csv.DictWriter(
                f, fieldnames=list(rv[0].keys() if rv else []) + ["synthetic"]
            )
            if rv
            else None
        )
        if w:
            w.writeheader()
            for r in rv:
                u = rng.random() < 0.5
                r.update(
                    previously_seen_by_analyst="no",
                    useful="yes" if u else "no",
                    reason="synthetic reason" if u else "",
                    reviewer_id="SYN",
                    synthetic="yes",
                )
                w.writerow(r)


def fill_session(work: Path, rng: random.Random):
    """Made-up times, to exercise `score`."""
    ss = list(csv.DictReader(open(work / "review" / "session_order.csv")))
    with open(work / "session_filled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ss[0].keys()) + ["synthetic"])
        w.writeheader()
        for r in ss:
            r.update(
                analyst_id="SYN",
                search_seconds=rng.randint(10, 300),
                reading_seconds=rng.randint(10, 300),
                verification_seconds=rng.randint(0, 200),
                synthetic="yes",
            )
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument(
        "--step",
        choices=["evidence", "stub", "dev-labels", "fill-session", "fill-review"],
        required=True,
    )
    a = p.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(353)
    if a.step == "evidence":
        prots, rows = make_evidence(work, rng)
        make_queue(work, rng, prots, rows)
    elif a.step == "stub":
        make_kmerseek_stub(work)
    elif a.step == "fill-session":
        fill_session(work, rng)
    elif a.step == "dev-labels":
        dev_labels(work, rng)
    else:
        fill_review(work, rng)


if __name__ == "__main__":
    main()
