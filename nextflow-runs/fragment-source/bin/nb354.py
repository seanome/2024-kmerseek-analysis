#!/usr/bin/env python3
"""Notebook 354 driver: run both lookup workflows on the frozen calls and score them.

  nb354.py dev     --frozen F --work W --kmerseek BIN [--workers N]
  nb354.py choose  --work W
  nb354.py test    --frozen F --work W --kmerseek BIN
  nb354.py summarise --frozen F --work W

`dev` runs every kmerseek setting of the grid and workflow C on the development calls and
their composition shuffles only. `choose` sets each method's confidence threshold and picks
the settings that go to the test split, writing dev_choice.json. `test` reads only that file
for its settings. It times each workflow end to end on one CPU, then scores the source-present
calls against the full database, the source-absent calls against the database without their
sources, and the shuffled controls. `summarise` applies the pass rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lookup

# Fixed before any run (notebook 354, first cell). Do not change after results exist.
MASSSPEC = "massspec18"
COARSER = ["hsdm17", "sdm12"]  # coarser alphabets that keep I = L; at most one goes to test
KSIZES = [4, 5, 6, 7, 8, 9]
FALSE_RATE_TARGET = 0.01  # threshold: at most 1% of shuffled development calls assigned
MATCHED_RECOVERY_PP = 0.02  # rule 4b: "matched recovery" = within 2 points of workflow C
RULE1_RECOVERY = 0.95
RULE2_FALSE = 0.01
RULE3_MEDIAN_LIST = 5
RULE4A_PP = 0.10
RULE4B_SPEEDUP = 5.0
COMPARISON = "comparison"


def setting_name(alphabet: str, k: int) -> str:
    return f"{alphabet}_k{k}"


def read_calls(frozen: Path) -> list[dict]:
    with open(frozen / "calls.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["accepted"] = set(r["accepted_in_db"].split(";"))
        r["source_absent"] = r["source_absent"] == "True"
    return rows


def write_scores(rows: list[tuple[str, str, int]], path: Path) -> None:
    with open(path, "w") as fh:
        fh.write("query\ttarget\tscore\n")
        for q, t, s in sorted(rows):
            fh.write(f"{q}\t{t}\t{s}\n")


def read_scores(path: Path) -> dict[str, list[tuple[int, str]]]:
    """{query: [(score, target)] sorted best first; ties by accession}."""
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path) as fh:
        next(fh)
        for line in fh:
            q, t, s = line.rstrip("\n").split("\t")
            out[q].append((int(s), t))
    for q in out:
        out[q].sort(key=lambda x: (-x[0], x[1]))
    return out


# ---------------------------------------------------------------- one workflow run


def run_kmerseek(kmerseek: str, db_fasta: Path, queries: Path, alphabet: str, k: int,
                 workdir: Path, db: dict[str, str], peptides: dict[str, str],
                 keep_index: bool = True, threads: int = 1) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    timer = lookup.Timer()
    index_dir = workdir / "index"
    if index_dir.exists() and not keep_index:
        shutil.rmtree(index_dir)
    if not index_dir.exists():
        timer.run("build", lookup.kmerseek_index, kmerseek, str(db_fasta), str(index_dir), alphabet, k, threads)
    out_csv = workdir / "search.csv"
    timer.run("search", lookup.kmerseek_search, kmerseek, str(queries), str(index_dir), alphabet, k,
              str(out_csv), threads)
    cands = timer.run("search", read_kmerseek_csv, out_csv)
    rows = timer.run("verify", lookup.verify, peptides, cands, db)
    write_scores(rows, workdir / "scores.tsv")
    n_cand = {q: len(v) for q, v in cands.items()}
    (workdir / "n_candidates.json").write_text(json.dumps(n_cand))
    (workdir / "timing.json").write_text(json.dumps(timer.steps, indent=2))
    if not keep_index:
        shutil.rmtree(index_dir, ignore_errors=True)
    return timer.steps


def read_kmerseek_csv(out_csv: Path) -> dict[str, set[str]]:
    with open(out_csv) as fh:
        header = fh.readline().strip().split(",")
    q_col = "query_name" if "query_name" in header else header[0]
    t_col = "match_name" if "match_name" in header else ("target_name" if "target_name" in header else header[1])
    return lookup.kmerseek_candidates(str(out_csv), q_col, t_col)


def run_comparison(db_fasta: Path, queries: Path, workdir: Path, db: dict[str, str],
                   peptides: dict[str, str], threads: int = 1) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    timer = lookup.Timer()
    norm_db = workdir / "db_normalised.fasta"
    norm_q = workdir / "queries_normalised.fasta"

    def build():
        lookup.write_fasta({a: lookup.normalise(s) for a, s in db.items()}, norm_db)
        lookup.makeblastdb(str(norm_db), str(workdir / "blastdb"))
        return lookup.build_exact_index(db)

    index = timer.run("build", build)

    def search():
        exact = lookup.exact_lookup(peptides, db, index)
        lookup.write_fasta({q: lookup.normalise(p) for q, p in peptides.items()}, norm_q)
        lookup.blastp_short(str(norm_q), str(workdir / "blastdb"), str(workdir / "blast.tsv"), threads)
        blast = lookup.blast_candidates(str(workdir / "blast.tsv"))
        cands = defaultdict(set)
        for d in (exact, blast):
            for q, accs in d.items():
                cands[q] |= accs
        return cands, exact

    cands, exact = timer.run("search", search)
    rows = timer.run("verify", lookup.verify, peptides, cands, db)
    write_scores(rows, workdir / "scores.tsv")
    (workdir / "n_candidates.json").write_text(json.dumps({q: len(v) for q, v in cands.items()}))
    (workdir / "exact_hits.json").write_text(json.dumps({q: sorted(v) for q, v in exact.items()}))
    (workdir / "timing.json").write_text(json.dumps(timer.steps, indent=2))
    for p in workdir.glob("blastdb.*"):
        p.unlink()
    return timer.steps


# ---------------------------------------------------------------- scoring


def top_score(ranked: list[tuple[int, str]]) -> int:
    return ranked[0][0] if ranked else 0


def threshold_from_shuffles(scores: dict, shuffle_ids: list[str]) -> int:
    """Smallest integer t with at most FALSE_RATE_TARGET of shuffles scoring >= t."""
    tops = sorted((top_score(scores.get(q, [])) for q in shuffle_ids), reverse=True)
    allowed = int(FALSE_RATE_TARGET * len(tops))  # this many may reach t
    # t must exceed the (allowed+1)-th highest top score.
    return tops[allowed] + 1 if allowed < len(tops) else 1


def recovered(ranked: list[tuple[int, str]], accepted: set[str], tau: int) -> bool:
    passing = [t for s, t in ranked if s >= tau]
    return any(t in accepted for t in passing[:lookup.TOP_N])


def list_length(ranked: list[tuple[int, str]], tau: int) -> int:
    return sum(s >= tau for s, _ in ranked)


def recovery_curve(scores_present: dict, scores_absent: dict, present: list[dict],
                   absent: list[dict]) -> list[dict]:
    """Recovery and false-assignment rate at every threshold that changes either."""
    taus = {1}
    for d in (scores_present, scores_absent):
        for ranked in d.values():
            taus |= {s for s, _ in ranked}
            taus |= {s + 1 for s, _ in ranked}
    out = []
    for tau in sorted(taus):
        rec = sum(recovered(scores_present.get(c["call_id"], []), c["accepted"], tau) for c in present)
        fa = sum(top_score(scores_absent.get(c["call_id"], [])) >= tau for c in absent)
        out.append({"tau": tau, "recovery": rec / len(present), "false_assignment": fa / len(absent)})
    return out


# ---------------------------------------------------------------- subcommands


def _dev_one(args_tuple):
    kmerseek, frozen, work, alphabet, k = args_tuple
    db = lookup.read_fasta(frozen / "db_full.fasta")
    peptides = {**lookup.read_fasta(frozen / "queries_dev.fasta"),
                **lookup.read_fasta(frozen / "queries_dev_shuffled.fasta")}
    wd = work / "dev" / setting_name(alphabet, k)
    q = wd / "queries.fasta"
    wd.mkdir(parents=True, exist_ok=True)
    lookup.write_fasta(peptides, q)
    if (wd / "scores.tsv").exists():
        return setting_name(alphabet, k), "done before"
    run_kmerseek(kmerseek, frozen / "db_full.fasta", q, alphabet, k, wd, db, peptides, keep_index=False)
    return setting_name(alphabet, k), "ok"


def cmd_dev(args) -> None:
    frozen, work = args.frozen, args.work
    jobs = [(args.kmerseek, frozen, work, a, k) for a in [MASSSPEC] + COARSER for k in KSIZES]
    with ProcessPoolExecutor(args.workers) as ex:
        for name, status in ex.map(_dev_one, jobs):
            print(name, status, flush=True)
    wd = work / "dev" / COMPARISON
    if not (wd / "scores.tsv").exists():
        db = lookup.read_fasta(frozen / "db_full.fasta")
        peptides = {**lookup.read_fasta(frozen / "queries_dev.fasta"),
                    **lookup.read_fasta(frozen / "queries_dev_shuffled.fasta")}
        wd.mkdir(parents=True, exist_ok=True)
        lookup.write_fasta(peptides, wd / "queries.fasta")
        run_comparison(frozen / "db_full.fasta", wd / "queries.fasta", wd, db, peptides, args.workers)
    print("comparison ok")


def cmd_choose(args) -> None:
    frozen, work = args.frozen, args.work
    calls = [c for c in read_calls(frozen) if c["split"] == "dev"]
    shuffle_ids = list(lookup.read_fasta(frozen / "queries_dev_shuffled.fasta"))
    table = []
    for name in [setting_name(a, k) for a in [MASSSPEC] + COARSER for k in KSIZES] + [COMPARISON]:
        scores = read_scores(work / "dev" / name / "scores.tsv")
        tau = threshold_from_shuffles(scores, shuffle_ids)
        rec = sum(recovered(scores.get(c["call_id"], []), c["accepted"], tau) for c in calls) / len(calls)
        lengths = [list_length(scores.get(c["call_id"], []), tau) for c in calls]
        alphabet, _, k = name.rpartition("_k") if name != COMPARISON else (COMPARISON, "", "0")
        table.append({"setting": name, "alphabet": alphabet, "k": int(k), "tau": tau,
                      "dev_recovery": rec, "dev_median_list_length": statistics.median(lengths),
                      "dev_shuffle_assigned": sum(top_score(scores.get(q, [])) >= tau for q in shuffle_ids) / len(shuffle_ids)})

    def best(alphabets):
        rows = [r for r in table if r["alphabet"] in alphabets]
        return max(rows, key=lambda r: (r["dev_recovery"], r["k"]))

    primary = best([MASSSPEC])
    coarse = best(COARSER)
    frozen_settings = [primary] + ([coarse] if coarse["dev_recovery"] > primary["dev_recovery"] else [])
    choice = {
        "kmerseek_settings": [{"alphabet": r["alphabet"], "k": r["k"], "tau": r["tau"]} for r in frozen_settings],
        "comparison_tau": next(r["tau"] for r in table if r["setting"] == COMPARISON),
        "best_coarser_on_dev": {"alphabet": coarse["alphabet"], "k": coarse["k"], "dev_recovery": coarse["dev_recovery"]},
        "rule": "massspec18: k with the highest dev recovery, ties to the larger k. Coarser "
                "alphabet: best (alphabet, k) of hsdm17 and sdm12, sent to test only if its dev "
                "recovery beats massspec18's.",
    }
    (work / "dev_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    with open(work / "dev_table.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(json.dumps(choice, indent=2))


def cmd_test(args) -> None:
    frozen, work = args.frozen, args.work
    choice = json.loads((work / "dev_choice.json").read_text())
    db_full = lookup.read_fasta(frozen / "db_full.fasta")
    db_absent = lookup.read_fasta(frozen / "db_source_absent.fasta")
    present = lookup.read_fasta(frozen / "queries_test_present.fasta")
    absent = lookup.read_fasta(frozen / "queries_test_absent.fasta")
    controls = lookup.read_fasta(frozen / "queries_test_controls.fasta")
    all_test = {**present, **absent}
    methods = [(setting_name(s["alphabet"], s["k"]), s) for s in choice["kmerseek_settings"]] + [(COMPARISON, None)]
    for name, s in methods:
        base = work / "test" / name
        runs = [
            # (run, database file, database, queries): "timed" is the end-to-end runtime:
            # a fresh build on the full database, then all 250 test calls.
            ("timed", frozen / "db_full.fasta", db_full, all_test),
            ("absent", frozen / "db_source_absent.fasta", db_absent, absent),
            ("controls", frozen / "db_full.fasta", db_full, controls),
        ]
        for run, db_file, db, peps in runs:
            wd = base / run
            if (wd / "scores.tsv").exists():
                continue
            wd.mkdir(parents=True, exist_ok=True)
            lookup.write_fasta(peps, wd / "queries.fasta")
            # Only the timed run is held to one thread; the others only need their scores.
            threads = 1 if run == "timed" else args.workers
            if s is None:
                steps = run_comparison(db_file, wd / "queries.fasta", wd, db, peps, threads)
            else:
                steps = run_kmerseek(args.kmerseek, db_file, wd / "queries.fasta", s["alphabet"], s["k"],
                                     wd, db, peps, keep_index=False, threads=threads)
            print(name, run, json.dumps(steps), flush=True)


def cmd_summarise(args) -> None:
    frozen, work = args.frozen, args.work
    choice = json.loads((work / "dev_choice.json").read_text())
    calls = read_calls(frozen)
    present = [c for c in calls if c["split"] == "test" and not c["source_absent"]]
    absent = [c for c in calls if c["split"] == "test" and c["source_absent"]]
    taus = {setting_name(s["alphabet"], s["k"]): s["tau"] for s in choice["kmerseek_settings"]}
    taus[COMPARISON] = choice["comparison_tau"]
    res = work / "results"
    res.mkdir(exist_ok=True)
    metrics, curves, per_call, runtime, lengths_rows = {}, [], [], [], []
    for name, tau in taus.items():
        sp = read_scores(work / "test" / name / "timed" / "scores.tsv")
        sa = read_scores(work / "test" / name / "absent" / "scores.tsv")
        sc = read_scores(work / "test" / name / "controls" / "scores.tsv")
        n_cand_timed = json.loads((work / "test" / name / "timed" / "n_candidates.json").read_text())
        rec = [recovered(sp.get(c["call_id"], []), c["accepted"], tau) for c in present]
        fa = [top_score(sa.get(c["call_id"], [])) >= tau for c in absent]
        ll = [list_length(sp.get(c["call_id"], []), tau) for c in present]
        ctrl_ids = list(lookup.read_fasta(frozen / "queries_test_controls.fasta"))
        ctrl = {kind: [top_score(sc.get(q, [])) >= tau for q in ctrl_ids if f"_{kind}" in q]
                for kind in ("comp", "dipep")}
        timing = json.loads((work / "test" / name / "timed" / "timing.json").read_text())
        metrics[name] = {
            "tau": tau,
            "recovery_top5": sum(rec) / len(rec), "n_present": len(rec), "n_recovered": sum(rec),
            "false_assignment": sum(fa) / len(fa), "n_absent": len(fa), "n_false": sum(fa),
            "median_list_length": statistics.median(ll),
            "median_candidates_before_check": statistics.median(n_cand_timed.get(c["call_id"], 0) for c in present),
            "control_assigned_composition": sum(ctrl["comp"]) / len(ctrl["comp"]),
            "control_assigned_dipeptide": sum(ctrl["dipep"]) / len(ctrl["dipep"]),
            "wall_s": {k: v["wall_s"] for k, v in timing.items()},
            "cpu_s": {k: v["cpu_s"] for k, v in timing.items()},
            "wall_total_s": sum(v["wall_s"] for v in timing.values()),
            "cpu_total_s": sum(v["cpu_s"] for v in timing.values()),
        }
        for row in recovery_curve(sp, sa, present, absent):
            curves.append({"method": name, **row})
        for c, r, l in zip(present, rec, ll):
            ranked = sp.get(c["call_id"], [])
            rank = next((i + 1 for i, (_, t) in enumerate(ranked) if t in c["accepted"]), None)
            per_call.append({"method": name, "call_id": c["call_id"], "case": "present",
                             "recovered": r, "list_length": l,
                             "n_candidates_before_check": n_cand_timed.get(c["call_id"], 0),
                             "top_score": top_score(ranked), "accepted_rank": rank,
                             "accepted_score": next((s for s, t in ranked if t in c["accepted"]), None),
                             "top_target": ranked[0][1] if ranked else "",
                             "n_tied_at_top": sum(s == top_score(ranked) for s, _ in ranked) if ranked else 0})
        for c, f in zip(absent, fa):
            ranked = sa.get(c["call_id"], [])
            per_call.append({"method": name, "call_id": c["call_id"], "case": "absent",
                             "recovered": None, "list_length": list_length(ranked, tau),
                             "n_candidates_before_check": None, "top_score": top_score(ranked),
                             "accepted_rank": None, "accepted_score": None,
                             "top_target": ranked[0][1] if ranked else "",
                             "n_tied_at_top": sum(s == top_score(ranked) for s, _ in ranked) if ranked else 0,
                             "confident": f})
        for step, v in timing.items():
            runtime.append({"method": name, "step": step, **v})
    # Rule 4
    comp_curve = [r for r in curves if r["method"] == COMPARISON]
    verdicts = {}
    for name in taus:
        if name == COMPARISON:
            continue
        m, c = metrics[name], metrics[COMPARISON]
        f_star = m["false_assignment"]
        comp_at = max((r["recovery"] for r in comp_curve if r["false_assignment"] <= f_star), default=0.0)
        rule4a = m["recovery_top5"] - comp_at >= RULE4A_PP
        matched = m["recovery_top5"] >= c["recovery_top5"] - MATCHED_RECOVERY_PP
        speedup = c["wall_total_s"] / m["wall_total_s"]
        rule4b = matched and speedup >= RULE4B_SPEEDUP
        rules = {
            "rule1_recovery": {"value": m["recovery_top5"], "needed": RULE1_RECOVERY, "holds": m["recovery_top5"] >= RULE1_RECOVERY},
            "rule2_false_assignment": {"value": m["false_assignment"], "needed": RULE2_FALSE, "holds": m["false_assignment"] <= RULE2_FALSE},
            "rule3_median_list": {"value": m["median_list_length"], "needed": RULE3_MEDIAN_LIST, "holds": m["median_list_length"] <= RULE3_MEDIAN_LIST},
            "rule4a_recovery_gain": {"value": m["recovery_top5"] - comp_at, "comparison_recovery_at_same_false_rate": comp_at,
                                     "false_rate": f_star, "needed": RULE4A_PP, "holds": rule4a},
            "rule4b_speedup": {"value": speedup, "matched_recovery": matched, "needed": RULE4B_SPEEDUP, "holds": rule4b},
        }
        all_hold = rules["rule1_recovery"]["holds"] and rules["rule2_false_assignment"]["holds"] and \
            rules["rule3_median_list"]["holds"] and (rule4a or rule4b)
        verdicts[name] = {"rules": rules, "pass": all_hold}
    overall = "PASS" if any(v["pass"] for v in verdicts.values()) else "FAIL"
    exact_absent = json.loads((work / "test" / COMPARISON / "absent" / "exact_hits.json").read_text())
    exact_timed = json.loads((work / "test" / COMPARISON / "timed" / "exact_hits.json").read_text())
    diagnostics = {
        # A source-absent call whose whole peptide (I as L, Q as K) still sits in another
        # protein of the reduced database: no sequence method can abstain on it.
        "absent_calls_with_exact_peptide_in_another_protein": sum(bool(exact_absent.get(c["call_id"])) for c in absent),
        "present_calls_with_exact_peptide_in_accepted_source": sum(bool(set(exact_timed.get(c["call_id"], [])) & c["accepted"]) for c in present),
        "present_calls_denovo_equals_db_search": sum(c["denovo_equals_db_search"] == "True" for c in present),
        "n_present": len(present), "n_absent": len(absent),
    }
    (res / "metrics.json").write_text(json.dumps({"verdict": overall, "per_setting": verdicts,
                                                  "metrics": metrics, "diagnostics": diagnostics},
                                                 indent=2, default=str) + "\n")
    for fname, rows in [("curve.tsv", curves), ("per_call.tsv", per_call), ("runtime.tsv", runtime)]:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(res / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    write_examples(frozen, work, taus, present, absent, res)
    print(overall)
    print(json.dumps(verdicts, indent=2, default=str))


def write_examples(frozen: Path, work: Path, taus: dict, present: list[dict], absent: list[dict], res: Path) -> None:
    """Alignments of every peptide with its accepted source and with each method's top
    candidate, for the notebook to print."""
    db_full = lookup.read_fasta(frozen / "db_full.fasta")
    rows = []
    for c in present + absent:
        for acc in sorted(c["accepted"]):
            a = lookup.align(c["peptide"], db_full[acc])
            rows.append({"call_id": c["call_id"], "pair": "accepted source", "method": "", "target": acc, **a})
    for name in taus:
        for run, cases in (("timed", present), ("absent", absent)):
            sc = read_scores(work / "test" / name / run / "scores.tsv")
            for c in cases:
                ranked = sc.get(c["call_id"], [])
                if ranked:
                    acc = ranked[0][1]
                    a = lookup.align(c["peptide"], db_full[acc])
                    rows.append({"call_id": c["call_id"], "pair": "top candidate", "method": name, "target": acc, **a})
    with open(res / "alignments.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dev", "choose", "test", "summarise"):
        p = sub.add_parser(name)
        p.add_argument("--frozen", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        if name in ("dev", "test"):
            p.add_argument("--kmerseek", required=True)
        if name in ("dev", "test"):
            p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"dev": cmd_dev, "choose": cmd_choose, "test": cmd_test, "summarise": cmd_summarise}[args.cmd](args)


if __name__ == "__main__":
    main()
