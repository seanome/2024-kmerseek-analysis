#!/usr/bin/env python3
"""Notebook 283: BHF against copies that keep, or lose, its low-complexity stretches.

Notebook 256 (PR 83) searched BHF and 300 dipeptide-shuffled copies of it against the
19_732 GENCODE v49 canonical human proteins at the 152 kmerseek settings of notebook 241.
Away from the E-value, BHF beat 95% of its copies in more settings than chance, and 100 of
the 153 winning (score, setting) pairs took their BHF region from 63-105, a lysine-rich
stretch. This script runs the control PR 83 named: copies that keep BHF's low-complexity
stretches in one piece, and BHF and copies with the stretches removed.

The stretches are the union of every rule that calls part of BHF low-complexity (choice
made with Olga, 2026-10-13):
  - kmerseek's own mask (982a055), at any of the 152 settings: 69-87, 80-98, 108-117,
    208-216 (bhf_low_complexity_utils.py; it finds a stretch at only 9 settings);
  - NCBI SEG (segmasker, BLAST+ 2.16.0, default 12/2.2/2.5): 70-93, 109-116;
  - the region notebook 256 found behind the wins: 63-105.
Merged, 1-based inclusive: 63-105, 108-117, 208-216.

Three sets of copies, 300 each:
  null 1  the 256 copies: a dipeptide shuffle of all of BHF (not searched again; read
          from 256's run, after the reproduction check below);
  null 2  stretches kept: each stretch verbatim at its own position, each segment
          between stretches dipeptide-shuffled on its own (`kept`);
  null 3  stretches removed: the stretches cut out of BHF, and the remaining 190 residues
          dipeptide-shuffled; the query called BHF in this run is BHF without its
          stretches (`removed`).

Steps (each resumes where it stopped):
  indexes      rebuild the 152 human indexes with notebook 241's own driver (same
               binary, FASTA, flags, Karlin-Altschul fit and retry), then check that
               plan.json is byte-identical to 241's and that the settings without a fit
               are the ones 256 recorded.
  reproduce    search 256's own 301 queries on 6 settings and require the reduced tables
               to equal 256's stored ones (every best value to 1e-9 relative; 256's own
               rerun differed by up to 1.5e-12).
  search kept|removed   the full run of null 2 or null 3.

Paths come from the environment so the same script runs on Sherlock and the laptop:
  NB283_DIR        output folder (default $SCRATCH/283-bhf-low-complexity)
  NB241_KMERSEEK   kmerseek 982a055 binary
  NB241_HUMAN      gencode.v49.pc_translations.canonical.fa
  NB241_KAPPA      the kappa table 241 read (default: notebooks/241_kappa_by_alphabet.tsv,
                   the dark-set table at c2eaaaaf, before funcgroups8 got a row)
  NB256_RUN        256's null_bhf_dipeptide folder (queries.fa, arms.json, top/)

Usage: 283_bhf_low_complexity_control.py indexes|reproduce|search [--null kept|removed]
                                         [--workers 3] [--n 300] [--seed 283]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bhf_low_complexity_utils as lc  # noqa: E402

ENV = os.environ
OUT = Path(
    ENV.get("NB283_DIR", Path(ENV.get("SCRATCH", "/tmp")) / "283-bhf-low-complexity")
)
KMERSEEK = Path(ENV.get("NB241_KMERSEEK", str(lc.KMERSEEK)))
HUMAN = Path(
    ENV.get(
        "NB241_HUMAN",
        "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
KAPPA = Path(ENV.get("NB241_KAPPA", HERE / "241_kappa_by_alphabet.tsv"))
RUN256 = Path(
    ENV.get(
        "NB256_RUN",
        "/Users/olga/data/botryllus/alphabet-ranking-three-cases/null_bhf_dipeptide",
    )
)
IDX_RUN = (
    OUT / "indexes"
)  # the 241 driver's output folder: idx/, search/, logs/, plan.json

# md5 of notebook 241's plan.json (152 settings), as listed in PR 83's inputs.
PLAN_MD5 = "9a77724514ae4e6833e773cd4b0da806"
# md5 of the kappa table 241 read: the dark-set kappa_by_alphabet.tsv as of commit c2eaaaaf
# (2026-09-21). The table gained a funcgroups8 row on 2026-10-02 (0b7a8525), which changes
# 8 funcgroups8 settings from the fallback penalty 2 / X-drop 8 to 0.29 / 1.18.
KAPPA_MD5 = "4c866e34919ea0590005ba75cf39860a"
# md5 of the three-protein queries.fa 241 searched (PR 83's inputs).
QUERIES_MD5 = "fc9ac30a7773b921c086c1684d4b631f"

# segmasker -in bhf.fa -outfmt interval (BLAST+ 2.16.0, default window 12, locut 2.2,
# hicut 2.5) prints 0-based inclusive "69 - 92" and "108 - 115".
SEG_STRETCHES = [(69, 93), (108, 116)]  # 0-based, end-exclusive
REGION_256 = [(62, 105)]  # BHF 63-105, notebook 256 section 3
EXPECTED_UNION = [(62, 105), (107, 117), (207, 216)]  # 1-based 63-105, 108-117, 208-216

# Settings for the reproduction check: one per alphabet family, a setting without a fit,
# the two-letter alphabet of 256's NFE2L1 wins, and gbmr4 k18, the one setting where a
# 256 win and a kmerseek-masked stretch coincide.
REPRODUCE = [
    ("hp_thomas_dill2", 17),
    ("gbmr4", 18),
    ("protein20", 6),
    ("polarity4", 13),
    ("uniprot18", 8),
    ("dayhoff6", 7),
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_fasta(p: Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            if name:
                out.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    out.append((name, "".join(buf)))
    return out


def bhf() -> str:
    q = dict(read_fasta(RUN256 / "queries.fa"))
    return q["BHF"]


def arms256() -> list[dict]:
    return json.loads((RUN256 / "arms.json").read_text())


def stretch_union(seq: str) -> list[tuple[int, int]]:
    """Merged union of kmerseek's mask over all 152 settings, SEG, and BHF 63-105."""
    iv = list(SEG_STRETCHES) + list(REGION_256)
    for a in arms256():
        iv += lc.stretches(seq, a["alphabet"], a["k"])
    merged: list[list[int]] = []
    for a, b in sorted(iv):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = [(a, b) for a, b in merged]
    assert out == EXPECTED_UNION, out
    return out


def mod256():
    """Notebook 256's search module, pointed at this run's binary, indexes and folders."""
    m = _load("nb256", HERE / "256_bhf_dipeptide_shuffle_null.py")
    m.KMERSEEK = KMERSEEK
    m.D = IDX_RUN
    # Keep the coordinates of each query's ten best human proteins (regions/), so the
    # notebook can print the aligned residues of a top hit.
    m.KEEP_REGIONS = True
    return m


def make_queries(null: str, n: int, seed: int) -> list[tuple[str, str]]:
    m = mod256()
    seq = bhf()
    st = stretch_union(seq)
    rng = random.Random(seed)
    if null == "kept":
        ref = seq
        copies = [
            lc.stretches_kept(seq, st, rng, m.dipeptide_shuffle) for _ in range(n)
        ]
        for c in copies:
            assert Counter(zip(c, c[1:])) == Counter(zip(seq, seq[1:]))
    elif null == "removed":
        ref = lc.stretches_removed(seq, st)
        copies = [m.dipeptide_shuffle(ref, rng) for _ in range(n)]
        for c in copies:
            assert (
                Counter(zip(c, c[1:])) == Counter(zip(ref, ref[1:]))
                and c[0] == ref[0]
                and c[-1] == ref[-1]
            )
    else:
        raise ValueError(null)
    qs = [("BHF", ref)] + [(f"{null}{i:03d}", c) for i, c in enumerate(copies)]
    assert len({s for _, s in qs}) == len(qs), "two queries are identical"
    return qs


def run_chunks(
    out: Path, qs: list[tuple[str, str]], arms: list[dict], workers: int
) -> int:
    """Search every chunk of every setting with 256's one_chunk; return the failures."""
    m = mod256()
    from alphabet_ensemble_utils import chunk_tags

    m.OUT = out
    for d in ("top", "tmp", "logs", "regions"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "queries.fa").write_text("".join(f">{h}\n{s}\n" for h, s in qs))
    (out / "arms.json").write_text(json.dumps(arms, indent=1))
    jobs = []
    for arm in arms:
        tags = chunk_tags("BHF", arm["alphabet"], arm["k"], arm["bits"], len(qs))
        size = 25 if len(tags) > 1 else len(qs)
        jobs += [(t, qs[i * size : (i + 1) * size], arm) for i, t in enumerate(tags)]
    print(
        f"{out.name}: {len(qs)} queries, {len(jobs)} chunks over {len(arms)} settings",
        file=sys.stderr,
        flush=True,
    )
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(lambda j: m.one_chunk(*j), jobs):
            failed += "FAILED" in msg
            print(msg, file=sys.stderr, flush=True)
    print(
        f"{out.name}: {len(jobs) - failed} of {len(jobs)} chunks ok",
        file=sys.stderr,
        flush=True,
    )
    return failed


def step_indexes(workers: int) -> None:
    drv = _load("nb241", HERE / "241_alphabet_ranking_driver.py")
    IDX_RUN.mkdir(parents=True, exist_ok=True)
    for d in ("idx", "search", "ka_survival", "pair", "logs"):
        (IDX_RUN / d).mkdir(exist_ok=True)
    q = IDX_RUN / "queries.fa"
    if not q.exists():
        q.write_text((RUN256.parent / "queries.fa").read_text())
    assert md5(q) == QUERIES_MD5, f"{q} is not the queries.fa 241 searched"
    assert (
        md5(KAPPA) == KAPPA_MD5
    ), f"{KAPPA} is not the kappa table 241 read (md5 {md5(KAPPA)})"
    drv.KMERSEEK, drv.HUMAN, drv.OUT, drv.QUERIES, drv.KAPPA_TSV = (
        KMERSEEK,
        HUMAN,
        IDX_RUN,
        q,
        KAPPA,
    )
    # Check the settings before spending an hour on indexes.
    sys.argv = ["241_alphabet_ranking_driver.py", "--plan-only"]
    drv.main()
    got = md5(IDX_RUN / "plan.json")
    assert (
        got == PLAN_MD5
    ), f"plan.json md5 {got}, 241's was {PLAN_MD5}: the settings differ"
    # 241's driver skips any index folder that has a CURRENT file, and RocksDB writes that
    # file seconds into a build. An index whose build was killed would be reused as if it
    # were whole. runs.jsonl gets a line only when a setting has finished, so delete every
    # index folder without one before resuming.
    done = set()
    if (IDX_RUN / "runs.jsonl").exists():
        for line in open(IDX_RUN / "runs.jsonl"):
            r = json.loads(line)
            if r.get("index_rc") == 0:
                done.add(f"human.{r['tag']}.rocksdb")
    for d in (IDX_RUN / "idx").glob("*.rocksdb"):
        if d.name not in done:
            print(f"removing unfinished index {d.name}", file=sys.stderr)
            shutil.rmtree(d)
            tag = d.name.removeprefix("human.").removesuffix(".rocksdb")
            # Its fit and search files may be from the killed run too; redo them with it.
            for f in [
                IDX_RUN / "ka_survival" / f"{tag}.csv",
                IDX_RUN / "ka_survival" / f"{tag}.retry.csv",
                IDX_RUN / "search" / f"{tag}.csv",
                IDX_RUN / "search" / f"{tag}.nofit",
                *(IDX_RUN / "pair").glob(f"{tag}.*.json"),
            ]:
                f.unlink(missing_ok=True)
    sys.argv = ["241_alphabet_ranking_driver.py", "--workers", str(workers)]
    drv.main()
    nofit_now = {
        p.name.removesuffix(".nofit") for p in (IDX_RUN / "search").glob("*.nofit")
    }
    nofit_256 = {f"{a['alphabet']}.k{a['k']}" for a in arms256() if a["nofit"]}
    finished = [json.loads(line) for line in open(IDX_RUN / "runs.jsonl")]
    built = {
        r["tag"]
        for r in finished
        if r.get("index_rc") == 0
        and (IDX_RUN / "idx" / f"human.{r['tag']}.rocksdb").exists()
    }
    print(
        f"indexes built: {len(built)} of 152; without a fit: {len(nofit_now)} now, {len(nofit_256)} in 256",
        file=sys.stderr,
    )
    assert len(built) == 152, sorted(
        set(f"{a['alphabet']}.k{a['k']}" for a in arms256()) - built
    )
    assert nofit_now == nofit_256, {
        "only now": sorted(nofit_now - nofit_256),
        "only 256": sorted(nofit_256 - nofit_now),
    }


def step_reproduce(workers: int) -> None:
    import polars as pl

    arms = [a for a in arms256() if (a["alphabet"], a["k"]) in REPRODUCE]
    assert len(arms) == len(REPRODUCE), arms
    qs = read_fasta(RUN256 / "queries.fa")
    out = OUT / "reproduce_null1"
    assert run_chunks(out, qs, arms, workers) == 0
    # The E-value and bit score carry rounding noise of about 1e-14 between runs, even two
    # runs on the same machine. Paralogs with near-equal E-values (FRG2/FRG2B/FRG2C,
    # AMY1A/1B/1C/AMY2A/2B) then swap places, change how many proteins count as tied at
    # the top, and move in and out of 10th place. The stored tables hold no per-protein
    # values, so such a swap cannot be told from a real change at 10th place. The gate
    # therefore checks what notebook 283 uses: every score's best value to 1e-9 relative,
    # every other column of the other 9 scores exactly, and, for these two, that the top
    # protein is one of 256's top 10. Top-10 and tie-count differences for these two are
    # counted and reported, not failed. The Poisson score is not exact either: the 283
    # rebuild on 2026-10-13 gave 8 of its best values 1.8e-15 below 256's (the last binary
    # digit of a number near 14), with the same top protein and every other column equal.
    from alphabet_ensemble_utils import chunk_tags

    expected = {
        f"{t}.parquet"
        for a in arms
        for t in chunk_tags("BHF", a["alphabet"], a["k"], a["bits"], len(qs))
    }
    found = {p.name for p in (out / "top").glob("*.parquet")}
    noisy = ["E-value", "bit score"]

    def close(a, b) -> bool:
        if a is None or b is None:
            return a is b
        return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))

    n_rows, n_swaps, n_tie_diff, n_rounding, bad = 0, 0, 0, 0, []
    if found != expected:
        bad.append(
            (
                "tables",
                "missing",
                sorted(expected - found),
                "extra",
                sorted(found - expected),
            )
        )
    for name in sorted(expected & found):
        key = ["query_name", "metric"]
        old = pl.read_parquet(RUN256 / "top" / name).sort(key)
        cur = pl.read_parquet(out / "top" / name).sort(key)
        n_rows += cur.height
        if old.select(key).to_dicts() != cur.select(key).to_dicts():
            bad.append((name, "different rows", old.height, cur.height))
            continue
        for metric_is_noisy in (False, True):
            sel = (
                pl.col("metric").is_in(noisy)
                if metric_is_noisy
                else ~pl.col("metric").is_in(noisy)
            )
            o, c = old.filter(sel), cur.filter(sel)
            cols = (
                ["n_hit"]
                if metric_is_noisy
                else [
                    "top_gene",
                    "n_tied_top",
                    "n_hit",
                    "top10",
                    "talk_best_rank",
                ]
            )
            for col in cols:
                if o[col].to_list() != c[col].to_list():
                    bad.append((name, "noisy" if metric_is_noisy else "exact", col))
            for q, m, a, b in zip(
                o["query_name"].to_list(),
                o["metric"].to_list(),
                o["top_value"].to_list(),
                c["top_value"].to_list(),
            ):
                if not close(a, b):
                    bad.append((name, q, m, "top_value", a, b))
                elif a != b and not metric_is_noisy:
                    n_rounding += 1
            if metric_is_noisy:
                for g, t10_old, t10_new in zip(
                    c["top_gene"].to_list(), o["top10"].to_list(), c["top10"].to_list()
                ):
                    if g not in t10_old:
                        bad.append((name, "top gene not in 256's top 10", g, t10_old))
                    n_swaps += t10_old != t10_new
                n_tie_diff += sum(
                    a != b
                    for a, b in zip(
                        o["n_tied_top"].to_list(), c["n_tied_top"].to_list()
                    )
                )
    print(
        f"reproduce: {n_rows} rows compared with 256's; E-value or bit score rows whose top 10 differ: "
        f"{n_swaps}, whose tie count differs: {n_tie_diff}; other scores' best values within "
        f"1e-9 but not equal: {n_rounding}; real differences: {len(bad)}",
        file=sys.stderr,
    )
    for b in bad[:20]:
        print("  ", b, file=sys.stderr)
    (out / "reproduce.json").write_text(
        json.dumps(
            {
                "n_rows": n_rows,
                "n_top10_differ_evalue_bits": n_swaps,
                "n_tied_differ_evalue_bits": n_tie_diff,
                "n_best_value_rounding_other_scores": n_rounding,
                "differ": bad,
            },
            indent=1,
            default=str,
        )
    )
    if bad:
        sys.exit(
            "the rebuilt indexes do not reproduce notebook 256; stop before the new nulls"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["indexes", "reproduce", "search"])
    ap.add_argument("--null", choices=["kept", "removed"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=283)
    args = ap.parse_args()
    if args.step == "indexes":
        step_indexes(args.workers)
    elif args.step == "reproduce":
        step_reproduce(args.workers)
    else:
        assert args.null, "--null kept|removed"
        out = OUT / f"null_{args.null}"
        want = {"null": args.null, "n": args.n, "seed": args.seed}
        manifest = out / "run.json"
        if manifest.exists() and json.loads(manifest.read_text()) != want:
            sys.exit(f"{out} holds a run with {manifest.read_text().strip()}")
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(want))
        qs = make_queries(args.null, args.n, args.seed)
        sys.exit(1 if run_chunks(out, qs, arms256(), args.workers) else 0)


if __name__ == "__main__":
    main()
