#!/usr/bin/env python3
"""Notebook 289: the reverse direction. Human BCL-2 family -> C. elegans proteome (is Ced9 found?),
and human CD47 -> B. burgdorferi B31 proteome (is P66 found?).

Notebook 241 searched Ced9 and P66 against the 19_732 GENCODE v49 canonical human proteins
at 152 kmerseek settings; neither partner came near the top. Notebook 285 (PR 128) pooled
those calls over BCL2's human paralog group. This notebook searches the other way, at the
same 152 settings (241's plan.json, committed as notebooks/241_plan.json), with notebook
241's own driver (index flags, Karlin-Altschul fit, fit retry, search flags) and the same
kmerseek build (982a055).

Queries (notebooks/289_queries.fa, written by `prepare`):
  the nine human BCL-2 family members, the Ensembl 115 paralog group notebook 285 found
  BCL2 in (BAK1 BAX BCL2 BCL2A1 BCL2L1 BCL2L10 BCL2L2 BOK MCL1), and CD47 (no paralog),
  each from the GENCODE v49 canonical translations; and 20 dipeptide-shuffled copies of
  each (notebook 256's shuffle, seed 289): <gene>_shuf00 .. <gene>_shuf19.
Targets:
  worm      UniProt reference proteome UP000001940 (C. elegans), canonical FASTA, release
            2026_03: 19_792 proteins, 8_123_346 residues. Holds Ced9 (sp|P41958|CED9_CAEEL),
            whose sequence equals notebook 241's Ced9 query.
  borrelia  UP000001807 (B. burgdorferi B31), release 2026_03: 1_291 proteins. Holds P66
            (tr|H7C7N8|H7C7N8_BORBU, 618 aa); notebook 241's P66 query is its residues 22-618.
  The BCL-2 family and its shuffles are searched against worm; CD47 and its shuffles
  against borrelia.

Rule, written before any search was run:
  1. Per call, as in notebook 285: a region with a finite E-value gives the target protein
     p = 1 - exp(-E * L_target / L_database); E = inf gives p = 1. (The script checks that
     kmerseek's search space is the same for every target, as it was for the human index.)
  2. Single query: a target's score is its lowest p over every call of every setting. The
     partner's rank among all target proteins is reported with its tie count.
  3. Pooled BCL-2 family (the mirror of notebook 285): at each residue of a target protein,
     each of the nine queries contributes its lowest p over its calls covering that target
     residue (p = 1 without one), and Fisher's method combines the nine (chi-squared, 18
     degrees of freedom). The target's score is its lowest combined p over its residues.
  4. Shuffle null: pool i is the i-th shuffle of each of the nine queries, scored the same
     way; for CD47, each shuffle alone.
  Pass, for each case (pooled BCL-2 family -> Ced9; CD47 -> P66): the partner ranks in the
  top 10 of its proteome AND scores better than at least 19 of the 20 shuffles (95%). The
  BCL2-alone rank and each family member's own rank are reported, not tested.
  The score is a ranking score, not a calibrated p-value: each protein's p is its best over
  152 settings and many residues (notebook 285's review, 2026-10-13).

Optional chaining (gapped calls). kmerseek regions are gapless. 982a055's search can chain
extended regions into one call: --chain-max-gap (largest distance on the query between
chained regions) and --chain-max-shift (largest net shift in diagonals, an indel). Both are
0 in notebook 241. NB289_CHAIN="30:10" or "30,10" (notebook 237's values) adds them to every search
and writes to a separate variant folder; the indexes are shared. Default: off.

Steps (each resumes where it stopped):
  prepare               laptop: write notebooks/289_queries.fa and check the two proteomes
  run --species worm|borrelia [--workers 3]
                        per setting: build the index, fit, search (241's one_arm), then keep
                        the regions as regions/<alphabet>.k<k>.parquet and delete the CSV
  summarise --species worm|borrelia
                        small tables in <variant>/summary/ for the notebook

Paths from the environment:
  NB289_DIR       output folder (default $SCRATCH/289-reverse-search)
  NB241_KMERSEEK  kmerseek 982a055 binary
  NB289_WORM, NB289_BORRELIA   the two proteome FASTAs (uncompressed)
  NB289_CHAIN     "gap,shift" to chain regions; unset for gapless
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = os.environ
OUT = Path(
    ENV.get("NB289_DIR", Path(ENV.get("SCRATCH", "/tmp")) / "289-reverse-search")
)
KMERSEEK = Path(
    ENV.get(
        "NB241_KMERSEEK",
        "/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek",
    )
)
# UniProt FASTA with each header cut to its first word (so kmerseek's target_name is exactly
# that word); `prepare` writes them from the downloaded .fasta.gz.
UNIPROT_DIR = Path("/Users/olga/data/uniprot/reference_proteomes")
DOWNLOADED = {
    "worm": UNIPROT_DIR / "UP000001940_6239.fasta.gz",
    "borrelia": UNIPROT_DIR / "UP000001807_224326.fasta.gz",
}
PROTEOMES = {
    "worm": Path(ENV.get("NB289_WORM", UNIPROT_DIR / "UP000001940_6239.first_word.fa")),
    "borrelia": Path(
        ENV.get("NB289_BORRELIA", UNIPROT_DIR / "UP000001807_224326.first_word.fa")
    ),
}
# md5 of the first-word FASTAs `prepare` wrote (UniProt release 2026_03, downloaded 2026-10-13).
PROTEOME_MD5 = {
    "worm": "e4d46f90ab23f2617c506b6e8b8a7740",
    "borrelia": "7bc24ce4bbb49be12e5040143f7b33ac",
}
QUERIES_MD5 = "2a771adf1fe8db4afd1ef7949dcc7510"
PARTNER = {"worm": "sp|P41958|CED9_CAEEL", "borrelia": "tr|H7C7N8|H7C7N8_BORBU"}
FAMILY = ["BAK1", "BAX", "BCL2", "BCL2A1", "BCL2L1", "BCL2L10", "BCL2L2", "BOK", "MCL1"]
QUERIES_OF = {"worm": FAMILY, "borrelia": ["CD47"]}
HUMAN = Path(
    ENV.get(
        "NB241_HUMAN",
        "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
QUERIES = HERE / "289_queries.fa"
PLAN = HERE / "241_plan.json"
PLAN_MD5 = (
    "9a77724514ae4e6833e773cd4b0da806"  # notebook 241's plan.json, as in notebook 283
)
N_SHUFFLES, SEED = 20, 289
TOP_N, SHUFFLES_TO_BEAT = 10, 19
KEEP = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_evalue",
    "region_ka_bits",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_fasta(p: Path) -> dict[str, str]:
    """Name = the header's first word (kmerseek's target_name), sequence."""
    out: dict[str, str] = {}
    name = None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = ""
        elif line:
            out[name] += line
    return out


def chi2_log10_sf_even(x, n: int):
    """log10 of the chi-squared upper tail with 2n degrees of freedom, at x (array).

    With even degrees of freedom the tail is exact: P(X > x) = exp(-x/2) * sum_{k<n} (x/2)^k / k!.
    Written here so Sherlock's venv needs no scipy; equals scipy.stats.chi2.logsf / ln 10.
    """
    import numpy as np

    h = np.asarray(x, dtype=float) / 2
    k = np.arange(n)
    with np.errstate(divide="ignore"):
        terms = (
            k[:, None] * np.log(h)[None, :]
            - np.array([math.lgamma(i + 1) for i in k])[:, None]
        )
    terms[0] = 0.0  # the k = 0 term is 1 (also at x = 0, where 0 * log 0 is nan)
    m = terms.max(axis=0)
    # Rounding can leave +1e-17 near x = 0; a log probability is never above 0.
    return np.minimum(
        (-h + m + np.log(np.exp(terms - m).sum(axis=0))) / math.log(10), 0.0
    )


def variant() -> str:
    c = ENV.get("NB289_CHAIN", "").strip()
    if not c:
        return "gapless"
    gap, shift = (int(x) for x in re.split(r"[,:]", c))
    return f"chain-gap{gap}-shift{shift}"


def chain_args() -> list[str]:
    c = ENV.get("NB289_CHAIN", "").strip()
    if not c:
        return []
    gap, shift = re.split(r"[,:]", c)
    return ["--chain-max-gap", gap, "--chain-max-shift", shift]


def plan(only: list[str] | None = None) -> list[dict]:
    """Notebook 241's 152 settings, or the ones named in `only` ("dayhoff6.k12", ...)."""
    assert md5(PLAN) == PLAN_MD5, f"{PLAN} is not notebook 241's plan.json"
    p = json.loads(PLAN.read_text())
    assert len(p) == 152, len(p)
    if only:
        p = [a for a in p if f"{a['alphabet']}.k{a['ksize']}" in only]
        assert len(p) == len(only), only
    return p


def step_prepare() -> None:
    """Laptop only: the query FASTA (real sequences and shuffles) and the proteome checks."""
    import gzip

    m256 = _load("nb256", HERE / "256_bhf_dipeptide_shuffle_null.py")
    human = read_fasta(HUMAN)
    rng = random.Random(SEED)
    qs: list[tuple[str, str]] = []
    for gene in FAMILY + ["CD47"]:
        # GENCODE header field 7 is the gene symbol; exactly one canonical protein per gene.
        hits = [h for h in human if h.split("|")[6] == gene]
        assert len(hits) == 1, (gene, hits)
        seq = human[hits[0]]
        qs.append((gene, seq))
        for i in range(N_SHUFFLES):
            s = m256.dipeptide_shuffle(seq, rng)
            assert (
                Counter(zip(s, s[1:])) == Counter(zip(seq, seq[1:]))
                and s[0] == seq[0]
                and s[-1] == seq[-1]
            )
            qs.append((f"{gene}_shuf{i:02d}", s))
    assert len({s for _, s in qs}) == len(qs), "two queries are identical"
    for sp, gz in DOWNLOADED.items():
        with gzip.open(gz, "rt") as fin, open(PROTEOMES[sp], "w") as fout:
            for line in fin:
                fout.write(line.split()[0] + "\n" if line.startswith(">") else line)
    QUERIES.write_text("".join(f">{h}\n{s}\n" for h, s in qs))
    print(f"{QUERIES.name}: {len(qs)} queries, md5 {md5(QUERIES)}")
    ced9 = read_fasta(
        Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases/queries.fa")
    )
    for sp, f in PROTEOMES.items():
        seqs = read_fasta(f)
        print(
            f"{sp}: {f.name} {len(seqs):_} proteins, {sum(map(len, seqs.values())):_} residues, md5 {md5(f)}"
        )
        assert PARTNER[sp] in seqs, PARTNER[sp]
    worm = read_fasta(PROTEOMES["worm"])
    assert worm[PARTNER["worm"]] == ced9["Ced9"]
    bor = read_fasta(PROTEOMES["borrelia"])
    assert bor[PARTNER["borrelia"]][21:] == ced9["P66"]


def step_run(species: str, workers: int, only: list[str] | None) -> None:
    import polars as pl

    drv = _load("nb241", HERE / "241_alphabet_ranking_driver.py")
    sp_dir = OUT / species
    vdir = sp_dir / variant()
    vdir.mkdir(parents=True, exist_ok=True)
    # Indexes and their Karlin-Altschul fits are shared by the gapless and chained variants.
    for d in ("idx", "ka_survival"):
        (sp_dir / d).mkdir(parents=True, exist_ok=True)
        if not (vdir / d).exists():
            (vdir / d).symlink_to(sp_dir / d)
    for d in ("search", "logs", "regions"):
        (vdir / d).mkdir(parents=True, exist_ok=True)
    target = PROTEOMES[species]
    assert (
        md5(target) == PROTEOME_MD5[species]
    ), f"{target} is not the proteome prepared on 2026-10-13"
    assert md5(QUERIES) == QUERIES_MD5, f"{QUERIES} changed since it was prepared"
    names = set(QUERIES_OF[species])
    q = vdir / "queries.fa"
    allq = read_fasta(QUERIES)
    keep = {h: s for h, s in allq.items() if h.split("_shuf")[0] in names}
    q.write_text("".join(f">{h}\n{s}\n" for h, s in keep.items()))
    print(
        f"{species}/{variant()}: {len(keep)} queries, target {target}", file=sys.stderr
    )

    # 241's driver, pointed at this proteome and these queries. Its index folders are named
    # human.<alphabet>.k<k>.rocksdb whatever the FASTA; here they sit under <species>/idx.
    drv.KMERSEEK, drv.HUMAN, drv.OUT, drv.QUERIES, drv.PAIRS = (
        KMERSEEK,
        target,
        vdir,
        q,
        {},
    )
    extra = chain_args()
    if extra:
        run0 = drv.run

        def run_with_chain(cmd, log):
            return run0(cmd + extra if cmd[1] == "search" else cmd, log)

        drv.run = run_with_chain

    done_file = vdir / "runs.jsonl"
    done = set()
    if done_file.exists():
        done = {
            json.loads(line)["tag"]
            for line in open(done_file)
            if json.loads(line).get("regions_rows") is not None
        }
    todo = [a for a in plan(only) if f"{a['alphabet']}.k{a['ksize']}" not in done]
    # A build killed part-way leaves a CURRENT file, and the driver would reuse the broken
    # index; remove every index of a setting not yet finished (notebook 283's lesson).
    if not extra:
        for a in todo:
            d = sp_dir / "idx" / f"human.{a['alphabet']}.k{a['ksize']}.rocksdb"
            if d.exists():
                shutil.rmtree(d)
                for f in [
                    vdir / "ka_survival" / f"{a['alphabet']}.k{a['ksize']}.csv",
                    vdir / "ka_survival" / f"{a['alphabet']}.k{a['ksize']}.retry.csv",
                ]:
                    f.unlink(missing_ok=True)

    def one(a: dict) -> dict:
        tag = f"{a['alphabet']}.k{a['ksize']}"
        rec = drv.one_arm(a["alphabet"], a["ksize"], a["penalty"], a["xdrop"], False)
        csv = vdir / "search" / f"{tag}.csv"
        if rec.get("search_rc") != 0 or not csv.exists():
            rec["regions_rows"] = None
            return rec
        out = vdir / "regions" / f"{tag}.parquet"
        if csv.stat().st_size == 0:
            pl.DataFrame(schema={c: pl.String for c in KEEP}).write_parquet(out)
            rec["regions_rows"] = 0
        else:
            (
                pl.scan_csv(csv, infer_schema_length=0)
                .select(KEEP)
                .with_columns(
                    pl.col("query_name").str.strip_chars(),
                    *[pl.col(c).cast(pl.Float64, strict=False) for c in KEEP[2:]],
                    alphabet=pl.lit(a["alphabet"]),
                    ksize=pl.lit(a["ksize"]),
                )
                .sink_parquet(out.with_suffix(".tmp"))
            )
            out.with_suffix(".tmp").rename(out)
            rec["regions_rows"] = pl.scan_parquet(out).select(pl.len()).collect().item()
        csv.unlink()
        return rec

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(one, todo):
            print(json.dumps(rec), file=sys.stderr, flush=True)
            with open(done_file, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    finished = {
        json.loads(line)["tag"]
        for line in open(done_file)
        if json.loads(line).get("regions_rows") is not None
    }
    want = len(plan(only))
    n_done = len(finished & {f"{a['alphabet']}.k{a['ksize']}" for a in plan(only)})
    print(
        f"{species}/{variant()}: {n_done} of {want} settings finished", file=sys.stderr
    )
    if n_done != want:
        sys.exit(
            f"{want - n_done} settings did not finish; rerun the same command to resume"
        )


def step_summarise(species: str, only: list[str] | None) -> None:
    import numpy as np
    import polars as pl

    vdir = OUT / species / variant()
    sdir = vdir / "summary"
    sdir.mkdir(exist_ok=True)
    tseqs = read_fasta(PROTEOMES[species])
    qseqs = read_fasta(vdir / "queries.fa")
    lengths = pl.DataFrame(
        {"target_name": list(tseqs), "L": [len(s) for s in tseqs.values()]}
    )
    l_db = int(lengths["L"].sum())
    n_targets = lengths.height
    partner = PARTNER[species]
    files = sorted((vdir / "regions").glob("*.parquet"))
    assert len(files) == len(plan(only)), (len(files), len(plan(only)))
    calls = (
        pl.scan_parquet(files)
        .filter(pl.col("region_evalue").is_finite())
        .join(lengths.lazy(), on="target_name", how="inner")
        # Two kinds of E-value (kmerseek 982a055, search.rs). A single region's E is for the
        # whole database, K m N exp(-lambda S) with N its residues, so log2(E) + bits is the
        # same for every target and the protein's own share is E * L / N. A chained region's
        # E is its pair-level sum P-value times the number of indexed proteins, and its bits
        # are -log2 P, so log2(E) + bits = log2(number of indexed proteins) and the protein's
        # own chance is E divided by that number. Added 2026-10-21, after the chained run's
        # search-space check failed (s.d. 5.1): the first version used E * L / N for both.
        .with_columns(
            n_index=((pl.col("region_evalue").log(2) + pl.col("region_ka_bits")) * math.log(2)).exp()
        )
        .with_columns(
            chained=pl.col("n_index").is_finite()
            & pl.col("n_index").is_between(0.9 * n_targets, 1.001 * n_targets)
        )
        # kmerseek writes E = 0 when the E-value underflows (notebook 287); floor it so
        # p stays above 0 and the score finite.
        .with_columns(
            x=pl.when(pl.col("chained"))
            .then(pl.col("region_evalue") / pl.col("n_index"))
            .otherwise(pl.col("region_evalue") * pl.col("L") / l_db)
            .clip(lower_bound=1e-300)
        )
        # p = 1 - exp(-x); below 1e-8 that subtraction loses every digit, and p = x there.
        .with_columns(
            p=pl.when(pl.col("x") < 1e-8)
            .then(pl.col("x"))
            .otherwise(1 - (-pl.col("x")).exp())
        )
        .with_columns(neg_ln_p=-pl.col("p").log())
        .collect()
    )
    # The search space must not depend on the target: log2(E) + bits constant per query and setting.
    print(f"chained calls (E from the sum statistic): {int(calls['chained'].sum()):_} of {calls.height:_}")
    space = (
        calls.filter(
            pl.col("region_ka_bits").is_not_null() & (pl.col("region_evalue") > 0) & ~pl.col("chained")
        )
        .with_columns(c=pl.col("region_evalue").log(2) + pl.col("region_ka_bits"))
        .group_by("query_name", "alphabet", "ksize")
        .agg(sd=pl.col("c").std(), n=pl.len())
        .filter(pl.col("n") > 1)
    )
    space.write_csv(sdir / "289_search_space_check.csv")
    print(
        f"search space: largest s.d. of log2(E) + bits within a query and setting: {space['sd'].max()}"
    )

    # Single queries: each target's best p over every call.
    best = (
        calls.group_by("query_name", "target_name")
        .agg(neg_ln_p=pl.col("neg_ln_p").max())
        .with_columns(log10_p=-pl.col("neg_ln_p") / math.log(10))
    )
    rows = []
    for q in qseqs:
        b = best.filter(pl.col("query_name") == q)
        v = b.filter(pl.col("target_name") == partner)["log10_p"]
        pv = float(v[0]) if v.len() else 0.0
        rows.append(
            {
                "query": q,
                "partner_log10_p": pv,
                "partner_rank": int((b["log10_p"] < pv).sum()) + 1,
                "partner_n_tied": (
                    int((b["log10_p"] == pv).sum()) if v.len() else n_targets - b.height
                ),
                "n_targets_with_a_call": b.height,
                "n_targets": n_targets,
            }
        )
    single = pl.DataFrame(rows)
    single.write_csv(sdir / "289_single_query_partner.csv")
    print(single.filter(~pl.col("query").str.contains("_shuf")))
    top = (
        best.filter(~pl.col("query_name").str.contains("_shuf"))
        .sort("query_name", "log10_p", "target_name")
        .group_by("query_name", maintain_order=True)
        .head(10)
    )
    top.write_csv(sdir / "289_single_query_top10.csv")

    # The partner's calls from the real queries, with residues. A gapless call has the same
    # length on both sides; a chained call need not, so each side is cut by its own ends.
    pc = calls.filter(
        (pl.col("target_name") == partner) & ~pl.col("query_name").str.contains("_shuf")
    )
    pc = pc.with_columns(
        query_residues=pl.struct(
            "query_name", "region_start", "region_end"
        ).map_elements(
            lambda r: qseqs[r["query_name"]][
                int(r["region_start"]) : int(r["region_end"])
            ],
            return_dtype=pl.String,
        ),
        target_residues=pl.struct("target_start", "target_end").map_elements(
            lambda r: tseqs[partner][int(r["target_start"]) : int(r["target_end"])],
            return_dtype=pl.String,
        ),
    )
    pc.drop("L", "x", "p").write_csv(sdir / "289_partner_calls.csv")

    if species == "worm":
        # Pooled family: per target residue, Fisher over the nine queries.
        sets = {"real": FAMILY} | {
            f"shuf{i:02d}": [f"{g}_shuf{i:02d}" for g in FAMILY]
            for i in range(N_SHUFFLES)
        }
        pooled_rows, track = [], []
        for name, members in sets.items():
            c = calls.filter(pl.col("query_name").is_in(members))
            e = (
                c.select(
                    "query_name",
                    "target_name",
                    "neg_ln_p",
                    r=pl.int_ranges(
                        pl.col("target_start").cast(pl.Int64),
                        pl.col("target_end").cast(pl.Int64),
                    ),
                )
                .explode("r")
                .group_by("query_name", "target_name", "r")
                .agg(pl.col("neg_ln_p").max())
                .group_by("target_name", "r")
                .agg(s=pl.col("neg_ln_p").sum(), n_q=pl.len())
            )
            lp = chi2_log10_sf_even(2 * e["s"].to_numpy(), len(members))
            e = e.with_columns(log10_p=pl.Series(lp))
            per_t = (
                e.sort("log10_p", "r")
                .group_by("target_name", maintain_order=True)
                .agg(
                    log10_p=pl.col("log10_p").first(),
                    best_residue0=pl.col("r").first(),
                    n_queries_there=pl.col("n_q").first(),
                )
            )
            v = per_t.filter(pl.col("target_name") == partner)
            pv = float(v["log10_p"][0]) if v.height else 0.0
            pooled_rows.append(
                {
                    "pool": name,
                    "partner_log10_p": pv,
                    "partner_best_residue0": (
                        int(v["best_residue0"][0]) if v.height else None
                    ),
                    "partner_n_queries_there": (
                        int(v["n_queries_there"][0]) if v.height else 0
                    ),
                    "partner_rank": int((per_t["log10_p"] < pv).sum()) + 1,
                    "partner_n_tied": (
                        int((per_t["log10_p"] == pv).sum())
                        if v.height
                        else n_targets - per_t.height
                    ),
                    "n_targets": n_targets,
                }
            )
            if name == "real":
                per_t.sort("log10_p", "target_name").head(25).write_csv(
                    sdir / "289_pooled_top25.csv"
                )
                c_p = (
                    c.filter(pl.col("target_name") == partner)
                    .select(
                        "query_name",
                        "neg_ln_p",
                        r=pl.int_ranges(
                            pl.col("target_start").cast(pl.Int64),
                            pl.col("target_end").cast(pl.Int64),
                        ),
                    )
                    .explode("r")
                    .group_by("query_name", "r")
                    .agg(pl.col("neg_ln_p").max())
                )
                c_p.write_csv(sdir / "289_partner_residue_evidence.csv")
            print(pooled_rows[-1], flush=True)
        pooled = pl.DataFrame(pooled_rows)
        pooled.write_csv(sdir / "289_pooled_family_partner.csv")
        real = pooled.filter(pl.col("pool") == "real").row(0, named=True)
        shuf = pooled.filter(pl.col("pool") != "real")["partner_log10_p"].to_numpy()
        verdict = {
            "case": "pooled BCL-2 family -> Ced9",
            "rank": real["partner_rank"],
            "n_targets": n_targets,
            "shuffles_beaten": int((shuf > real["partner_log10_p"]).sum()),
            "pass": real["partner_rank"] <= TOP_N
            and int((shuf > real["partner_log10_p"]).sum()) >= SHUFFLES_TO_BEAT,
        }
    else:
        r = single.filter(pl.col("query") == "CD47").row(0, named=True)
        shuf = single.filter(pl.col("query").str.starts_with("CD47_shuf"))[
            "partner_log10_p"
        ].to_numpy()
        verdict = {
            "case": "CD47 -> P66",
            "rank": r["partner_rank"],
            "n_targets": n_targets,
            "shuffles_beaten": int((shuf > r["partner_log10_p"]).sum()),
            "pass": r["partner_rank"] <= TOP_N
            and int((shuf > r["partner_log10_p"]).sum()) >= SHUFFLES_TO_BEAT,
        }
    (sdir / "289_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "run", "summarise"])
    ap.add_argument("--species", choices=list(PROTEOMES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument(
        "--only",
        default="",
        help="test run: comma-separated settings, e.g. dayhoff6.k12",
    )
    args = ap.parse_args()
    only = [x for x in args.only.split(",") if x] or None
    if args.step == "prepare":
        step_prepare()
    else:
        assert args.species, "--species worm|borrelia"
        assert KMERSEEK.exists() or args.step == "summarise", KMERSEEK
        (
            step_run(args.species, args.workers, only)
            if args.step == "run"
            else step_summarise(args.species, only)
        )


if __name__ == "__main__":
    main()
