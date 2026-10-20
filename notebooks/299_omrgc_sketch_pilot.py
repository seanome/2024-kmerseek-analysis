#!/usr/bin/env python3
"""Notebook 299: can hashed reduced-alphabet sketches stand in for DIAMOND when searching a
whole gene catalog for a protein? Pilot on OM-RGC v2 (board idea 12, specified 2026-10-14).

Subcommands, in the order they run:

  queries       (laptop, seconds) 100 random arctic clam proteins, the BHF regions copied
                from PR 80's table, and 20 dipeptide-shuffled copies of each.
  setting I     print "alphabet ksize scaled" for sketch setting I (0-17); used by sbatch.
  step1-table   read the 18 timing logs of the 1 GB slice, extrapolate to the whole catalog,
                and say whether it fits in two compute days at either scaled value.
  plan-step3    print the step 3 sbatch commands with time and memory sized from step 1.
  score         recall of DIAMOND's hits and shuffled-hit rate per setting, verdict.json and
                the figure.
  logan         step 5 arithmetic: what the step 1 rate means for Logan's proteins.

Sherlock paths come from OMRGC_WORK (default $SCRATCH/299-omrgc-pilot). The decision rules
are in notebooks/299_data/rules.md and are copied into the notebook's first cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "299_data"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

sys.path.insert(0, str(ROOT / "nextflow-runs" / "scope40-lowcomplexity-null" / "bin"))
from shuffle_fasta import dipeptide_counts, parse_fasta, shuffle_dipeptide  # noqa: E402

ALPHABETS = ("protein", "dayhoff", "hp")
KSIZES = (7, 10, 13)
# Scaled 10 was added by Olga on 2026-10-20, after step 1 had passed and while step 3 ran for
# 100 and 1000. It goes last so settings 0-17 keep their array indices (running SLURM tasks
# look their setting up by index); scaled 10 is settings 18-26.
SCALEDS = (100, 1_000, 10)
SETTINGS = [(a, k, s) for s in SCALEDS for a in ALPHABETS for k in KSIZES]

QUERY_SEED = 299
N_CLAM_QUERIES = 100
N_SHUFFLES = 20
TRUTH_EVALUE = 1e-5
PASS_RECALL = 0.80
PASS_SHUFFLED_RATE = 0.01
COMPUTE_BUDGET_CPU_HOURS = 1_536.0  # two days of a 32-CPU node (Olga, 2026-10-19)
N_SAMPLE_PARTS = 32  # parts of the 1% random gene sample, sketched as extra array tasks
TIMEOUT_EXIT = 124
# Figure: marker fill encodes scaled (filled = 10, half = 100, open = 1000).
FILL_BY_SCALED = {10: "full", 100: "left", 1_000: "none"}
FILL_LABEL = {
    "full": "filled",
    "left": "half-filled",
    "none": "open",
}  # exit status of `timeout` when it stops a command
FIG1_GENES = (
    "ZNF292",
    "RSF1",
    "TSHZ1",
    "TSHZ2",
    "TSHZ3",
    "RNMT",
    "SFI1",
    "TRAPPC10",
    "NDNF",
)


def setting_id(alphabet: str, ksize: int, scaled: int) -> str:
    """Name used for files and table rows, e.g. "hp.k10.s100"."""
    return f"{alphabet}.k{ksize}.s{scaled}"


def work_dir() -> Path:
    """Sherlock working folder for this pilot."""
    return Path(
        os.environ.get(
            "OMRGC_WORK", Path(os.environ.get("SCRATCH", "/tmp")) / "299-omrgc-pilot"
        )
    )


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    """Write (name, sequence) pairs, 60 residues per line."""
    with open(path, "w") as out:
        for name, seq in records:
            out.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i : i + 60] + "\n")


def md5_of(path: Path) -> str:
    """md5 of a file, for provenance."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- queries


def bhf_regions_from_pr80(hits_json: Path, bhf_seq: str) -> list[dict]:
    """For each Figure 1 gene, its lowest-E BHF region over notebook 241's settings.

    Copied from PR 80's reports/case_gallery/hits241.BHF.json, whose region rows are
    [query_start, query_end, target_start, target_end, E, mean_idf, tfidf, query_residues,
    target_residues] with 0-based, end-exclusive query coordinates. The residues stored in
    the table are checked against BHF[start:end], so a coordinate slip stops the script.
    """
    hits = json.loads(hits_json.read_text())
    best: dict[str, tuple[str, list]] = {}
    for arm, entry in hits.items():
        for row in entry["rows"]:
            if row["gene"] not in FIG1_GENES:
                continue
            for region in row["rg"]:
                if row["gene"] not in best or region[4] < best[row["gene"]][1][4]:
                    best[row["gene"]] = (arm, region)
    out = []
    for gene in FIG1_GENES:
        arm, (q_start, q_end, t_start, t_end, evalue, *_rest, q_res, t_res) = best[gene]
        assert bhf_seq[q_start:q_end] == q_res, (gene, q_start, q_end)
        _, alphabet, ksize = arm.split("|")
        out.append(
            dict(
                query_id=f"BHF_{q_start}_{q_end}_{gene}",
                gene=gene,
                kmerseek_alphabet=alphabet,
                kmerseek_ksize=int(ksize),
                bhf_start_0based=q_start,
                bhf_end_exclusive=q_end,
                target_start_0based=t_start,
                target_end_exclusive=t_end,
                kmerseek_region_evalue=evalue,
                sequence=q_res,
            )
        )
    return out


def cmd_queries(args: argparse.Namespace) -> None:
    """Write the real and shuffled query FASTA files and their table."""
    import polars as pl

    DATA.mkdir(parents=True, exist_ok=True)
    clam = {
        name.split()[0]: seq.rstrip("*") for name, seq in parse_fasta(args.clam_fasta)
    }
    rng = random.Random(QUERY_SEED)
    clam_ids = rng.sample(sorted(clam), N_CLAM_QUERIES)

    bhf_seq = dict(parse_fasta(args.bhf_fasta))["BHF"]
    regions = bhf_regions_from_pr80(args.bhf_hits_json, bhf_seq)
    pl.DataFrame(regions).write_csv(DATA / "bhf_regions_from_pr80.tsv", separator="\t")

    real = [(qid, clam[qid]) for qid in clam_ids] + [
        (r["query_id"], r["sequence"]) for r in regions
    ]
    shuffled, rows = [], []
    for qid, seq in real:
        kind = "clam" if qid in clam else "bhf_region"
        rows.append(
            dict(
                query_id=qid,
                parent_id=qid,
                kind=kind,
                shuffled=False,
                copy=0,
                length=len(seq),
            )
        )
        for copy in range(1, N_SHUFFLES + 1):
            copy_rng = random.Random(f"{QUERY_SEED}:{qid}:{copy}")
            shuf = shuffle_dipeptide(seq, copy_rng)
            assert dipeptide_counts(shuf) == dipeptide_counts(seq), qid
            name = f"{qid}__dishuf{copy:02d}"
            shuffled.append((name, shuf))
            rows.append(
                dict(
                    query_id=name,
                    parent_id=qid,
                    kind=kind,
                    shuffled=True,
                    copy=copy,
                    length=len(shuf),
                )
            )

    write_fasta(real, DATA / "queries_real.faa")
    write_fasta(shuffled, DATA / "queries_shuffled.faa")
    pl.DataFrame(rows).write_csv(DATA / "queries.tsv", separator="\t")
    provenance = dict(
        clam_fasta=str(args.clam_fasta),
        clam_fasta_md5=md5_of(args.clam_fasta),
        clam_proteins=len(clam),
        clam_sample_seed=QUERY_SEED,
        clam_sample_method="random.Random(299).sample(sorted(protein ids), 100)",
        shuffle="dipeptide (Altschul-Erikson), nextflow-runs/scope40-lowcomplexity-null/bin/shuffle_fasta.py, "
        "seed per copy = random.Random(f'299:{query_id}:{copy}')",
        bhf_regions_source="PR 80, reports/case_gallery/hits241.BHF.json (branch olgabot/case-gallery-reference-pairs): "
        "for each Figure 1 gene, its lowest-E region over notebook 241's settings",
        n_real=len(real),
        n_shuffled=len(shuffled),
    )
    (DATA / "queries_provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n"
    )
    print(json.dumps(provenance, indent=1))


# ---------------------------------------------------------------- timing logs

TIME_FIELDS = {
    "user_s": r"User time \(seconds\): ([\d.]+)",
    "sys_s": r"System time \(seconds\): ([\d.]+)",
    "max_rss_kb": r"Maximum resident set size \(kbytes\): (\d+)",
    "exit_status": r"Exit status: (\d+)",
    "wall": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)",
}


def parse_gnu_time(path: Path) -> dict | None:
    """Read a `/usr/bin/time -v` log. None if the file is missing (job killed first)."""
    if not path.exists():
        return None
    text = path.read_text()
    out: dict = {}
    for key, pattern in TIME_FIELDS.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else None
    if out["wall"] is None:
        return None
    parts = [float(p) for p in out["wall"].split(":")]
    wall_s = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    return dict(
        cpu_s=float(out["user_s"]) + float(out["sys_s"]),
        wall_s=wall_s,
        max_rss_gb=int(out["max_rss_kb"]) / 1024**2,
        exit_status=int(out["exit_status"]),
    )


def cmd_drop_empty(args: argparse.Namespace) -> None:
    """Copy the sketches that hold at least one hash from SRC (.sig.gz) into DST (.zip).

    Two things make sourmash's own .zip writer slow here, both measured:
    - Empty sketches: all identical. In step 1 run 1 (jobs 48551687) the nine scaled 1000
      settings and hp k7 scaled 100 hit the 40_000 s limit; on 20_000 catalog genes at
      protein k7 scaled 1000, writing .zip took 132 s and .sig.gz 2.5 s (7_856 of 20_000
      empty). A gene with no hash shares no hash with any query, so it is dropped (checked:
      same 25 manysearch hits with and without them).
    - Identical non-empty sketches under different gene names (hp k7 scaled 100: 9_815
      distinct among 19_914). sourmash names each file by the sketch's md5 and, when the
      name exists with other content, tries name_1, name_2, ... reading each, so a group of
      g identical sketches costs about g^2/2 reads; run 2's tasks for those settings ran
      for hours here. So each sketch is written under its own position number instead,
      with the same manifest sourmash writes.
    """
    from io import StringIO

    import sourmash
    from sourmash import signature as sigmod
    from sourmash.manifest import CollectionManifest
    from sourmash.sbt_storage import ZipStorage

    storage = ZipStorage(str(args.dst), mode="w")
    rows, kept, dropped = [], 0, 0
    for sig in sourmash.load_file_as_signatures(str(args.src)):
        if not len(sig.minhash.hashes):
            dropped += 1
            continue
        buf = sigmod.save_signatures_to_json([sig], compression=1)
        location = storage.save(f"signatures/{kept}.sig.gz", buf, overwrite=True)
        rows.append(
            CollectionManifest.make_manifest_row(sig, location, include_signature=False)
        )
        kept += 1
    manifest_fp = StringIO()
    CollectionManifest(rows).write_to_csv(manifest_fp, write_header=True)
    storage.save(
        "SOURMASH-MANIFEST.csv",
        manifest_fp.getvalue().encode("utf-8"),
        overwrite=True,
        compress=True,
    )
    storage.flush()
    storage.close()
    print(f"{args.src}: kept {kept} sketches, dropped {dropped} empty")


def cmd_indices(args: argparse.Namespace) -> None:
    """Print the SLURM array indices (comma-separated) of the settings at the given scaled values."""
    print(",".join(str(i) for i, (_, _, s) in enumerate(SETTINGS) if s in args.scaled))


def cmd_setting(args: argparse.Namespace) -> None:
    """Print one sketch setting for a SLURM array index."""
    alphabet, ksize, scaled = SETTINGS[args.index]
    print(alphabet, ksize, scaled)


def cmd_step1_table(args: argparse.Namespace) -> None:
    """Timing per setting on the 1 GB slice and the whole-catalog arithmetic."""
    import polars as pl

    work = work_dir()
    stats = json.loads((work / "catalog_stats.json").read_text())
    slice_gb = stats["slice_bytes"] / 1e9
    catalog_gb = stats["catalog_fna_bytes"] / 1e9
    limit_s = stats["step1_time_limit_s"]
    rows = []
    for alphabet, ksize, scaled in SETTINGS:
        sid = setting_id(alphabet, ksize, scaled)
        timing = parse_gnu_time(work / "step1" / f"{sid}.time")
        if timing is None:
            sys.exit(
                f"{sid}: no timing log; the job did not run or was killed before sourmash ended"
            )
        if timing["exit_status"] not in (0, TIMEOUT_EXIT):
            sys.exit(
                f"{sid}: sourmash failed with exit {timing['exit_status']}; read its log, do not score it"
            )
        finished = timing["exit_status"] == 0
        # Writing the .zip index of the non-empty sketches is part of the cost (drop-empty).
        zip_timing = (
            parse_gnu_time(work / "step1" / f"{sid}.zip.time") if finished else None
        )
        if finished and (
            zip_timing is None or zip_timing["exit_status"] not in (0, TIMEOUT_EXIT)
        ):
            sys.exit(
                f"{sid}: no successful .zip index step; read its log, do not score it"
            )
        # Stopped by `timeout` at limit_s: a lower bound, it needs at least limit_s.
        sketch_cpu_s = timing["cpu_s"] if finished else float(limit_s)
        zip_stopped = (
            zip_timing is not None and zip_timing["exit_status"] == TIMEOUT_EXIT
        )
        if zip_timing is None:
            zip_cpu_s = 0.0
        else:
            zip_cpu_s = float(limit_s) if zip_stopped else zip_timing["cpu_s"]
        finished = finished and not zip_stopped
        cpu_s = sketch_cpu_s + zip_cpu_s
        rows.append(
            dict(
                setting=sid,
                alphabet=alphabet,
                ksize=ksize,
                scaled=scaled,
                finished=finished,
                sketch_cpu_s=sketch_cpu_s,
                zip_cpu_s=zip_cpu_s,
                slice_cpu_s=cpu_s,
                slice_wall_s=timing["wall_s"] if timing else None,
                slice_max_rss_gb=timing["max_rss_gb"] if timing else None,
                cpu_s_per_gb=cpu_s / slice_gb,
                catalog_cpu_hours_arithmetic=cpu_s / slice_gb * catalog_gb / 3600,
                is_lower_bound=not finished,
            )
        )
    table = pl.DataFrame(rows)
    per_scaled = (
        table.group_by("scaled")
        .agg(
            pl.col("catalog_cpu_hours_arithmetic")
            .sum()
            .alias("all_nine_cpu_hours_arithmetic"),
            pl.col("is_lower_bound").any().alias("any_lower_bound"),
        )
        .sort("scaled")
        .with_columns(
            (pl.col("all_nine_cpu_hours_arithmetic") <= COMPUTE_BUDGET_CPU_HOURS).alias(
                "fits_two_compute_days"
            )
        )
    )
    # A lower bound that already exceeds the budget is a definite "does not fit"; a lower
    # bound under it is unknown and counts as not fitting, since the rule needs a number.
    per_scaled = per_scaled.with_columns(
        pl.when(pl.col("fits_two_compute_days") & pl.col("any_lower_bound"))
        .then(None)
        .otherwise(pl.col("fits_two_compute_days"))
        .alias("fits_two_compute_days")
    )
    TABLES.mkdir(exist_ok=True)
    table.write_csv(TABLES / "299_step1_sketch_translate_timing_1gb_slice_omrgc_v2.csv")
    per_scaled.write_csv(TABLES / "299_step1_whole_catalog_arithmetic_omrgc_v2.csv")
    fits_any = bool(per_scaled["fits_two_compute_days"].fill_null(False).any())
    step1 = dict(
        slice_gb=slice_gb,
        catalog_fna_gb=catalog_gb,
        catalog_genes=stats["catalog_genes"],
        budget_cpu_hours=COMPUTE_BUDGET_CPU_HOURS,
        per_scaled=per_scaled.to_dicts(),
        fits_at_some_scaled=fits_any,
        verdict=(
            "continue to step 2"
            if fits_any
            else "DEAD at step 1: the whole catalog does not fit in two compute days at either scaled"
        ),
    )
    DATA.mkdir(exist_ok=True)
    (DATA / "step1_verdict.json").write_text(json.dumps(step1, indent=1) + "\n")
    print(table)
    print(per_scaled)
    print(step1["verdict"])


def cmd_plan_step3(args: argparse.Namespace) -> None:
    """Size step 3's catalog-sketching arrays from step 1's measured rates.

    Prints one line per scaled value, "scaled n_chunks last_array_index hours mem_gb", for
    299_gate.sbatch to submit; the reasoning goes to stderr. Exits 1 when step 1 killed the
    idea.
    """
    step1 = json.loads((DATA / "step1_verdict.json").read_text())
    if not step1["fits_at_some_scaled"]:
        sys.exit("step 1 killed the idea; there is no step 3 to plan")
    stats = json.loads((work_dir() / "catalog_stats.json").read_text())
    timing = {
        r["setting"]: r
        for r in _read_csv(
            TABLES / "299_step1_sketch_translate_timing_1gb_slice_omrgc_v2.csv"
        )
    }
    # One catalog part per task. sourmash holds every sketch of a task in memory until it
    # writes the file, and a task sketches all nine settings at once, so memory grows with
    # the genes per task times nine. Measured on Sherlock (array 48648425, 2026-10-20): about
    # 3.8 GB per part of 182_716 genes for the nine settings at scaled 100; 43 parts per task
    # ran out of 11 GB after three. Step 1 gives the same from one setting on the 1 GB slice.
    n_chunks = stats["n_parts"]
    chunk_gb = step1["catalog_fna_gb"] / n_chunks
    for scaled in [x for x in SCALEDS if not args.scaled or x in args.scaled]:
        rate = sum(
            float(timing[setting_id(a, k, scaled)]["cpu_s_per_gb"])
            for a in ALPHABETS
            for k in KSIZES
        )
        rss_per_gb = (
            max(
                float(timing[setting_id(a, k, scaled)]["slice_max_rss_gb"] or 0)
                for a in ALPHABETS
                for k in KSIZES
            )
            / step1["slice_gb"]
        )
        mem = rss_per_gb * len(ALPHABETS) * len(KSIZES) * chunk_gb
        hours = rate * chunk_gb / 3600
        print(
            f"scaled {scaled}: {rate:.0f} CPU-s per GB for all nine settings, {n_chunks} tasks of "
            f"{chunk_gb:.3f} GB, predicted {hours:.2f} h and {mem:.1f} GB per task",
            file=sys.stderr,
        )
        # Ask: three times the predicted memory plus 2 GB; four times the time, at least 1 h.
        print(
            scaled,
            n_chunks,
            n_chunks + N_SAMPLE_PARTS - 1,
            max(1, int(4 * hours) + 1),
            int(3 * mem) + 2,
        )


def _read_csv(path: Path) -> list[dict]:
    import csv

    with open(path) as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- scoring


def read_tsv(path: Path, schema: dict):
    """Headerless TSV with a fixed schema; an empty file is an empty table, not an error."""
    import polars as pl

    if path.stat().st_size == 0:
        return pl.DataFrame(schema=schema)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(schema),
        schema_overrides=schema,
    )


def cmd_score(args: argparse.Namespace) -> None:
    """Recall of DIAMOND's hits and shuffled-hit rate for every setting; verdict.json."""
    import polars as pl

    work = work_dir()
    queries = pl.read_csv(DATA / "queries.tsv", separator="\t")
    is_shuffled = dict(zip(queries["query_id"], queries["shuffled"]))
    kind = dict(zip(queries["query_id"], queries["kind"]))

    dm_cols = "qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen".split()
    diamond = pl.read_csv(
        work / "diamond" / "hits.tsv",
        separator="\t",
        has_header=False,
        new_columns=dm_cols,
    )
    diamond = (
        diamond.filter(pl.col("evalue") < TRUTH_EVALUE)
        .select(pl.col("qseqid").alias("query"), pl.col("sseqid").alias("target"))
        .unique()
    )
    diamond = diamond.with_columns(
        pl.col("query").replace_strict(is_shuffled).alias("shuffled")
    )
    truth = diamond.filter(~pl.col("shuffled")).select("query", "target")
    n_truth = truth.height
    n_real_queries = int((~queries["shuffled"]).sum())
    stats = json.loads((work / "catalog_stats.json").read_text())
    catalog_genes, sample_genes = stats["catalog_genes"], stats["sample_genes"]
    dm_time = parse_gnu_time(work / "diamond" / "blastp.time")
    dm_shuffled = diamond.filter(pl.col("shuffled")).height
    rows = [
        dict(
            method="DIAMOND blastp",
            setting="diamond",
            count_source="full",
            alphabet="protein (DIAMOND)",
            ksize=None,
            scaled=None,
            real_hits=n_truth,
            truth_hits_found=n_truth,
            recall=1.0 if n_truth else None,
            shuffled_hits=dm_shuffled,
            shuffled_rate=dm_shuffled / (N_SHUFFLES * n_truth) if n_truth else None,
            precision_vs_diamond=1.0 if n_truth else None,
            wall_s=dm_time["wall_s"] if dm_time else None,
            cpu_s=dm_time["cpu_s"] if dm_time else None,
            max_rss_gb=dm_time["max_rss_gb"] if dm_time else None,
        )
    ]
    for alphabet, ksize, scaled in SETTINGS:
        sid = setting_id(alphabet, ksize, scaled)
        out = work / "search" / sid
        if not (out / "done").exists():
            print(
                f"{sid}: search did not finish (no {out / 'done'}); left out",
                file=sys.stderr,
            )
            continue
        status = json.loads((out / "status.json").read_text())
        source = status["count_source"]  # "full" or "sample_1pct"
        runs = ["recall", "sample"] + (["full"] if source == "full" else [])
        for run in runs:
            bad = int((out / f"{run}.bad.txt").read_text())
            if bad:
                sys.exit(
                    f"{sid} {run}: {bad} manysearch lines had the wrong column count; fix the reducer first"
                )
        count_run = "full" if source == "full" else "sample"
        counts = read_tsv(
            out / f"{count_run}.counts.tsv", {"query": pl.Utf8, "n": pl.Int64}
        )
        counts = counts.with_columns(
            pl.col("query")
            .replace_strict(is_shuffled, return_dtype=pl.Boolean)
            .alias("shuffled")
        )
        # Recall comes from the search against only the genes DIAMOND hit (exact).
        pairs = read_tsv(
            out / "recall.pairs.tsv",
            {
                "query": pl.Utf8,
                "target": pl.Utf8,
                "evalue": pl.Float64,
                "containment": pl.Float64,
                "shared": pl.Int64,
            },
        )
        real_hits = int(counts.filter(~pl.col("shuffled"))["n"].sum())
        n_shuf = int(counts.filter(pl.col("shuffled"))["n"].sum())
        # On the sample, hit counts are scaled to the catalog; the ratio is unchanged.
        to_catalog = catalog_genes / sample_genes if source == "sample_1pct" else 1.0
        found = (
            pairs.filter(pl.col("evalue") < TRUTH_EVALUE)
            .select("query", "target")
            .unique()
            .join(truth, on=["query", "target"])
            .height
        )
        timing = parse_gnu_time(out / f"{count_run}.time")
        n_empty = json.loads((out / "query_hashes.json").read_text())
        rows.append(
            dict(
                method="sourmash manysearch",
                setting=sid,
                alphabet=alphabet,
                ksize=ksize,
                scaled=scaled,
                count_source=source,
                real_hits=real_hits * to_catalog,
                truth_hits_found=found,
                recall=found / n_truth if n_truth else None,
                shuffled_hits=n_shuf * to_catalog,
                shuffled_rate=n_shuf / (N_SHUFFLES * real_hits) if real_hits else None,
                precision_vs_diamond=(
                    found / (real_hits * to_catalog) if real_hits else None
                ),
                real_queries_with_no_hash=n_empty["real_queries_with_no_hash"],
                time_source=(
                    "full catalog search"
                    if source == "full"
                    else "1% gene sample search (not the full search time)"
                ),
                wall_s=timing["wall_s"] if timing else None,
                cpu_s=timing["cpu_s"] if timing else None,
                max_rss_gb=timing["max_rss_gb"] if timing else None,
            )
        )
    if len(rows) == 1:
        sys.exit("no sourmash setting finished its search; nothing to score")
    table = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        (
            (pl.col("recall") >= PASS_RECALL)
            & (pl.col("shuffled_rate") < PASS_SHUFFLED_RATE)
            & (pl.col("method") != "DIAMOND blastp")
        )
        .fill_null(False)
        .alias("passes")
    )
    table.write_csv(TABLES / "299_kill_test_recall_vs_shuffled_rate_omrgc_v2.csv")
    passing = table.filter(pl.col("passes"))["setting"].to_list()
    verdict = dict(
        rule=f"pass if some sourmash setting finds >= {PASS_RECALL:.0%} of DIAMOND's real-query hits at E < {TRUTH_EVALUE} "
        f"while its shuffled copies hit < {PASS_SHUFFLED_RATE:.0%} as many targets as the real queries (per copy)",
        n_real_queries=n_real_queries,
        n_diamond_truth_pairs=n_truth,
        passing_settings=passing,
        result="PASS" if passing else "DEAD at step 3",
        per_setting=table.to_dicts(),
    )
    step1_path = DATA / "step1_verdict.json"
    if step1_path.exists():
        verdict["step1"] = json.loads(step1_path.read_text())
    (DATA / "verdict.json").write_text(
        json.dumps(verdict, indent=1, default=str) + "\n"
    )
    print(table)
    print(verdict["result"], passing)
    plot_kill_test(table)


def plot_kill_test(table) -> None:
    """Figure 1: recall of DIAMOND's hits (x) against shuffled-hit rate (y), one mark per setting.

    Colour = alphabet, shape = k, filled = scaled 100 and open = scaled 1000, a ring = the
    y value was estimated on the 1% gene sample. The y axis is logarithmic; a rate of 0 sits
    on its own line at the bottom. The grey box is the pass region fixed before the run.
    """
    sys.path.insert(0, str(ROOT / "notebooks"))
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import pubfig as pf

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 85)
    colour = {
        "protein": pf.OKABE_ITO["blue"],
        "dayhoff": pf.OKABE_ITO["vermillion"],
        "hp": pf.OKABE_ITO["bluish_green"],
    }
    marker = {7: "o", 10: "s", 13: "^"}
    zero_line, y_min = (
        0.005,
        0.01,
    )  # % values: where a rate of 0 is drawn, and the axis floor above it
    rates = [r * 100 for r in table["shuffled_rate"].drop_nulls().to_list() if r > 0]
    y_max = max([150.0] + [r * 1.5 for r in rates])

    def y_of(rate):
        return zero_line if rate == 0 else rate * 100

    ax.add_patch(
        mpatches.Rectangle(
            (PASS_RECALL * 100, zero_line * 0.7),
            103 - PASS_RECALL * 100,
            PASS_SHUFFLED_RATE * 100 - zero_line * 0.7,
            facecolor=pf.GREY,
            alpha=0.3,
            edgecolor="none",
            zorder=1,
        )
    )
    undrawn = []
    for row in table.filter(table["method"] == "sourmash manysearch").iter_rows(
        named=True
    ):
        if row["recall"] is None or row["shuffled_rate"] is None:
            undrawn.append(row["setting"])
            continue
        x, y = row["recall"] * 100, y_of(row["shuffled_rate"])
        ax.plot(
            x,
            y,
            marker=marker[row["ksize"]],
            markersize=5,
            linestyle="",
            zorder=3,
            color=colour[row["alphabet"]],
            markerfacecoloralt="white",
            fillstyle=FILL_BY_SCALED[row["scaled"]],
        )
        if row["count_source"] == "sample_1pct":
            ax.scatter(
                x,
                y,
                marker="o",
                s=90,
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
    dm = table.filter(table["method"] == "DIAMOND blastp").row(0, named=True)
    if dm["shuffled_rate"] is not None:
        ax.scatter(
            100, y_of(dm["shuffled_rate"]), marker="*", color="black", s=45, zorder=4
        )
    if undrawn:
        fig.supxlabel(
            f"Not drawn, no real-query hit so no rate ({len(undrawn)}): "
            + ", ".join(undrawn),
            fontsize=5,
        )
    ax.set_yscale("log")
    ax.set_ylim(zero_line * 0.6, y_max)
    ticks = [t for t in (0.01, 0.1, 1, 10, 100) if t < y_max]
    ax.set_yticks([zero_line] + ticks, ["0"] + [f"{t:g}" for t in ticks])
    ax.axhline(
        np.sqrt(zero_line * y_min), color="black", linewidth=0.5, linestyle=(0, (2, 2))
    )
    ax.set_xlim(-3, 103)
    ax.set_xlabel("DIAMOND hits found (% of real-query pairs at E < 1e-5)")
    ax.set_ylabel("Shuffled-copy hits per copy\n(% of real-query hits, log scale)")
    handles = [
        mpatches.Patch(
            facecolor=pf.GREY, alpha=0.3, label="pass: >= 80% found, < 1% shuffled"
        )
    ]
    handles += [
        mlines.Line2D([], [], color=c, marker="o", linestyle="", label=a)
        for a, c in colour.items()
    ]
    handles += [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=m,
            markerfacecolor="none",
            linestyle="",
            label=f"k = {k}",
        )
        for k, m in marker.items()
    ]
    handles += [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="",
            markerfacecoloralt="white",
            fillstyle=FILL_BY_SCALED[scaled],
            label=f"scaled {scaled} ({FILL_LABEL[FILL_BY_SCALED[scaled]]})",
        )
        for scaled in sorted(FILL_BY_SCALED)
    ] + [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=8,
            markerfacecolor="none",
            linewidth=0.5,
            linestyle="",
            label="ring: y from 1% gene sample",
        ),
        mlines.Line2D(
            [], [], color="black", marker="*", linestyle="", label="DIAMOND blastp"
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    FIGURES.mkdir(exist_ok=True)
    pf.save(
        fig, str(FIGURES / "299_recall_vs_shuffled_rate_sourmash_vs_diamond_omrgc_v2")
    )


# ---------------------------------------------------------------- Logan arithmetic


def cmd_logan(args: argparse.Namespace) -> None:
    """Step 5: CPU-hours to sketch Logan's proteins, from step 1's per-gene rate."""
    step1 = json.loads((DATA / "step1_verdict.json").read_text())
    stats = json.loads((work_dir() / "catalog_stats.json").read_text())
    genes_in_slice = stats["slice_genes"]
    logan_proteins = 109.4e9  # Logan preprint v2 (bioRxiv 2024.07.30.605881, 2025-09-01): "109.4 billion proteins"
    rows = []
    for scaled_row in step1["per_scaled"]:
        cpu_h_catalog = scaled_row["all_nine_cpu_hours_arithmetic"]
        cpu_s_per_gene = cpu_h_catalog * 3600 / step1["catalog_genes"]
        rows.append(
            dict(
                scaled=scaled_row["scaled"],
                cpu_s_per_gene_all_nine=cpu_s_per_gene,
                logan_cpu_hours_arithmetic=cpu_s_per_gene * logan_proteins / 3600,
                includes_lower_bounds=scaled_row["any_lower_bound"],
            )
        )
    out = dict(
        logan_proteins=logan_proteins,
        genes_in_slice=genes_in_slice,
        per_scaled=rows,
        note="Arithmetic, not measured. Per-gene time includes translating DNA in six frames, "
        "which Logan's proteins do not need. Where includes_lower_bounds is true, some settings "
        "stopped at the step 1 time limit, so the number is not a bound in either direction.",
    )
    (DATA / "step5_logan_arithmetic.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queries")
    q.add_argument(
        "--clam-fasta",
        type=Path,
        default=Path.home() / "data/arctic-clam/Glanz_Clam_AllModels.proteins.fasta",
    )
    q.add_argument(
        "--bhf-fasta",
        type=Path,
        required=True,
        help="PR 80 tables/case_gallery/241/queries.fa",
    )
    q.add_argument(
        "--bhf-hits-json",
        type=Path,
        required=True,
        help="PR 80 reports/case_gallery/hits241.BHF.json",
    )
    q.set_defaults(func=cmd_queries)
    s = sub.add_parser("setting")
    s.add_argument("index", type=int)
    s.set_defaults(func=cmd_setting)
    sub.add_parser("step1-table").set_defaults(func=cmd_step1_table)
    ps = sub.add_parser("plan-step3")
    ps.add_argument("--scaled", type=int, nargs="*", help="only these scaled values")
    ps.set_defaults(func=cmd_plan_step3)
    ix = sub.add_parser("indices")
    ix.add_argument("--scaled", type=int, nargs="+", required=True)
    ix.set_defaults(func=cmd_indices)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("logan").set_defaults(func=cmd_logan)
    d = sub.add_parser("drop-empty")
    d.add_argument("src", type=Path)
    d.add_argument("dst", type=Path)
    d.set_defaults(func=cmd_drop_empty)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
