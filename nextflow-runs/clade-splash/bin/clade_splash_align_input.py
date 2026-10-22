#!/usr/bin/env python3
"""Prepare notebook 357 alignment FASTAs with disk-backed domain and sequence lookup.

Separate from clade_splash_pfam.py so fixing this step preserves the completed
Pfam-overlap task's Nextflow cache key.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import shutil
import tempfile
from itertools import groupby
from pathlib import Path

import polars as pl

from clade_splash import TARGET_LENGTH, parse_setting


def cmd_align_input(a: argparse.Namespace) -> None:
    """FASTA of the domain envelopes to hmmalign, one per (anchor, family) of the anchors whose
    window is inside one Pfam family in at least half of their occurrences."""
    anchors = pl.read_parquet(a.pfam).filter(
        ~pl.col("outside_pfam")
        & (pl.col("n_occ_in_pfam") * 2 >= pl.col("n_occurrences"))
    )
    # SQLite keeps the domain join, sorting, and protein sequences on disk. The old
    # implementation retained all joined buckets and all required sequences in RAM.
    # Use node-local storage in Slurm; the task directory is the portable fallback.
    scratch = os.environ.get("L_SCRATCH_JOB") or os.environ.get("L_SCRATCH") or "."
    os.environ.setdefault("SQLITE_TMPDIR", scratch)
    # Indexes and the largest bucket's external sort need disk as well as RAM.
    disk_estimate = (
        3 * sum(Path(p).stat().st_size for p in a.domtblout if Path(p).exists())
        + 2 * sum(Path(p).stat().st_size for p in a.fastas)
        + 16 * max((Path(p).stat().st_size for p in a.occurrences), default=0)
    )
    if shutil.disk_usage(scratch).free < disk_estimate:
        raise OSError(
            f"Insufficient temporary disk space: need about {disk_estimate / 2**30:.1f} GiB in {scratch}"
        )
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory(prefix="align-input-", dir=scratch) as tmp:
        with sqlite3.connect(str(Path(tmp) / "alignment.sqlite")) as db:
            db.execute("PRAGMA cache_size=-65536")  # 64 MiB, independent of input size
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA journal_mode=OFF")  # disposable database, never a cache
            db.execute("CREATE TABLE domains (query TEXT, fam TEXT, lo INT, hi INT)")
            for path in a.domtblout:
                if not Path(path).exists():
                    continue
                batch = []
                with open(path) as fh:
                    for line in fh:
                        if line.startswith("#") or not line.strip():
                            continue
                        f = line.split()
                        batch.append((f[0], f[4].split(".")[0], int(f[19]), int(f[20])))
                        if len(batch) == 10000:
                            db.executemany(
                                "INSERT INTO domains VALUES (?,?,?,?)", batch
                            )
                            batch.clear()
                db.executemany("INSERT INTO domains VALUES (?,?,?,?)", batch)
                db.commit()
            print("Building disk index for Pfam domains", flush=True)
            db.execute("CREATE INDEX domain_query ON domains(query, fam)")
            db.execute(
                "CREATE TABLE hits (tag TEXT, setting TEXT, gap INT, anchor TEXT, "
                "fam TEXT, query TEXT, species INT, protein INT, pos INT, lo INT, hi INT)"
            )
            keep = anchors.select("setting", "gap", "anchor", "main_pfam")
            k_of = {s: parse_setting(s)[1] for s in keep["setting"].unique().to_list()}
            db.execute("CREATE TABLE sequences (query TEXT PRIMARY KEY, sequence TEXT)")
            print("Indexing unique protein sequences on disk", flush=True)
            for path in a.fastas:
                name, seq = None, []

                def save_sequence():
                    if name is not None:
                        db.execute(
                            "INSERT OR IGNORE INTO sequences VALUES (?,?)",
                            (name, "".join(seq)),
                        )

                with open(path) as fh:
                    for line in fh:
                        if line.startswith(">"):
                            save_sequence()
                            query = line[1:].split()[0]
                            name = query
                            seq = []
                        elif name is not None:
                            seq.append(line.strip())
                    save_sequence()
                db.commit()
            for path in a.occurrences:
                print(f"Preparing alignment inputs from {Path(path).name}", flush=True)
                # Project before reading and retain only eligible anchors in this bucket.
                occ = (
                    pl.scan_parquet(path)
                    .select(
                        "setting", "gap", "anchor", "species_idx", "protein_idx", "pos0"
                    )
                    .join(keep.lazy(), on=["setting", "gap", "anchor"])
                    .collect(engine="streaming")
                )
                db.execute(
                    "CREATE TEMP TABLE bucket (tag TEXT, setting TEXT, gap INT, anchor TEXT, "
                    "fam TEXT, query TEXT, species INT, protein INT, pos INT)"
                )

                def records():
                    for setting, gap, anchor, sp, protein, pos, fam in occ.iter_rows():
                        tag = f"{setting.replace(':', '_')}_g{gap}_{anchor}_{fam}"
                        # Anchor codes are unsigned 64-bit; SQLite integers are signed.
                        yield (
                            tag,
                            setting,
                            gap,
                            str(anchor),
                            fam,
                            f"s{sp}_p{protein}",
                            sp,
                            protein,
                            pos,
                        )

                db.executemany(
                    "INSERT INTO bucket VALUES (?,?,?,?,?,?,?,?,?)", records()
                )
                db.execute(
                    "INSERT INTO hits SELECT b.*, d.lo, d.hi FROM bucket b "
                    "JOIN domains d ON b.query=d.query AND b.fam=d.fam"
                )
                db.execute("DROP TABLE bucket")
                db.commit()
                del occ
                # Match the former unique/sort order within each hash bucket.
                # Domain rows with the same start are duplicates under the original rule.
                rows = db.execute(
                    "SELECT tag, setting, gap, anchor, fam, query, species, protein, pos, lo, hi "
                    "FROM hits GROUP BY tag, species, protein, pos, lo "
                    "ORDER BY tag, species, protein, pos, lo"
                )
                for tag, group in groupby(rows, key=lambda r: r[0]):
                    if (out / f"{tag}.fa").exists():
                        raise ValueError(
                            f"Anchor {tag} occurs in more than one hash bucket"
                        )
                    with open(out / f"{tag}.fa", "w") as fh:
                        for (
                            _,
                            setting,
                            gap,
                            anchor,
                            fam,
                            query,
                            sp,
                            protein,
                            pos,
                            lo,
                            hi,
                        ) in group:
                            ts = pos + k_of[setting] + gap
                            if not (lo - 1 <= ts and ts + TARGET_LENGTH <= hi):
                                continue
                            seq = db.execute(
                                "SELECT sequence FROM sequences WHERE query=?", (query,)
                            ).fetchone()
                            if seq is None:
                                raise ValueError(
                                    f"Protein {query} is missing from the supplied FASTA files"
                                )
                            fh.write(f">{query}|{pos}|{lo}|{sp}\n{seq[0][lo - 1:hi]}\n")
                    manifest.append(
                        {
                            "setting": setting,
                            "gap": gap,
                            "anchor": int(anchor),
                            "pfam_acc": fam,
                            "tag": tag,
                        }
                    )
                db.execute("DELETE FROM hits")
                db.commit()
    man = pl.DataFrame(
        manifest,
        schema={
            "setting": pl.String,
            "gap": pl.Int64,
            "anchor": pl.UInt64,
            "pfam_acc": pl.String,
            "tag": pl.String,
        },
    ).sort("tag")
    man.write_csv(a.manifest, separator="\t")
    # Parts of a.part_size rows, so hmmalign and the column tests run in parallel. At least
    # one part, possibly empty, so the rest of the pipeline still runs.
    parts = Path(a.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    n_parts = max(1, -(-man.height // a.part_size))
    for i in range(n_parts):
        man.slice(i * a.part_size, a.part_size).write_csv(
            parts / f"manifest_{i:04d}.tsv", separator="\t"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pfam", required=True)
    p.add_argument("--occurrences", nargs="+", required=True)
    p.add_argument("--domtblout", nargs="+", required=True)
    p.add_argument("--fastas", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--part-size", type=int, default=500)
    a = p.parse_args(argv)
    if a.part_size < 1:
        p.error("--part-size must be positive")
    cmd_align_input(a)


if __name__ == "__main__":
    main()
