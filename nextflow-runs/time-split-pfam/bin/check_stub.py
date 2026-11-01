#!/usr/bin/env python3
"""Check a `make stub` run: every output is named by its query chunk, and scoreTransfers
was staged one file per (setting, arm, chunk) and one per comparator and chunk.

Usage: check_stub.py <results dir>. Reads the settings and chunk count from the run's own
inputs (<results>/../settings.tsv, <results>/../inputs/queries.fasta and the chunk size 2
that `make stub` passes), so the expected names are not copied from the outputs.
"""

import sys
from pathlib import Path

CHUNK_SIZE = 2  # --query_chunk_size in `make stub`
SKIPPED = {("gbmr4", "13")}  # --skip_settings in `make stub`


def main() -> None:
    res = Path(sys.argv[1])
    stub = res.parent
    rows = (stub / "settings.tsv").read_text().splitlines()[1:]
    settings = [tuple(r.split("\t")[:2]) for r in rows if r]
    kept = [s for s in settings if s not in SKIPPED]
    n_seq = sum(
        1
        for line in (stub / "inputs/queries.fasta").read_text().splitlines()
        if line.startswith(">")
    )
    n_chunks = -(-n_seq // CHUNK_SIZE)
    chunks = [f"chunk{i:03d}" for i in range(1, n_chunks + 1)]

    want = {
        "kmerseek": {
            f"{a}.k{k}.{arm}.{c}.transfers.tsv.gz"
            for a, k in kept
            for arm in ("exact", "extend")
            for c in chunks
        },
        "mmseqs2": {f"{c}.mmseqs2.tsv.gz" for c in chunks},
        "phmmer": {f"{c}.phmmer.domtbl.gz" for c in chunks},
        "hmmscan": {f"{c}.hmmscan.domtbl.gz" for c in chunks},
    }
    errors = []
    for d, names in want.items():
        have = {
            p.name for p in (res / d).iterdir() if not p.name.endswith(".stats.tsv")
        }
        if have != names:
            errors.append(f"{d}: published {sorted(have)}, expected {sorted(names)}")

    # The scorer's staged listing: `ls kmerseek stats mmseqs2 phmmer hmmscan`.
    staged = (res / "scores/staged.csv").read_text().split()
    for d, names in want.items():
        missing = names - set(staged)
        if missing:
            errors.append(f"scoreTransfers staged no {sorted(missing)}")
    n_stats = sum(1 for s in staged if s.endswith(".stats.tsv"))
    if n_stats != len(want["kmerseek"]):
        errors.append(
            f"scoreTransfers staged {n_stats} stats files, expected {len(want['kmerseek'])}"
        )
    skipped_seen = [s for s in staged if s.startswith("gbmr4.k13.")]
    if skipped_seen:
        errors.append(f"skipped setting was searched: {skipped_seen}")

    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(
        f"stub OK: {n_chunks} chunks ({', '.join(chunks)}), {len(kept)} settings x 2 arms, "
        f"{len(want['kmerseek'])} kmerseek outputs, 1 file per chunk for each comparator; "
        f"{len(settings) - len(kept)} skipped setting not searched"
    )


if __name__ == "__main__":
    main()
