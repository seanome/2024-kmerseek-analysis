#!/usr/bin/env python3
"""Write the fixed inputs of nextflow-runs/convergence-afp/assets/.

1. kmerseek_settings.tsv: every kmerseek setting of the dark-set midi run (kmerseek 0.4,
   the run the convergence test copies), one row each. A setting is one alphabet, k-size,
   scaled value, low-complexity mask and mismatch penalty. The list is read from that
   run's own per-species outputs: the settings that produced E-values plus the settings
   whose Karlin-Altschul fit was refused, which together are the run's full table. Every
   species must give the same table, or the script stops. ka_queries is how many
   reference sequences the index-time E-value fit searches, by the dark-set rule.
2. alphabet_classes.tsv: the residue classes of each alphabet, so the composition scan
   can count letters in any setting's alphabet. Read from kmerseek's src/rust/alphabets.rs
   at the commit the searches use; the HP alphabets and dayhoff6 (written there as
   per-residue tables, not cluster lists) come from notebooks/hp_conservation_utils.py,
   which copied them from the same file. Every alphabet present in both sources must
   agree, or the script stops.
3. dark_set_measured/*.tsv: the first-attempt memory and time asks the dark-set run
   measured on a reference of reviewed Swiss-Prot minus one clade (about 572_700
   sequences, the size of this reference), copied from that branch at one commit.

    python scripts/convergence_afp_assets.py \
        --dark-results ~/data/invertebrate-dark-set/results-0.4/protein-evalues \
        --alphabets-rs <kmerseek checkout>/src/rust/alphabets.rs \
        --alphabets-rs-source "kmerseek 3c353d4:src/rust/alphabets.rs" \
        --dark-branch-commit ec91da5
"""

import argparse
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "nextflow-runs" / "convergence-afp" / "assets"
DARK_DIR = "nextflow-runs/invertebrate-dark-set/assets"
DARK_TABLES = [
    "kmerseek_index_memory_measured.tsv",
    "kmerseek_search_memory_measured.tsv",
    "kmerseek_search_time_measured.tsv",
]
SPECIES = ["human", "worm", "yeast", "borrelia", "botryllus"]
KEYS = ["alphabet", "ksize", "scaled", "low_complexity_mask", "extension"]
AA20 = "ACDEFGHIKLMNPQRSTVWY"
# The dark-set run's X-drop is this many times the mismatch penalty
# (params.kmerseek_extend_xdrop_per_penalty in its main.nf).
XDROP_PER_PENALTY = 4
# How many reference sequences the index-time Karlin-Altschul fit searches: the dark-set
# pipeline's kaQueriesFor, with its defaults (params.kmerseek_ka_queries 500,
# _bits_base 30, _bits_per_doubling 3, _max 3000). Bits are k x log2(classes).
KA_QUERIES, KA_BITS_BASE, KA_BITS_PER_DOUBLING, KA_QUERIES_MAX = 500, 30.0, 3.0, 3000


def ka_queries(ksize: int, n_classes: int) -> int:
    bits = ksize * math.log2(n_classes)
    if bits <= KA_BITS_BASE:
        return KA_QUERIES
    n = KA_QUERIES * 2 ** ((bits - KA_BITS_BASE) / KA_BITS_PER_DOUBLING)
    return int(round(min(KA_QUERIES_MAX, n) / 50) * 50)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def penalty_string(c: float) -> str:
    """The dark-set pipeline's penaltyString: 2 -> "2", 6.36 -> "6.36", 6.4 -> "6.4"."""
    s = f"{c:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def settings_table(results: Path) -> tuple[pl.DataFrame, list[str]]:
    tables, provenance = [], []
    for sp in SPECIES:
        ev = results / f"{sp}_protein_kmerseek_min_evalue.csv.gz"
        nofit = results / f"{sp}_kmerseek_unfitted.csv"
        a = pl.read_csv(ev, columns=KEYS, infer_schema_length=0).unique()
        b = pl.read_csv(nofit, columns=KEYS, infer_schema_length=0).unique()
        t = pl.concat([a, b]).unique().sort(KEYS)
        tables.append(t)
        provenance += [f"{ev.name}\t{md5(ev)}", f"{nofit.name}\t{md5(nofit)}"]
    first = tables[0]
    for sp, t in zip(SPECIES[1:], tables[1:]):
        if not t.equals(first):
            sys.exit(f"{sp} has a different settings table from {SPECIES[0]}")
    t = (
        first.with_columns(
            ksize=pl.col("ksize").cast(pl.Int64),
            scaled=pl.col("scaled").cast(pl.Int64),
            penalty=pl.col("extension").str.strip_prefix("extend-c"),
        )
        .with_columns(
            xdrop=pl.col("penalty").map_elements(
                lambda c: penalty_string(XDROP_PER_PENALTY * float(c)),
                return_dtype=pl.Utf8,
            ),
        )
        .with_columns(
            # The column the code calls arm_label in other pipelines; here it is the
            # setting label, and it is the name every output file carries.
            setting_label=pl.format(
                "{}.k{}.s{}.lc{}.{}",
                "alphabet",
                "ksize",
                "scaled",
                "low_complexity_mask",
                "extension",
            )
        )
        .sort("alphabet", "ksize", "scaled")
        .select(
            "setting_label",
            "alphabet",
            "ksize",
            "scaled",
            "low_complexity_mask",
            "penalty",
            "xdrop",
        )
    )
    bad = t.filter(~pl.col("penalty").str.contains(r"^\d+(\.\d+)?$"))
    if bad.height:
        sys.exit(f"settings without an extend arm: {bad}")
    return t, provenance


def alphabet_table(rs_path: Path) -> pl.DataFrame:
    src = rs_path.read_text()
    from_rs = {
        name.lower(): re.findall(r'"([A-Z]+)"', body)
        for name, body in re.findall(
            r"const (\w+)_CLUSTERS: &\[&str\] =\s*&\[(.*?)\];", src, re.S
        )
    }
    sys.path.insert(0, str(REPO / "notebooks"))
    import hp_conservation_utils as hcu

    merged = dict(hcu.ALPHABET_CLUSTERS)
    for name, clusters in from_rs.items():
        if name in merged:
            a = sorted("".join(sorted(c)) for c in merged[name])
            b = sorted("".join(sorted(c)) for c in clusters)
            if a != b:
                sys.exit(f"{name}: alphabets.rs and hp_conservation_utils disagree")
        else:
            merged[name] = clusters
    rows = []
    for name, clusters in merged.items():
        if "".join(sorted("".join(clusters))) != AA20:
            sys.exit(f"{name}: classes do not cover the 20 amino acids exactly once")
        for i, c in enumerate(clusters):
            for aa in c:
                rows.append({"alphabet": name, "residue": aa, "class_index": i})
    return pl.DataFrame(rows).sort("alphabet", "residue")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dark-results", type=Path, required=True)
    ap.add_argument("--alphabets-rs", type=Path, required=True)
    ap.add_argument(
        "--alphabets-rs-source",
        required=True,
        help="where the alphabets.rs copy came from, e.g. kmerseek 3c353d4",
    )
    ap.add_argument("--dark-branch-commit", required=True)
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    settings, prov = settings_table(args.dark_results)
    settings.write_csv(ASSETS / "kmerseek_settings.tsv", separator="\t")

    classes = alphabet_table(args.alphabets_rs)
    missing = set(settings["alphabet"]) - set(classes["alphabet"])
    if missing:
        sys.exit(f"no class table for {sorted(missing)}")
    n_classes = dict(
        classes.group_by("alphabet").agg(pl.col("class_index").n_unique()).iter_rows()
    )
    settings = settings.with_columns(
        keyspace_bits=pl.struct("alphabet", "ksize").map_elements(
            lambda r: round(r["ksize"] * math.log2(n_classes[r["alphabet"]]), 3),
            return_dtype=pl.Float64,
        ),
        ka_queries=pl.struct("alphabet", "ksize").map_elements(
            lambda r: ka_queries(r["ksize"], n_classes[r["alphabet"]]),
            return_dtype=pl.Int64,
        ),
    )
    settings.write_csv(ASSETS / "kmerseek_settings.tsv", separator="\t")
    classes.write_csv(ASSETS / "alphabet_classes.tsv", separator="\t")

    out = ASSETS / "dark_set_measured"
    out.mkdir(exist_ok=True)
    for name in DARK_TABLES:
        text = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "show",
                f"{args.dark_branch_commit}:{DARK_DIR}/{name}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (out / name).write_text(text)

    lines = [
        "file\tsource\tmd5",
        *[
            f"kmerseek_settings.tsv\t{p.split(chr(9))[0]}\t{p.split(chr(9))[1]}"
            for p in prov
        ],
        f"alphabet_classes.tsv\t{args.alphabets_rs_source}\t{md5(args.alphabets_rs)}",
        *[
            f"dark_set_measured/{n}\tgit {args.dark_branch_commit}:{DARK_DIR}/{n}\t"
            f"{md5(out / n)}"
            for n in DARK_TABLES
        ],
    ]
    (ASSETS / "PROVENANCE.tsv").write_text("\n".join(lines) + "\n")

    print(f"{settings.height} settings, {settings['alphabet'].n_unique()} alphabets")
    print(settings.group_by("scaled").len().sort("scaled"))
    print(f"{classes['alphabet'].n_unique()} alphabet class tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
