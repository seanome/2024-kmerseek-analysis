#!/usr/bin/env python3
"""Write the search inputs of the time-split test (notebook 280, Stops 2 and 3).

    make_search_inputs.py reference --qs <query_set> --out <dir>
        reference.fasta      Swiss-Prot 2019_08, header = bare accession
        sprot_domains.tsv    accession, Pfam 33.1 family, start, end (1-based, end included)
    make_search_inputs.py reference-pfam-seed --seed <Pfam-A.seed.gz> --sprot <reference.fasta>
                          --queries <queries.fasta> [<queries.fasta> ...] --out <dir>
        reference.fasta      Pfam 33.1 seed members, gaps removed, uppercase, one entry per
                             distinct sequence, named <family>__<accession>_<start>_<end>
        sprot_domains.tsv    entry, family, 1, entry length: the whole entry is the domain.
                             Same name and columns as `reference`, so main.nf and
                             score_transfers.py read it unchanged.
        seed_members.tsv.gz  every seed member, the entry that holds it, and why it was
                             dropped, if it was
        counts.json          entries, families, residues, residues of Swiss-Prot 2019_08,
                             and log2 of their ratio
    make_search_inputs.py queries --qs <query_set> --sample <proposed_sample_queries.csv>
                          [--n-pilot 200 --controls <identity_sample.csv> --n-controls 20]
                          --out <dir>
        queries.fasta        real queries and one DECOY_ copy each (dipeptide shuffle)
        queries.fasta.md5
        queries.tsv          query, md5, is_decoy, source, half (tune/test), role
        instances.tsv        the kept t1 instances on the real queries
    make_search_inputs.py settings --ranges <tsv> --kappa <tsv> --out <dir>
        settings.tsv         alphabet, ksize, c_opt: every alphabet, each k from k_min to
                             k_max, every other k when there are more than 5

Queries and targets are named by bare accession so kmerseek, MMseqs2 and HMMER all report
the same name (see the project memory on the mmseqs UniProt-header trap).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from time_split_common import (  # noqa: E402
    DECOY_PREFIX,
    DECOY_SEED,
    dipeptide_shuffle,
    pair_counts,
    query_half,
)

PILOT_SEED = 280
CONTROL_SEED = 283
CONTROL_MIN_IDENTITY = 90.0  # positive control: closest t0 Swiss-Prot domain >= 90% identical
CONTROL_MIN_COVERAGE = 0.8  # ... over at least 80% of the t1 domain
MAX_K_PER_ALPHABET = 5  # more k-sizes than this in an alphabet's range: take every other


def wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def reference(qs: Path, out: Path) -> None:
    sp = pl.read_parquet(qs / "sprot_2019_08.parquet")
    with open(out / "reference.fasta", "w") as fh:
        for acc, seq in sp.select("acc", "seq").iter_rows():
            fh.write(f">{acc}\n{wrap(seq)}\n")
    dom = pl.read_parquet(qs / "sprot_2019_08_pfam33.1_domains.parquet").sort("sp_acc", "sp_start")
    dom.write_csv(out / "sprot_domains.tsv", separator="\t")
    # Pfam 38.2 clans, for the same-clan secondary readout.
    clans = qs.parent / "data/Pfam38.2/Pfam-A.clans.tsv.gz"
    if clans.exists():
        pl.read_csv(clans, separator="\t", has_header=False, infer_schema=False).write_csv(
            out / "pfam_clans_t1.tsv", separator="\t", include_header=False)
    print(f"reference: {sp.height:_} proteins; {dom.height:_} Pfam 33.1 domains")


SEED_NAME = re.compile(r"^(?P<id>\S+)/(?P<start>\d+)-(?P<end>\d+)$")
SUBSTRING_PROBE = 20  # k of the query k-mer index used to find seed members inside queries


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def read_pfam_seed(path: Path):
    """Yield (family, accession, start, end, sequence) for every seed member of a Pfam
    Stockholm file. Family is the PF accession without its version; accession is the
    member's UniProt accession without its version (#=GS ... AC), or its ID when there is
    none. The sequence has its gap characters removed and is uppercased (Pfam writes
    insert-state residues in lower case)."""
    fam, acc_of, rows, order = None, {}, {}, []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#=GF AC"):
                fam = line.split()[2].split(".")[0]
            elif line.startswith("#=GS ") and line.split()[2] == "AC":
                parts = line.split()
                acc_of[parts[1]] = parts[3].split(".")[0]
            elif line.startswith("//"):
                for name in order:
                    m = SEED_NAME.match(name)
                    if not m:
                        raise SystemExit(f"{fam}: seed name {name!r} is not ID/start-end")
                    seq = re.sub(r"[.\-]", "", "".join(rows[name])).upper()
                    yield (fam, acc_of.get(name, m["id"]), int(m["start"]), int(m["end"]), seq)
                fam, acc_of, rows, order = None, {}, {}, []
            elif line.startswith("#") or not line.strip():
                continue
            else:
                name, aln = line.split()
                if name not in rows:
                    rows[name] = []
                    order.append(name)
                rows[name].append(aln)


def reference_pfam_seed(seed: Path, sprot: Path, query_fastas: list[Path], out: Path) -> None:
    # Every sequence any run searches, decoys included: a seed member must not be one of
    # them or lie inside one.
    queries: dict[str, str] = {}
    for f in query_fastas:
        queries.update(read_fasta(f))
    query_acc = {q for q in queries if not q.startswith(DECOY_PREFIX)}
    qseqs = list(queries.values())
    probe: dict[str, list[int]] = {}
    for i, qs in enumerate(qseqs):
        for j in range(len(qs) - SUBSTRING_PROBE + 1):
            probe.setdefault(qs[j:j + SUBSTRING_PROBE], []).append(i)
    joined = "\n".join(qseqs)

    def inside_a_query(s: str) -> bool:
        if len(s) < SUBSTRING_PROBE:
            return s in joined
        return any(s in qseqs[i] for i in set(probe.get(s[:SUBSTRING_PROBE], ())))

    members, entry_of_seq, entries, domains = [], {}, [], set()
    for fam, acc, start, end, seq in read_pfam_seed(seed):
        name = f"{fam}__{acc}_{start}_{end}"
        if acc in query_acc:
            drop = "accession is a query"
        elif inside_a_query(seq):
            drop = "sequence inside a query"
        elif not seq:
            drop = "empty"
        else:
            drop = ""
        entry = ""
        if not drop:
            # Identical sequences share one entry: kmerseek 0.4.0 merges index entries with
            # the same k-mer set and reports the second under the first one's coordinates
            # (kmerseek PR 141), so duplicates would not be searched separately anyway. The
            # entry keeps one domain row per family, so a hit transfers every family that
            # has this sequence in its seed, as two separate entries would.
            entry = entry_of_seq.get(seq)
            if entry is None:
                entry = entry_of_seq[seq] = name
                entries.append((name, seq))
            domains.add((entry, fam, len(seq)))
        members.append((name, fam, acc, start, end, len(seq), entry, drop))

    for name, _seq in entries:
        if "|" in name or "," in name or any(c.isspace() for c in name):
            raise SystemExit(f"entry name {name!r} holds a character the parsers split on")
    with open(out / "reference.fasta", "w") as fh:
        for name, seq in entries:
            fh.write(f">{name}\n{wrap(seq)}\n")
    pl.DataFrame(sorted(domains), schema=["sp_acc", "pfam_acc", "sp_end"], orient="row").select(
        "sp_acc", "pfam_acc", pl.lit(1).alias("sp_start"), "sp_end"
    ).write_csv(out / "sprot_domains.tsv", separator="\t")
    mem = pl.DataFrame(members, orient="row", schema=[
        "member", "pfam_acc", "accession", "start", "end", "length", "entry", "dropped"])
    with gzip.open(out / "seed_members.tsv.gz", "wb") as fh:
        mem.write_csv(fh, separator="\t")

    seed_res = sum(len(s) for _n, s in entries)
    sprot_seqs = read_fasta(sprot)
    sprot_res = sum(len(s) for s in sprot_seqs.values())
    letters = sorted(set("".join(s for _n, s in entries)))
    counts = {
        "seed_members": mem.height,
        "members_dropped_accession_is_a_query": mem.filter(pl.col("dropped") == "accession is a query").height,
        "members_dropped_sequence_inside_a_query": mem.filter(pl.col("dropped") == "sequence inside a query").height,
        "members_dropped_empty": mem.filter(pl.col("dropped") == "empty").height,
        "members_kept": mem.filter(pl.col("dropped") == "").height,
        "entries": len(entries),
        "members_sharing_an_entry": mem.filter(pl.col("dropped") == "").height - len(entries),
        "entries_in_more_than_one_family": pl.DataFrame(sorted(domains), schema=["e", "f", "l"], orient="row")
        .group_by("e").len().filter(pl.col("len") > 1).height,
        "families_in_seed_file": mem["pfam_acc"].n_unique(),
        "families_with_an_entry": len({f for _e, f, _l in domains}),
        "residues": seed_res,
        "entry_length_median": sorted(len(s) for _n, s in entries)[len(entries) // 2],
        "entries_shorter_than_10_aa": sum(len(s) < 10 for _n, s in entries),
        "letters": "".join(letters),
        "queries_checked": len(queries),
        "sprot2019_08_proteins": len(sprot_seqs),
        "sprot2019_08_residues": sprot_res,
        "residue_ratio_sprot_over_seed": sprot_res / seed_res,
        "bits_saved_log2_ratio": math.log2(sprot_res / seed_res),
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts, indent=1))


def queries(qs: Path, sample_csv: Path, out: Path, n_pilot: int | None,
            controls_csv: Path | None, n_controls: int) -> None:
    kept = pl.read_parquet(qs / "instances_kept.parquet")
    sample = pl.read_csv(sample_csv)
    roles = sample.select("md5", "acc").with_columns(role=pl.lit("sample"))
    if n_pilot is not None:
        rng = random.Random(PILOT_SEED)
        idx = sorted(rng.sample(range(roles.height), min(n_pilot, roles.height)))
        roles = roles[idx]
    if controls_csv is not None and n_controls:
        ident = pl.read_csv(controls_csv)
        pool = (ident.filter((pl.col("pident") >= CONTROL_MIN_IDENTITY)
                             & (pl.col("query_coverage") >= CONTROL_MIN_COVERAGE))
                .filter(~pl.col("md5").is_in(roles["md5"].implode()))
                .unique("md5").sort("acc"))
        if pool.height < n_controls:
            raise SystemExit(f"only {pool.height} positive-control candidates, need {n_controls}")
        rng = random.Random(CONTROL_SEED)
        pick = pool[sorted(rng.sample(range(pool.height), n_controls))]
        roles = pl.concat([roles, pick.select("md5", "acc").with_columns(role=pl.lit("control"))])
    seqs = (pl.scan_parquet(qs / "pfamseq_33.1.parquet").select("acc", "seq")
            .join(roles.lazy().select("acc"), on="acc", how="semi").collect())
    roles = roles.join(seqs, on="acc").sort("role", "acc")

    rng = random.Random(DECOY_SEED)
    lines, rows = [], []
    for acc, md5, role, seq in roles.select("acc", "md5", "role", "seq").iter_rows():
        decoy = dipeptide_shuffle(seq, rng)
        assert pair_counts(decoy) == pair_counts(seq), acc
        half = query_half(acc)
        lines.append(f">{acc}\n{wrap(seq)}\n>{DECOY_PREFIX}{acc}\n{wrap(decoy)}\n")
        rows.append((acc, md5, False, acc, half, role, len(seq)))
        rows.append((f"{DECOY_PREFIX}{acc}", hashlib.md5(decoy.encode()).hexdigest(), True, acc,
                     half, role, len(decoy)))
    text = "".join(lines)
    (out / "queries.fasta").write_text(text)
    (out / "queries.fasta.md5").write_text(hashlib.md5(text.encode()).hexdigest() + "  queries.fasta\n")
    qtab = pl.DataFrame(rows, schema=["query", "md5", "is_decoy", "source", "half", "role", "length"],
                        orient="row")
    qtab.write_csv(out / "queries.tsv", separator="\t")
    inst = kept.join(roles.select("md5", "role"), on="md5").select(
        pl.col("acc").alias("query"), "role", "pfam_acc", "pfam_id", "pfam_type", "clan_t1",
        "t1_start", "t1_end", "length", "length_bin", "kingdom", "uncharacterized_t0")
    inst.write_csv(out / "instances.tsv", separator="\t")
    print(qtab.group_by("role", "is_decoy", "half").len().sort("role", "is_decoy", "half"))
    print(f"{inst.height:_} instances on {inst['query'].n_unique():_} real queries")


def settings(ranges: Path, kappa: Path, out: Path) -> None:
    r = pl.read_csv(ranges, separator="\t")
    kap = pl.read_csv(kappa, separator="\t", comment_prefix="#").select("alphabet", "c_opt")
    rows = []
    for a, lo, hi in r.select("alphabet", "k_min", "k_max").iter_rows():
        ks = list(range(lo, hi + 1))
        ks = ks if len(ks) <= MAX_K_PER_ALPHABET else ks[::2]
        rows += [(a, k) for k in ks]
    s = pl.DataFrame(rows, schema=["alphabet", "ksize"], orient="row").join(kap, on="alphabet", how="left")
    missing = s.filter(pl.col("c_opt").is_null())["alphabet"].unique().to_list()
    if missing:
        raise SystemExit(f"no c_opt for {missing}")
    s.write_csv(out / "settings.tsv", separator="\t")
    print(s.group_by("alphabet", maintain_order=True).agg("ksize", "c_opt"))
    print(f"{s.height} alphabet-k pairs, {2 * s.height} search settings (exact and extend)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "reference-pfam-seed", "queries", "settings"])
    ap.add_argument("--seed", type=Path, help="Pfam-A.seed.gz")
    ap.add_argument("--sprot", type=Path, help="the Swiss-Prot reference.fasta, for its residue count")
    ap.add_argument("--queries", type=Path, nargs="+", help="every queries.fasta any run searches")
    ap.add_argument("--qs", type=Path)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--n-pilot", type=int)
    ap.add_argument("--controls", type=Path)
    ap.add_argument("--n-controls", type=int, default=0)
    ap.add_argument("--ranges", type=Path)
    ap.add_argument("--kappa", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "reference":
        reference(a.qs, a.out)
    elif a.mode == "reference-pfam-seed":
        reference_pfam_seed(a.seed, a.sprot, a.queries, a.out)
    elif a.mode == "queries":
        queries(a.qs, a.sample, a.out, a.n_pilot, a.controls, a.n_controls)
    else:
        settings(a.ranges, a.kappa, a.out)


if __name__ == "__main__":
    main()
