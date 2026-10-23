#!/usr/bin/env python3
"""Build the time-split query set for notebook 280, and count every step.

t0 = Pfam 33.1 (May 2020, UniProtKB reference proteomes 2019_08).
t1 = Pfam 38.2 (data files dated 22 January 2026, published June 2026; reference
     proteomes 2025_03).

A query is a protein sequence in Pfam 33.1 pfamseq with no Pfam-A domain at t0 whose
identical sequence (same MD5) carries a Pfam-A domain at t1. One instance is one
(query, family, t1 start, t1 end). Coordinates here are Pfam's: 1-based, end included.

Steps (the numbers match the plan in the PR body):
 1. every pfamseq 33.1 sequence, with its MD5 computed as Pfam does it (lowercase hex of
    the uppercase sequence), checked against the t0 regions file on 1_000 proteins
 2. no row in the t0 regions file, by accession or by MD5
 3. MD5 in the t1 regions file; the rest split into "still in the 2025_03 reference
    proteomes, still no domain" and "left the reference proteomes" by Pfam 38.0 pfamseq
 4. gained family existed at t0 (listed in Pfam 33.1 Pfam-A.clans.tsv)
 5. family has at least one domain, with coordinates, on a Swiss-Prot 2019_08 protein
 6. tags: Pfam type at t1, length, kingdom, "Uncharacterized protein" at t0
 7. query MD5 not equal to any Swiss-Prot 2019_08 sequence

Run inside the kmerseek 0.4.0-rc5 image (polars). Memory: about 120 GB at peak, from
holding the pfamseq table (Pfam 33.1 has ~47 million sequences).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import tarfile
from pathlib import Path

import polars as pl

SEED = 280
SHORT_AA = 60  # "short region": under 60 aa


def md5_of(seq: str) -> str:
    """Pfam's MD5: lowercase hex digest of the uppercase sequence, no '*' or line breaks."""
    return hashlib.md5(seq.upper().encode()).hexdigest()


def parse_pfamseq(path: Path, out: Path) -> None:
    """pfamseq.gz -> parquet of acc, version, uniprot_id, mnemonic, description, length,
    md5, seq. Header: '>ACC.VERSION ID description {evidence}'."""
    rows: dict[str, list] = {k: [] for k in
                             ("acc", "version", "uniprot_id", "mnemonic", "description",
                              "length", "md5", "seq")}
    writer_parts, part = [], 0

    def flush():
        nonlocal rows, part
        if not rows["acc"]:
            return
        p = out.with_name(f"{out.stem}.part{part:03d}.parquet")
        pl.DataFrame(rows).write_parquet(p, compression="zstd")
        writer_parts.append(p)
        part += 1
        rows = {k: [] for k in rows}

    def add(header: str, chunks: list[str]):
        seq = "".join(chunks)
        first, _, rest = header.partition(" ")
        acc, _, ver = first.partition(".")
        uid, _, desc = rest.partition(" ")
        desc = re.sub(r"\s*\{ECO:[^}]*\}\s*$", "", desc)
        rows["acc"].append(acc)
        rows["version"].append(int(ver) if ver.isdigit() else None)
        rows["uniprot_id"].append(uid)
        rows["mnemonic"].append(uid.rpartition("_")[2] if "_" in uid else None)
        rows["description"].append(desc)
        rows["length"].append(len(seq))
        rows["md5"].append(md5_of(seq))
        rows["seq"].append(seq)
        if len(rows["acc"]) >= 2_000_000:
            flush()

    header, chunks = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    add(header, chunks)
                header, chunks = line[1:].rstrip("\n"), []
            else:
                chunks.append(line.strip())
    if header is not None:
        add(header, chunks)
    flush()
    pl.concat([pl.scan_parquet(p) for p in writer_parts]).sink_parquet(out, compression="zstd")
    for p in writer_parts:
        p.unlink()


def md5s_of_fasta(path: Path) -> set[str]:
    """MD5 of every sequence in a gzipped FASTA (used for Pfam 38.0 pfamseq)."""
    out, chunks, started = set(), [], False
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if started:
                    out.add(md5_of("".join(chunks)))
                chunks, started = [], True
            else:
                chunks.append(line.strip())
    if started:
        out.add(md5_of("".join(chunks)))
    return out


def parse_sprot(tar_path: Path, out: Path) -> pl.DataFrame:
    """Swiss-Prot 2019_08 flat file (inside the release tarball) -> acc, uniprot_id,
    kingdom (first OC word), length, md5, seq. Primary accession only."""
    with tarfile.open(tar_path) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("uniprot_sprot.dat.gz"))
        raw = tf.extractfile(member)
        rows = {k: [] for k in ("acc", "uniprot_id", "kingdom", "length", "md5", "seq")}
        with gzip.open(raw, "rt") as fh:
            uid = acc = king = None
            seq, in_seq = [], False
            for line in fh:
                tag = line[:2]
                if tag == "ID":
                    uid = line[5:].split()[0]
                elif tag == "AC" and acc is None:
                    acc = line[5:].split(";")[0].strip()
                elif tag == "OC" and king is None:
                    king = line[5:].split(";")[0].strip().rstrip(".")
                elif tag == "SQ":
                    in_seq = True
                elif line.startswith("//"):
                    s = "".join(seq)
                    rows["acc"].append(acc)
                    rows["uniprot_id"].append(uid)
                    rows["kingdom"].append(king)
                    rows["length"].append(len(s))
                    rows["md5"].append(md5_of(s))
                    rows["seq"].append(s)
                    uid = acc = king = None
                    seq, in_seq = [], False
                elif in_seq:
                    seq.append(line.replace(" ", "").strip())
    df = pl.DataFrame(rows)
    df.write_parquet(out, compression="zstd")
    return df


def read_regions(path: Path, columns: list[str]) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", has_header=True, infer_schema=False).select(columns)


def pfam_types(hmm_dat: Path) -> pl.DataFrame:
    """Pfam-A.hmm.dat.gz -> pfam_acc (no version), pfam_id, pfam_type (the #=GF TP line)."""
    rows, cur = [], {}
    with gzip.open(hmm_dat, "rt") as fh:
        for line in fh:
            if line.startswith("#=GF ID"):
                cur["pfam_id"] = line.split(None, 2)[2].strip()
            elif line.startswith("#=GF AC"):
                cur["pfam_acc"] = line.split(None, 2)[2].strip().split(".")[0]
            elif line.startswith("#=GF TP"):
                cur["pfam_type"] = line.split(None, 2)[2].strip()
            elif line.startswith("//"):
                rows.append(cur)
                cur = {}
    return pl.DataFrame(rows)


def clans(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", has_header=False, infer_schema=False,
                       new_columns=["pfam_acc", "clan_acc", "clan_id", "pfam_id", "pfam_desc"])


def speclist_kingdoms(path: Path) -> pl.DataFrame:
    """UniProt speclist.txt -> mnemonic, kingdom code (A, B, E, V, O)."""
    rows = []
    pat = re.compile(r"^([A-Z0-9]{1,5})\s+([ABEVO])\s+(\d+):\s+N=")
    for line in path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return pl.DataFrame(rows, schema=["mnemonic", "kingdom_code"], orient="row").unique("mnemonic")


KINGDOM = {"A": "Archaea", "B": "Bacteria", "E": "Eukaryota", "V": "Viruses", "O": "Other"}


def length_bin(col: str) -> pl.Expr:
    return (pl.when(pl.col(col) < SHORT_AA).then(pl.lit(f"<{SHORT_AA} aa"))
            .otherwise(pl.lit(f">={SHORT_AA} aa")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d, out = args.data, args.out
    out.mkdir(parents=True, exist_ok=True)
    funnel: list[dict] = []

    def count(step: str, what: str, n: int, note: str = "") -> None:
        funnel.append({"step": step, "what": what, "n": n, "note": note})
        print(f"[{step}] {what}: {n:_} {note}", flush=True)

    # ---- step 1: pfamseq 33.1 ------------------------------------------------------
    seq_pq = out / "pfamseq_33.1.parquet"
    if not seq_pq.exists():
        parse_pfamseq(d / "Pfam33.1/pfamseq.gz", seq_pq)
    seqs = pl.scan_parquet(seq_pq)
    n1 = seqs.select(pl.len()).collect().item()
    count("1", "sequences in Pfam 33.1 pfamseq", n1)

    t0 = read_regions(d / "Pfam33.1/Pfam-A.regions.tsv.gz",
                      ["pfamseq_acc", "md5", "pfamA_acc", "seq_start", "seq_end"])
    t0_seq = t0.select("pfamseq_acc", "md5").unique().collect()
    t0_seq.write_parquet(out / "t0_regions_seqs.parquet")

    # MD5 check on 1_000 proteins that have a t0 row.
    rng = random.Random(SEED)
    accs = t0_seq["pfamseq_acc"].unique().sort().to_list()
    pick = pl.DataFrame({"pfamseq_acc": rng.sample(accs, 1_000)})
    chk = (pick.join(t0_seq, on="pfamseq_acc")
           .join(seqs.select(pl.col("acc").alias("pfamseq_acc"), pl.col("md5").alias("md5_ours"))
                 .collect(), on="pfamseq_acc", how="left"))
    n_bad = chk.filter(pl.col("md5") != pl.col("md5_ours")).height
    n_missing = chk.filter(pl.col("md5_ours").is_null()).height
    count("1", "MD5 check: proteins checked", chk["pfamseq_acc"].n_unique())
    count("1", "MD5 check: disagree", n_bad, f"(not found in pfamseq: {n_missing})")
    if n_bad or n_missing:
        chk.write_csv(out / "md5_check_failures.csv")
        raise SystemExit("MD5 disagreement: see md5_check_failures.csv")

    # ---- step 2: no t0 domain ---------------------------------------------------------
    t0_accs = t0_seq.select(pl.col("pfamseq_acc").alias("acc")).unique()
    t0_md5 = t0_seq.select("md5").unique()
    no_acc = seqs.select("acc", "md5").collect().join(t0_accs, on="acc", how="anti")
    count("2a", "no t0 row by accession", no_acc.height)
    dark0 = no_acc.join(t0_md5, on="md5", how="anti")
    count("2", "no t0 row by accession or by MD5", dark0.height,
          f"(dropped {no_acc.height - dark0.height:_} whose identical sequence has a t0 domain)")

    # ---- step 3: same MD5 has a t1 domain -------------------------------------------
    t1 = read_regions(d / "Pfam38.2/Pfam-A.regions.tsv.gz",
                      ["pfamseq_acc", "md5", "pfamA_acc", "seq_start", "seq_end"])
    dark_md5 = dark0.select("md5").unique()
    t1_hits = (t1.join(dark_md5.lazy(), on="md5", how="semi").collect()
               .with_columns(pl.col("seq_start").cast(pl.Int64), pl.col("seq_end").cast(pl.Int64)))
    t1_hits.write_parquet(out / "t1_regions_on_t0_dark.parquet")
    gained_md5 = t1_hits.select("md5").unique()
    q3 = dark0.join(gained_md5, on="md5", how="semi")
    count("3", "same sequence has a Pfam-A domain at t1", q3.height,
          f"({q3['md5'].n_unique():_} distinct sequences)")
    rest = dark0.join(gained_md5, on="md5", how="anti")
    t1_md5_file = out / "pfamseq_38.0_md5.parquet"
    if not t1_md5_file.exists():
        pl.DataFrame({"md5": sorted(md5s_of_fasta(d / "Pfam38.0/pfamseq.gz"))}).write_parquet(t1_md5_file)
    t1_all = pl.read_parquet(t1_md5_file)
    still = rest.join(t1_all, on="md5", how="semi")
    count("3b", "still in 2025_03 reference proteomes, still no domain", still.height)
    count("3c", "left the reference proteomes (cannot be checked)", rest.height - still.height)

    # One row per (query sequence, t1 domain). A query is one MD5; keep one accession per
    # MD5 (the first in sort order) so two accessions with one sequence are not counted twice.
    q_acc = q3.sort("acc").unique("md5", keep="first")
    inst = (t1_hits.select("md5", pl.col("pfamA_acc").alias("pfam_acc"),
                           pl.col("seq_start").alias("t1_start"), pl.col("seq_end").alias("t1_end"))
            .unique()
            .join(q_acc, on="md5"))
    count("3", "instances (query, family, t1 start, t1 end)", inst.height)

    # ---- step 4: family existed at t0 -------------------------------------------------
    c33 = clans(d / "Pfam33.1/Pfam-A.clans.tsv.gz")
    c38 = clans(d / "Pfam38.2/Pfam-A.clans.tsv.gz")
    types = pfam_types(d / "Pfam38.2/Pfam-A.hmm.dat.gz")
    inst = (inst.join(types, on="pfam_acc", how="left")
            .join(c38.select("pfam_acc", pl.col("clan_acc").alias("clan_t1")), on="pfam_acc", how="left")
            .with_columns(length=pl.col("t1_end") - pl.col("t1_start") + 1,
                          existed_t0=pl.col("pfam_acc").is_in(c33["pfam_acc"].implode())))
    inst = inst.with_columns(length_bin=length_bin("length"))

    # ---- step 5: family has a domain on a Swiss-Prot 2019_08 protein -------------------
    sp_pq = out / "sprot_2019_08.parquet"
    sprot = (pl.read_parquet(sp_pq) if sp_pq.exists()
             else parse_sprot(d / "uniprot_2019_08/uniprot_sprot-only2019_08.tar.gz", sp_pq))
    count("ref", "Swiss-Prot 2019_08 entries", sprot.height)
    up = read_regions(d / "Pfam33.1/Pfam-A.regions.uniprot.tsv.gz",
                      ["uniprot_acc", "md5", "pfamA_acc", "seq_start", "seq_end"])
    sp_dom = (up.join(sprot.lazy().select(pl.col("acc").alias("uniprot_acc"),
                                          pl.col("md5").alias("sp_md5")),
                      on="uniprot_acc", how="inner").collect())
    n_md5_off = sp_dom.filter(pl.col("md5") != pl.col("sp_md5")).select("uniprot_acc").n_unique()
    sp_dom = (sp_dom.filter(pl.col("md5") == pl.col("sp_md5"))
              .select(pl.col("uniprot_acc").alias("sp_acc"), pl.col("pfamA_acc").alias("pfam_acc"),
                      pl.col("seq_start").cast(pl.Int64).alias("sp_start"),
                      pl.col("seq_end").cast(pl.Int64).alias("sp_end"))
              .unique())
    sp_dom.write_parquet(out / "sprot_2019_08_pfam33.1_domains.parquet")
    count("ref", "t0 Pfam domains on Swiss-Prot 2019_08 proteins", sp_dom.height,
          f"({sp_dom['sp_acc'].n_unique():_} proteins, {sp_dom['pfam_acc'].n_unique():_} families;"
          f" {n_md5_off:_} proteins dropped: their 33.1 MD5 differs from the 2019_08 sequence)")
    fam_sp = sp_dom.group_by("pfam_acc").agg(n_sp_domains=pl.len())
    inst = (inst.join(fam_sp, on="pfam_acc", how="left")
            .with_columns(pl.col("n_sp_domains").fill_null(0))
            .with_columns(reachable=pl.col("n_sp_domains") > 0))

    # ---- step 6: tags ------------------------------------------------------------------
    kings = speclist_kingdoms(d / "uniprot_2019_08/speclist_current.txt")
    meta = seqs.select("acc", "mnemonic", "description", pl.col("length").alias("query_length")).collect()
    inst = (inst.join(meta, on="acc", how="left")
            .join(kings, on="mnemonic", how="left")
            .with_columns(kingdom=pl.col("kingdom_code").replace_strict(KINGDOM, default="unknown"),
                          uncharacterized_t0=pl.col("description").str.contains(
                              "(?i)^uncharacterized protein")))

    # ---- step 7: query is not itself a Swiss-Prot 2019_08 sequence ---------------------
    inst = inst.with_columns(query_in_sprot=pl.col("md5").is_in(sprot["md5"].implode()))

    inst = inst.with_columns(
        step=pl.when(~pl.col("existed_t0")).then(pl.lit("4: family created after t0"))
        .when(~pl.col("reachable")).then(pl.lit("5: no t0 Swiss-Prot domain of the family"))
        .when(pl.col("query_in_sprot")).then(pl.lit("7: query sequence is in Swiss-Prot 2019_08"))
        .otherwise(pl.lit("kept")))
    inst.write_parquet(out / "instances_all.parquet")
    kept = inst.filter(pl.col("step") == "kept")
    kept.write_parquet(out / "instances_kept.parquet")

    count("4", "instances whose family existed at t0", inst.filter("existed_t0").height,
          f"(created after t0: {inst.filter(~pl.col('existed_t0')).height:_})")
    s5 = inst.filter(pl.col("existed_t0") & pl.col("reachable"))
    count("5", "and the family has a t0 Swiss-Prot domain", s5.height,
          f"({100 * s5.height / max(1, inst.filter('existed_t0').height):.1f}% of step 4)")
    count("7", "and the query is not a Swiss-Prot 2019_08 sequence", kept.height,
          f"({kept['md5'].n_unique():_} queries, {kept['pfam_acc'].n_unique():_} families)")

    # Funnel by Pfam type x length bin, for steps 3 to 7 (instances).
    by = (inst.with_columns(
            s3=pl.lit(True), s4=pl.col("existed_t0"),
            s5=pl.col("existed_t0") & pl.col("reachable"),
            s7=pl.col("step") == "kept")
          .group_by("pfam_type", "length_bin")
          .agg(pl.col("s3").sum(), pl.col("s4").sum(), pl.col("s5").sum(), pl.col("s7").sum(),
               queries_s7=pl.col("md5").filter(pl.col("step") == "kept").n_unique())
          .sort("pfam_type", "length_bin"))
    by.write_csv(out / "funnel_by_type_length.csv")
    print(by)
    for col in ("kingdom", "uncharacterized_t0"):
        t = kept.group_by(col).agg(n_instances=pl.len(), n_queries=pl.col("md5").n_unique()).sort(col)
        t.write_csv(out / f"kept_by_{col}.csv")
        print(t)
    pl.DataFrame(funnel).write_csv(out / "funnel.csv")
    (out / "funnel.json").write_text(json.dumps(funnel, indent=1))


if __name__ == "__main__":
    main()
