#!/usr/bin/env python3
"""Notebook 357: clade-grouped SPLASH on reduced-alphabet protein k-mers.

An anchor is a k-mer in a reduced alphabet (Dayhoff, 6 letters, or HP, 2 letters). Its
target is the 10 residues that follow it (after a gap of 0, 5 or 10 residues), in the full
20-letter alphabet. For every anchor this script builds a table with one row per target,
one column per taxonomic class, and in each cell the number of species in that class whose
proteome has the anchor followed by that target. The OASIS test of Baharav, Tse and
Salzman (PNAS 2024), as implemented in refresh-bio/SPLASH, asks whether the target
distribution differs between classes.

The frozen rule is notebooks/357_preregistration.md. Subcommands, in pipeline order:

    select      reference proteomes + NCBI taxonomy -> one species per family, classes
                with at least 5 species
    fetch       download the canonical FASTA of each species in a chunk
    anchors     unique anchors of each species, split into hash buckets
    candidates  anchors present in at least 15 species (needed for any labelling to pass)
    targets     every occurrence of a candidate anchor, with its target at each gap
    test        count tables, the OASIS test and the composition scan, for the real class
                labels and the two shuffles
    collect     Benjamini-Hochberg q-values and the counts the rule needs
    occurrences rows of the significant anchors, with residues, for Pfam and the figures
    pfam-overlap, align-control  the two Pfam controls
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import types
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------------------
# Alphabets. The reduced tables are sourmash 0.20.0's, which kmerseek 0.4.0 calls for the
# built-in `dayhoff` (dayhoff6) and `hp` (hp_lehninger2) moltypes
# (sourmash src/encodings.rs, DAYHOFFTABLE and HPTABLE).
# ---------------------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
INVALID = 255
DAYHOFF_GROUPS = {"a": "C", "b": "AGPST", "c": "DENQ", "d": "HKR", "e": "ILMV", "f": "FWY"}
HP_GROUPS = {"h": "AFGILMPVWY", "p": "CDEHKNQRST"}
ALPHABETS = {"dayhoff": DAYHOFF_GROUPS, "hp": HP_GROUPS}
TARGET_LENGTH = 10
TARGET_BASE = len(AMINO_ACIDS)
# 5 species in each of 3 classes is the smallest table the rule keeps.
MIN_SPECIES_PER_CLADE = 5
MIN_CLADES = 3
MIN_SPECIES_FOR_ANY_LABELLING = MIN_SPECIES_PER_CLADE * MIN_CLADES
N_ANCHOR_BUCKETS = 256
N_TARGET_BUCKETS = 64
LABELLINGS = ("real", "shuffle_random", "shuffle_tree")
# Golden-ratio multiplier, so bucket numbers do not follow the low digits of a base-6 code.
_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)

_AA_INDEX = np.full(256, INVALID, dtype=np.uint8)
for _i, _a in enumerate(AMINO_ACIDS):
    _AA_INDEX[ord(_a)] = _i


def reduced_lookup(alphabet: str) -> tuple[np.ndarray, str]:
    """Map residue index (0-19) to reduced-letter index; also return the letters in order."""
    groups = ALPHABETS[alphabet]
    letters = "".join(sorted(groups))
    table = np.full(256, INVALID, dtype=np.uint8)
    for li, letter in enumerate(letters):
        for aa in groups[letter]:
            table[AMINO_ACIDS.index(aa)] = li
    return table, letters


def parse_setting(setting: str) -> tuple[str, int]:
    """'dayhoff:14' -> ('dayhoff', 14)."""
    alphabet, k = setting.split(":")
    if alphabet not in ALPHABETS:
        raise SystemExit(f"unknown alphabet {alphabet!r}")
    return alphabet, int(k)


def bucket_of(codes: np.ndarray, n_buckets: int) -> np.ndarray:
    """Hash bucket of each uint64 code (top bits of a multiplicative hash)."""
    shift = np.uint64(64 - int(math.log2(n_buckets)))
    return ((codes * _HASH_MULT) >> shift).astype(np.int64)


# ---------------------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------------------


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    """Accessions (the second |-field of a UniProt header) and sequences."""
    opener = gzip.open if str(path).endswith(".gz") else open
    names, seqs, chunks = [], [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if names:
                    seqs.append("".join(chunks))
                    chunks = []
                header = line[1:].split()[0]
                parts = header.split("|")
                names.append(parts[1] if len(parts) >= 3 else header)
            else:
                chunks.append(line.strip())
    if names:
        seqs.append("".join(chunks))
    return names, seqs


def concat_proteome(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One residue-index array with an INVALID separator after each protein.

    Returns the array and the start offset of each protein in it. A window that crosses a
    separator contains INVALID, so no anchor or target spans two proteins.
    """
    joined = "\x00".join(s.upper() for s in seqs) + "\x00"
    raw = np.frombuffer(joined.encode("ascii", errors="replace"), dtype=np.uint8)
    aa = _AA_INDEX[raw]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]])
    return aa, starts


def window_all(flags: np.ndarray, width: int) -> np.ndarray:
    """out[i] is True when flags[i : i + width] are all True; length len(flags) - width + 1."""
    n = len(flags) - width + 1
    if n <= 0:
        return np.zeros(0, dtype=bool)
    c = np.concatenate([[0], np.cumsum(flags.astype(np.int64))])
    return (c[width:] - c[:-width]) == width


def anchor_codes(aa: np.ndarray, alphabet: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Code of the reduced k-mer starting at every position, and whether it is usable.

    Usable means all k residues are among the 20 standard ones and the window is not
    removed by kmerseek 0.4.0's low-complexity mask (src/rust/sketch.rs, add_protein with
    remove_low_complexity on): a window whose raw residues are all the same letter is
    dropped for every alphabet, and a window whose encoded letters are all the same is
    dropped too for the HP alphabet only (`is_hp_moltype`), not for Dayhoff.
    """
    table, letters = reduced_lookup(alphabet)
    base = np.uint64(len(letters))
    red = table[aa]
    valid_res = aa != INVALID
    n = len(aa) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=bool)
    usable = window_all(valid_res, k)
    raw_homo = window_all(np.concatenate([aa[1:] == aa[:-1], [False]]), k - 1)[:n]
    usable &= ~raw_homo
    if alphabet == "hp":
        enc_homo = window_all(np.concatenate([red[1:] == red[:-1], [False]]), k - 1)[:n]
        usable &= ~enc_homo
    red64 = np.where(red == INVALID, 0, red).astype(np.uint64)
    codes = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        codes = codes * base + red64[j : j + n]
    return codes, usable


def target_codes(aa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-20 code of the 10 residues starting at every position, and whether all 10 are valid."""
    n = len(aa) - TARGET_LENGTH + 1
    valid = window_all(aa != INVALID, TARGET_LENGTH)
    a64 = np.where(aa == INVALID, 0, aa).astype(np.uint64)
    codes = np.zeros(n, dtype=np.uint64)
    for j in range(TARGET_LENGTH):
        codes = codes * np.uint64(TARGET_BASE) + a64[j : j + n]
    return codes, valid


def decode_anchor(code: int, alphabet: str, k: int) -> str:
    _, letters = reduced_lookup(alphabet)
    b = len(letters)
    out = []
    for _ in range(k):
        out.append(letters[code % b])
        code //= b
    return "".join(reversed(out))


def decode_target(code: int) -> str:
    out = []
    for _ in range(TARGET_LENGTH):
        out.append(AMINO_ACIDS[code % TARGET_BASE])
        code //= TARGET_BASE
    return "".join(reversed(out))


# ---------------------------------------------------------------------------------------
# select: species and classes
# ---------------------------------------------------------------------------------------

RANKS_KEPT = ("domain", "kingdom", "phylum", "class", "order", "family")


def read_taxdump(nodes: str, names: str) -> tuple[dict, dict, dict]:
    parent, rank, sci = {}, {}, {}
    with open(nodes) as fh:
        for line in fh:
            f = line.split("\t|\t")
            tid = int(f[0])
            parent[tid] = int(f[1])
            rank[tid] = f[2]
    with open(names) as fh:
        for line in fh:
            f = line.rstrip("\t|\n").split("\t|\t")
            if f[3] == "scientific name":
                sci[int(f[0])] = f[1]
    return parent, rank, sci


def ranked_lineage(tid: int, parent: dict, rank: dict, sci: dict) -> dict:
    """Name and taxid at each rank in RANKS_KEPT. NCBI renamed superkingdom to domain in 2025
    and uses realm for viruses; all three are stored as `domain`."""
    out = {}
    seen = 0
    while tid in parent and seen < 200:
        r = rank.get(tid, "")
        r = "domain" if r in ("superkingdom", "domain", "realm") else r
        if r in RANKS_KEPT and r not in out:
            out[r] = sci.get(tid, str(tid))
            out[f"{r}_taxid"] = tid
        if parent[tid] == tid:
            break
        tid = parent[tid]
        seen += 1
    return out


def busco_complete(s: str | None) -> float:
    """'C:99.6%[S:97.4%,D:2.2%],F:...' -> 99.6; missing -> -1."""
    if not s or not s.startswith("C:"):
        return -1.0
    try:
        return float(s[2:].split("%")[0])
    except ValueError:
        return -1.0


def cmd_select(a: argparse.Namespace) -> None:
    proteomes = pl.read_csv(a.proteomes, separator="\t", infer_schema_length=0)
    proteomes = proteomes.rename({"Proteome Id": "upid", "Organism Id": "taxid", "Organism": "organism",
                                  "Protein count": "protein_count", "BUSCO": "busco", "CPD": "cpd"})
    readme = parse_reference_readme(a.readme)
    parent, rank, sci = read_taxdump(a.nodes, a.names)
    rows = []
    n_in = proteomes.height
    dropped = defaultdict(int)
    for r in proteomes.iter_rows(named=True):
        if r["upid"] not in readme:
            dropped["not in FTP README"] += 1
            continue
        # CPD (UniProt's Complete Proteome Detector) is "Outlier (high value)" or
        # "Outlier (low value)" for a proteome whose size is far from its relatives'.
        if (r["cpd"] or "").strip().lower().startswith("outlier"):
            dropped["CPD outlier"] += 1
            continue
        tid = int(r["taxid"])
        lin = ranked_lineage(tid, parent, rank, sci)
        if tid not in parent:
            dropped["taxid not in NCBI taxonomy"] += 1
            continue
        if "class" not in lin or "family" not in lin:
            dropped["no class or no family"] += 1
            continue
        rows.append({
            "upid": r["upid"], "taxid": tid, "organism": r["organism"],
            "protein_count": int(r["protein_count"] or 0),
            "busco_complete": busco_complete(r["busco"]), "cpd": r["cpd"],
            "ftp_superregnum": readme[r["upid"]]["superregnum"],
            **{k: lin.get(k) for k in RANKS_KEPT},
            **{f"{k}_taxid": lin.get(f"{k}_taxid") for k in RANKS_KEPT},
        })
    df = pl.DataFrame(rows)
    # One species per family: highest BUSCO completeness, then CPD 'Standard', then the
    # most proteins, then the smallest proteome id, so the choice is reproducible.
    df = df.with_columns((pl.col("cpd").str.to_lowercase() == "standard").alias("_std"))
    df = df.sort(["family_taxid", "busco_complete", "_std", "protein_count", "upid"],
                 descending=[False, True, True, True, False])
    one = df.group_by("family_taxid", maintain_order=True).first().drop("_std")
    per_class = one.group_by("class_taxid").len()
    keep = per_class.filter(pl.col("len") >= MIN_SPECIES_PER_CLADE)["class_taxid"]
    sel = one.filter(pl.col("class_taxid").is_in(keep.implode()))
    if a.max_per_class:
        # Pilot only: the first N species (by proteome id) of the M classes with the most
        # species, which mixes large eukaryote proteomes with bacteria and viruses.
        top = sel.group_by("class_taxid").len().sort(["len", "class_taxid"], descending=[True, False])
        classes = top["class_taxid"].head(a.max_classes).to_list()
        sel = (sel.filter(pl.col("class_taxid").is_in(classes))
               .sort(["class_taxid", "upid"]).group_by("class_taxid", maintain_order=True).head(a.max_per_class))
    sel = sel.sort(["domain", "phylum", "class", "family", "upid"], nulls_last=True)
    sel = sel.with_row_index("species_idx").with_columns(pl.col("species_idx").cast(pl.Int32))
    sel = sel.with_columns(
        pl.struct(["upid", "taxid", "ftp_superregnum"]).map_elements(
            lambda s: f"{s['ftp_superregnum']}/{s['upid']}/{s['upid']}_{s['taxid']}.fasta.gz",
            return_dtype=pl.String).alias("ftp_path"))
    sel.write_csv(a.out, separator="\t")
    summary = {
        "n_reference_proteomes": n_in, "dropped": dict(dropped),
        "n_with_class_and_family": df.height, "n_families": one.height,
        "n_classes_any": per_class.height, "n_classes_kept": int(sel["class_taxid"].n_unique()),
        "n_species_selected": sel.height,
        "species_per_class": {c: int(n) for c, n in sel.group_by("class").len().sort("len", descending=True).iter_rows()},
        "pilot": bool(a.max_per_class),
    }
    Path(a.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "species_per_class"}, indent=2))


UNIPROT_PROTEOMES_URL = ("https://rest.uniprot.org/proteomes/stream?query=reference:true"
                         "&fields=upid,organism_id,organism,protein_count,busco,cpd&format=tsv")
NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"


def cmd_inputs(a: argparse.Namespace) -> None:
    """Download the reference-proteome list, the FTP README and the NCBI taxonomy, and record
    the UniProt release and the download date."""
    import tarfile
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UNIPROT_PROTEOMES_URL) as r:
        release = r.headers.get("X-UniProt-Release")
        (out / "reference_proteomes.tsv").write_bytes(r.read())
    readme_url = f"{a.base_url.rstrip('/')}/README"
    urllib.request.urlretrieve(readme_url, out / "README")
    urllib.request.urlretrieve(NCBI_TAXDUMP_URL, out / "new_taxdump.tar.gz")
    with tarfile.open(out / "new_taxdump.tar.gz") as t:
        for m in ("nodes.dmp", "names.dmp"):
            t.extract(m, out, filter="data")
    meta = {"uniprot_release_rest": release, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "proteomes_url": UNIPROT_PROTEOMES_URL, "readme_url": readme_url, "taxdump_url": NCBI_TAXDUMP_URL,
            "taxdump_md5": hashlib.md5((out / "new_taxdump.tar.gz").read_bytes()).hexdigest()}
    for line in (out / "README").read_text(errors="replace").splitlines()[:40]:
        if "Release" in line or "release" in line:
            meta.setdefault("readme_release_lines", []).append(line.strip())
    (out / "inputs.json").write_text(json.dumps(meta, indent=2))
    (out / "new_taxdump.tar.gz").unlink()
    print(json.dumps(meta, indent=2))


def parse_reference_readme(path: str) -> dict:
    """Proteome id -> superregnum folder, from the reference_proteomes README table."""
    out = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and f[0].startswith("UP") and f[0][2:].isdigit():
                out[f[0]] = {"taxid": f[1], "superregnum": f[3].strip().capitalize()}
    return out


# ---------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------


def cmd_fetch(a: argparse.Namespace) -> None:
    species = pl.read_csv(a.species, separator="\t").filter(pl.col("species_idx").is_in(a.idx))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = []
    for r in species.iter_rows(named=True):
        url = f"{a.base_url.rstrip('/')}/{r['ftp_path']}"
        dest = outdir / f"{r['species_idx']}.fasta.gz"
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(url, dest)
                names, seqs = read_fasta(dest)
                break
            except Exception as e:  # network or truncated gzip: try again, then fail loudly
                if attempt == 4:
                    raise SystemExit(f"{url}: {e}")
                time.sleep(10 * (attempt + 1))
        md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        stats.append({"species_idx": r["species_idx"], "upid": r["upid"], "n_proteins": len(seqs),
                      "n_residues": sum(len(s) for s in seqs), "md5": md5, "url": url})
    pl.DataFrame(stats).write_csv(a.stats, separator="\t")


# ---------------------------------------------------------------------------------------
# anchors and candidates (pass 1)
# ---------------------------------------------------------------------------------------


def cmd_anchors(a: argparse.Namespace) -> None:
    """Unique usable anchors of each species, split into N_ANCHOR_BUCKETS parquet files."""
    alphabet, k = parse_setting(a.setting)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    parts: dict[int, list[pl.DataFrame]] = defaultdict(list)
    letter_counts = np.zeros(len(ALPHABETS[alphabet]), dtype=np.int64)
    windows = {"examined": 0, "usable": 0}
    for fa in a.fastas:
        idx = int(Path(fa).name.split(".")[0])
        _, seqs = read_fasta(fa)
        aa, _ = concat_proteome(seqs)
        table, _ = reduced_lookup(alphabet)
        red = table[aa]
        letter_counts += np.bincount(red[red != INVALID], minlength=len(letter_counts))
        codes, usable = anchor_codes(aa, alphabet, k)
        windows["examined"] += int(window_all(aa != INVALID, k).sum())
        windows["usable"] += int(usable.sum())
        u = np.unique(codes[usable])
        b = bucket_of(u, N_ANCHOR_BUCKETS)
        order = np.argsort(b, kind="stable")
        u, b = u[order], b[order]
        cuts = np.searchsorted(b, np.arange(N_ANCHOR_BUCKETS + 1))
        for bi in range(N_ANCHOR_BUCKETS):
            if cuts[bi + 1] > cuts[bi]:
                parts[bi].append(pl.DataFrame({"anchor": u[cuts[bi]:cuts[bi + 1]],
                                               "species_idx": np.full(cuts[bi + 1] - cuts[bi], idx, dtype=np.int32)}))
    for bi in range(N_ANCHOR_BUCKETS):
        df = pl.concat(parts[bi]) if parts[bi] else pl.DataFrame(schema={"anchor": pl.UInt64, "species_idx": pl.Int32})
        df.write_parquet(out / f"b{bi:03d}.parquet")
    Path(a.stats).write_text(json.dumps({"setting": a.setting, "letter_counts": letter_counts.tolist(), **windows}))


def cmd_candidates(a: argparse.Namespace) -> None:
    """Anchors present in at least MIN_SPECIES_FOR_ANY_LABELLING species, for some buckets.

    Buckets are done one at a time, so memory is that of the largest bucket."""
    kept_all, hists = [], []
    for bucket in a.bucket:
        files = [str(Path(d) / f"b{bucket:03d}.parquet") for d in a.dirs]
        df = pl.concat([pl.read_parquet(f) for f in files])
        g = df.group_by("anchor").agg(pl.col("species_idx").n_unique().alias("n_species"))
        hists.append(g.group_by("n_species").len())
        kept_all.append(g.filter(pl.col("n_species") >= MIN_SPECIES_FOR_ANY_LABELLING)["anchor"].to_numpy())
        del df, g
    hist = pl.concat(hists).group_by("n_species").agg(pl.col("len").sum()).sort("n_species")
    np.save(a.out, np.sort(np.concatenate(kept_all)).astype(np.uint64))
    hist.write_parquet(a.hist)


# ---------------------------------------------------------------------------------------
# targets (pass 2)
# ---------------------------------------------------------------------------------------


def cmd_targets(a: argparse.Namespace) -> None:
    """Every occurrence of a candidate anchor and its target at each gap, bucketed by anchor."""
    alphabet, k = parse_setting(a.setting)
    cand = np.sort(np.concatenate([np.load(f) for f in a.candidates]))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    parts: dict[int, list[pl.DataFrame]] = defaultdict(list)
    accessions = []
    for fa in a.fastas:
        idx = int(Path(fa).name.split(".")[0])
        names, seqs = read_fasta(fa)
        aa, starts = concat_proteome(seqs)
        codes, usable = anchor_codes(aa, alphabet, k)
        hit = np.zeros(len(codes), dtype=bool)
        if len(cand):
            j = np.searchsorted(cand, codes)
            j[j == len(cand)] = 0
            hit = (cand[j] == codes) & usable
        pos = np.nonzero(hit)[0]
        tcodes, tvalid = target_codes(aa)
        prot = np.searchsorted(starts, pos, side="right") - 1
        for gap in a.gaps:
            tstart = pos + k + gap
            ok = tstart < len(tcodes)
            ok[ok] &= tvalid[tstart[ok]]
            p = pos[ok]
            if not len(p):
                continue
            df = pl.DataFrame({
                "anchor": codes[p], "gap": np.full(len(p), gap, dtype=np.int16),
                "species_idx": np.full(len(p), idx, dtype=np.int32),
                "target": tcodes[tstart[ok]], "protein_idx": prot[ok].astype(np.int32),
                "pos0": (p - starts[prot[ok]]).astype(np.int32),
            })
            b = bucket_of(df["anchor"].to_numpy(), N_TARGET_BUCKETS)
            df = df.with_columns(pl.Series("bucket", b))
            for (bi,), sub in df.partition_by("bucket", as_dict=True).items():
                parts[int(bi)].append(sub.drop("bucket"))
        accessions.append(pl.DataFrame({"species_idx": np.full(len(names), idx, dtype=np.int32),
                                        "protein_idx": np.arange(len(names), dtype=np.int32),
                                        "accession": names}))
    schema = {"anchor": pl.UInt64, "gap": pl.Int16, "species_idx": pl.Int32, "target": pl.UInt64,
              "protein_idx": pl.Int32, "pos0": pl.Int32}
    for bi in range(N_TARGET_BUCKETS):
        df = pl.concat(parts[bi]) if parts[bi] else pl.DataFrame(schema=schema)
        df.write_parquet(out / f"b{bi:02d}.parquet")
    pl.concat(accessions).write_parquet(a.accessions)


# ---------------------------------------------------------------------------------------
# labellings
# ---------------------------------------------------------------------------------------


def labellings(species: pl.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Class label of each species_idx under the real labels and the two shuffles.

    shuffle_random permutes the class labels of all species. shuffle_tree permutes them only
    among species of the same phylum (or, without a phylum, the same kingdom, else domain),
    so a label moves only between sister classes and every column keeps its size.
    """
    sp = species.sort("species_idx")
    real = sp["class"].to_numpy().astype(object)
    rng = np.random.default_rng(seed)
    rand = real[rng.permutation(len(real))]
    group = (sp["phylum"].fill_null(sp["kingdom"]).fill_null(sp["domain"]).fill_null("none")).to_numpy()
    tree = real.copy()
    rng2 = np.random.default_rng(seed + 1)
    for gval in np.unique(group):
        ix = np.nonzero(group == gval)[0]
        tree[ix] = real[ix[rng2.permutation(len(ix))]]
    n_max = int(sp["species_idx"].max()) + 1
    out = {}
    for name, lab in (("real", real), ("shuffle_random", rand), ("shuffle_tree", tree)):
        full = np.empty(n_max, dtype=object)
        full[sp["species_idx"].to_numpy()] = lab
        out[name] = full
    return out


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------


def load_oasis(path: str) -> types.ModuleType:
    """Import refresh-bio/SPLASH's OASIS.py unchanged.

    OASIS.py imports scipy at the top but only calls it in the asymptotic p-value
    (scipy.stats.norm.cdf) and in computeChi2Test, neither of which this pipeline calls. The
    pipeline image has no scipy, so a stand-in module that raises on any use is installed
    first: a call to scipy would fail loudly rather than give a wrong number.
    """
    if "scipy" not in sys.modules:
        try:
            import scipy  # noqa: F401
        except ImportError:
            def _refuse(*_a, **_k):
                raise RuntimeError("scipy is not available; only the finite-sample OASIS p-value is used")
            stub = types.ModuleType("scipy")
            stub.stats = types.SimpleNamespace(norm=types.SimpleNamespace(cdf=_refuse),
                                               contingency=types.SimpleNamespace(chi2_contingency=_refuse))
            sys.modules["scipy"] = stub
            sys.modules["scipy.stats"] = stub.stats
    spec_name = "oasis_splash"
    import importlib.util
    spec = importlib.util.spec_from_file_location(spec_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution, Q(df/2, x/2), the regularized upper
    incomplete gamma function. Series for x < a + 1, Lentz continued fraction otherwise
    (Press et al., Numerical Recipes, 3rd ed., section 6.2). Tested against scipy."""
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if z < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10_000):
            ap += 1.0
            d *= z / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - gln))
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


def pearson_chi2(table: np.ndarray) -> tuple[float, int, float]:
    """Pearson's chi-square of independence after dropping empty rows and columns."""
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0].astype(float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0, 0, 1.0
    e = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    x2 = float(((t - e) ** 2 / e).sum())
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return x2, df, chi2_sf(x2, df)


def composition_table(targets: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """20 x J residue counts: each species-target pair adds its 10 residues to its column.

    `targets` holds the target codes (rows of the count table), `counts` the I x J table.
    """
    comp = np.zeros((TARGET_BASE, counts.shape[1]), dtype=np.int64)
    for i, code in enumerate(targets):
        c = int(code)
        for _ in range(TARGET_LENGTH):
            comp[c % TARGET_BASE] += counts[i]
            c //= TARGET_BASE
    return comp


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def entropy_bits(code: int, alphabet: str, k: int) -> float:
    """Shannon entropy of the reduced letters of one anchor, in bits per residue."""
    s = decode_anchor(code, alphabet, k)
    counts = np.array([s.count(c) for c in set(s)], dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def cmd_test(a: argparse.Namespace) -> None:
    """Count tables and tests for one anchor bucket of one setting, every gap and labelling."""
    alphabet, k = parse_setting(a.setting)
    oasis = load_oasis(a.oasis)
    species = pl.read_csv(a.species, separator="\t")
    labs = labellings(species, a.seed)
    files = [str(Path(d) / f"b{a.bucket:02d}.parquet") for d in a.dirs]
    occ = pl.concat([pl.read_parquet(f) for f in files])
    # A species counts once per (anchor, target), however many proteins carry it.
    pairs = occ.select("anchor", "gap", "species_idx", "target").unique()
    rows = []
    for lab_name in LABELLINGS:
        lab = labs[lab_name]
        df = pairs.with_columns(pl.Series("clade", lab[pairs["species_idx"].to_numpy()].astype(str)))
        carry = df.select("anchor", "gap", "species_idx", "clade").unique()
        per_clade = carry.group_by("anchor", "gap", "clade").len("n_species")
        good = per_clade.filter(pl.col("n_species") >= MIN_SPECIES_PER_CLADE)
        n_good = good.group_by("anchor", "gap").len("n_clades").filter(pl.col("n_clades") >= MIN_CLADES)
        kept = good.join(n_good.select("anchor", "gap"), on=["anchor", "gap"])
        tab = (df.join(kept.select("anchor", "gap", "clade"), on=["anchor", "gap", "clade"])
               .group_by("anchor", "gap", "target", "clade").len("n"))
        # Distinct species in the kept classes. The table total is larger whenever a species
        # carries the anchor with more than one target (paralogs, repeats).
        n_carrying = {(an, g): n for an, g, n in carry.join(kept.select("anchor", "gap", "clade"),
                                                              on=["anchor", "gap", "clade"])
                      .group_by("anchor", "gap").agg(pl.col("species_idx").n_unique()).iter_rows()}
        for (anchor, gap), sub in tab.partition_by(["anchor", "gap"], as_dict=True).items():
            wide = sub.pivot(on="clade", index="target", values="n").fill_null(0).sort("target")
            clades = sorted(c for c in wide.columns if c != "target")
            x = wide.select(clades).to_numpy().astype(np.int64)
            tcodes = wide["target"].to_numpy()
            # return_f_c also gives the row and column vectors the p-value used, which the
            # effect size needs.
            p, f, c = oasis.OASIS_pvalue(x, return_f_c=True)
            p = float(p)
            eff = float(oasis.effectSize_bin(x, np.asarray(c), np.asarray(f)))
            x2, dfree, pc = pearson_chi2(composition_table(tcodes, x))
            rows.append({
                "setting": a.setting, "labelling": lab_name, "gap": int(gap), "anchor": int(anchor),
                "n_species": int(n_carrying[(anchor, gap)]), "table_total": int(x.sum()),
                "n_clades": len(clades), "n_targets": x.shape[0],
                "p_oasis": p, "effect_size": eff, "comp_chi2": x2, "comp_df": dfree, "p_comp": pc,
                "clades": ";".join(clades),
                # OASIS's column weights c (one per class, same order as `clades`): the
                # classes at the two ends are the ones the targets split.
                "clade_weights": ";".join(f"{v:.4f}" for v in np.asarray(c, dtype=float)),
            })
    schema = {"setting": pl.String, "labelling": pl.String, "gap": pl.Int64, "anchor": pl.UInt64,
              "n_species": pl.Int64, "table_total": pl.Int64, "n_clades": pl.Int64, "n_targets": pl.Int64, "p_oasis": pl.Float64,
              "effect_size": pl.Float64, "comp_chi2": pl.Float64, "comp_df": pl.Int64, "p_comp": pl.Float64,
              "clades": pl.String, "clade_weights": pl.String}
    out = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    out.write_parquet(a.out)


def cmd_collect(a: argparse.Namespace) -> None:
    """q-values per (setting, gap, labelling); the significant anchors; the counts."""
    df = pl.concat([pl.read_parquet(f) for f in a.inputs])
    parts = []
    for _, sub in df.group_by("setting", "gap", "labelling", maintain_order=True):
        sub = sub.with_columns(pl.Series("q_oasis", bh(sub["p_oasis"].to_numpy())))
        parts.append(sub)
    df = pl.concat(parts)
    sig = df.filter(pl.col("q_oasis") < a.alpha)
    # Composition scan: BH over the significant anchors of each (setting, gap, labelling).
    cparts = []
    for _, sub in sig.group_by("setting", "gap", "labelling", maintain_order=True):
        cparts.append(sub.with_columns(pl.Series("q_comp", bh(sub["p_comp"].to_numpy()))))
    sig = pl.concat(cparts) if cparts else sig.with_columns(pl.lit(None, pl.Float64).alias("q_comp"))
    sig = sig.with_columns(
        pl.struct(["anchor", "setting"]).map_elements(
            lambda s: decode_anchor(int(s["anchor"]), *parse_setting(s["setting"])), return_dtype=pl.String
        ).alias("anchor_letters"))
    sig = sig.with_columns(
        pl.struct(["anchor", "setting"]).map_elements(
            lambda s: entropy_bits(int(s["anchor"]), *parse_setting(s["setting"])), return_dtype=pl.Float64
        ).alias("anchor_entropy_bits"))
    df.write_parquet(a.all_out)
    sig.write_parquet(a.sig_out)
    counts = (df.group_by("setting", "gap", "labelling")
              .agg(pl.len().alias("n_anchors_tested"),
                   (pl.col("q_oasis") < a.alpha).sum().alias("n_significant"))
              .sort("setting", "gap", "labelling"))
    counts.write_csv(a.counts_out, separator="\t")
    print(counts)


# ---------------------------------------------------------------------------------------
# occurrences of significant anchors, Pfam overlap, alignment control
# ---------------------------------------------------------------------------------------


def cmd_occurrences(a: argparse.Namespace) -> None:
    """Rows of the significant anchors (real labels) from one target bucket, with residues.

    Writes the occurrences, and a FASTA of every protein that carries one, for hmmscan.
    """
    sig = pl.read_parquet(a.significant).filter(pl.col("labelling") == "real").select("setting", "gap", "anchor").unique()
    sig = sig.filter(pl.col("setting") == a.setting)
    alphabet, k = parse_setting(a.setting)
    files = [str(Path(d) / f"b{a.bucket:02d}.parquet") for d in a.dirs]
    occ = pl.concat([pl.read_parquet(f) for f in files]).join(sig.drop("setting"), on=["anchor", "gap"])
    acc = pl.concat([pl.read_parquet(f) for f in a.accessions])
    occ = occ.join(acc, on=["species_idx", "protein_idx"], how="left").with_columns(pl.col("accession").cast(pl.String))
    seqs_needed = occ.select("species_idx", "protein_idx").unique()
    by_species = defaultdict(set)
    for s, p in seqs_needed.iter_rows():
        by_species[s].add(p)
    fasta_of = {int(p.name.split(".")[0]): p for d in a.fasta_dirs for p in Path(d).glob("*.fasta.gz")}
    seq_of = {}
    for s, prots in by_species.items():
        names, seqs = read_fasta(fasta_of[s])
        for p in prots:
            seq_of[(s, p)] = (names[p], seqs[p])
    window = []
    for s, p, pos0, gap in occ.select("species_idx", "protein_idx", "pos0", "gap").iter_rows():
        seq = seq_of[(s, p)][1]
        window.append((seq[pos0:pos0 + k], seq[pos0 + k:pos0 + k + gap], seq[pos0 + k + gap:pos0 + k + gap + TARGET_LENGTH],
                       len(seq)))
    occ = occ.with_columns(
        pl.Series("anchor_residues", [w[0] for w in window], dtype=pl.String),
        pl.Series("gap_residues", [w[1] for w in window], dtype=pl.String),
        pl.Series("target_residues", [w[2] for w in window], dtype=pl.String),
        pl.Series("protein_length", [w[3] for w in window], dtype=pl.Int32),
        pl.lit(a.setting).alias("setting"),
    )
    occ.write_parquet(a.out)
    with open(a.fasta_out, "w") as fh:
        for (s, p), (name, seq) in sorted(seq_of.items()):
            fh.write(f">s{s}_p{p} {name}\n{seq}\n")


# ---------------------------------------------------------------------------------------
# residue totals and the k arithmetic
# ---------------------------------------------------------------------------------------


def cmd_totals(a: argparse.Namespace) -> None:
    """Residue and protein totals of the species set, and reduced-letter frequencies."""
    stats = pl.concat([pl.read_csv(f, separator="\t") for f in a.fetch_stats])
    letters = defaultdict(lambda: None)
    for f in a.anchor_stats:
        d = json.loads(Path(f).read_text())
        cur = letters[d["setting"]]
        letters[d["setting"]] = np.array(d["letter_counts"]) if cur is None else cur + np.array(d["letter_counts"])
    out = {"n_species": stats.height, "n_proteins": int(stats["n_proteins"].sum()),
           "n_residues": int(stats["n_residues"].sum()),
           "letter_counts": {s: v.tolist() for s, v in letters.items()},
           "letters": {s: reduced_lookup(parse_setting(s)[0])[1] for s in letters}}
    stats.write_csv(a.stats_out, separator="\t")
    Path(a.out).write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inputs")
    s.add_argument("--base-url", required=True)
    s.add_argument("--outdir", required=True)
    s.set_defaults(func=cmd_inputs)

    s = sub.add_parser("select")
    s.add_argument("--proteomes", required=True)
    s.add_argument("--readme", required=True)
    s.add_argument("--nodes", required=True)
    s.add_argument("--names", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--summary", required=True)
    s.add_argument("--max-per-class", type=int, default=0)
    s.add_argument("--max-classes", type=int, default=0)
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("fetch")
    s.add_argument("--species", required=True)
    s.add_argument("--idx", type=int, nargs="+", required=True)
    s.add_argument("--base-url", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--stats", required=True)
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("anchors")
    s.add_argument("--setting", required=True)
    s.add_argument("--fastas", nargs="+", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--stats", required=True)
    s.set_defaults(func=cmd_anchors)

    s = sub.add_parser("candidates")
    s.add_argument("--bucket", type=int, nargs="+", required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--hist", required=True)
    s.set_defaults(func=cmd_candidates)

    s = sub.add_parser("targets")
    s.add_argument("--setting", required=True)
    s.add_argument("--gaps", type=int, nargs="+", required=True)
    s.add_argument("--candidates", nargs="+", required=True)
    s.add_argument("--fastas", nargs="+", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--accessions", required=True)
    s.set_defaults(func=cmd_targets)

    s = sub.add_parser("test")
    s.add_argument("--setting", required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--species", required=True)
    s.add_argument("--oasis", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("collect")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--all-out", required=True)
    s.add_argument("--sig-out", required=True)
    s.add_argument("--counts-out", required=True)
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("occurrences")
    s.add_argument("--setting", required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--accessions", nargs="+", required=True)
    s.add_argument("--significant", required=True)
    s.add_argument("--fasta-dirs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fasta-out", required=True)
    s.set_defaults(func=cmd_occurrences)

    s = sub.add_parser("totals")
    s.add_argument("--fetch-stats", nargs="+", required=True)
    s.add_argument("--anchor-stats", nargs="+", required=True)
    s.add_argument("--stats-out", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_totals)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
