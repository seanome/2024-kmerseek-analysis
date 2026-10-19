"""Seedless local alignment benchmark on ice-binding proteins (notebook 349).

Asks whether Smith-Waterman alignment with affine gaps, in the full amino-acid alphabet,
kmerseek's default hydrophobic/polar alphabet (hp_lehninger2) and Dayhoff's six classes,
ranks ice-binding proteins of a separate evolutionary origin above real control proteins
and above shuffled copies of the same proteins.

Steps, each one a subcommand, all driven by scripts/ice_binding_sw_config.toml:

    manifest  sequence manifest of the truth set: mature boundaries, origin, evidence
    panel     controls (matched background, hard controls, shuffles), dedupe, splits
    align     every query against every target of one split, for the grid or the selection
    select    calibrate on development nulls and pick one setting per alphabet
    evaluate  metrics, control and shuffle comparisons, subgroups, decision, alignments
    all       the five in order

Run with the project interpreter:

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
        scripts/ice_binding_sw_benchmark.py all
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import resource
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "scripts" / "ice_binding_sw_config.toml"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
# Names that mark a protein as ice-related; such entries never serve as controls.
ICE_NAME = re.compile(
    r"antifreeze|ice[- ]binding|ice[- ]structuring|ice[- ]nucleation|thermal hysteresis",
    re.IGNORECASE,
)
# Class letters used for the reduced alphabets, by class index from alphabet_classes.tsv.
CLASS_LETTERS = {"hp_lehninger2": "HP", "dayhoff6": "abcdef"}
REAL_NEGATIVE_STATUS = "unverified_background"


# --------------------------------------------------------------------------------------
# Config, logging of runtime and memory
# --------------------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Read the TOML config and resolve repo-relative paths."""
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_path"] = str(path)
    cfg["_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    cfg["_out"] = REPO / cfg["inputs"]["out_dir"]
    return cfg


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (macOS reports bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def log_runtime(cfg: dict, step: str, seconds: float, detail: str = "") -> None:
    """Append one row to runtime.tsv: step, wall seconds, peak RSS so far."""
    path = cfg["_out"] / "runtime.tsv"
    new = not path.exists()
    with open(path, "a") as fh:
        if new:
            fh.write("step\tdetail\twall_seconds\tpeak_rss_mb\tfinished_at\n")
        fh.write(
            f"{step}\t{detail}\t{seconds:.2f}\t{peak_rss_mb():.1f}\t"
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )


# --------------------------------------------------------------------------------------
# Sequence helpers
# --------------------------------------------------------------------------------------


def composition(seq: str, letters: str = AMINO_ACIDS) -> np.ndarray:
    """Fraction of each letter in `letters`; letters outside are ignored."""
    counts = np.array([seq.count(a) for a in letters], dtype=float)
    total = counts.sum()
    return counts / total if total else counts


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two frequency vectors (0 = same, 1 = disjoint)."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def distinct_3mer_fraction(seq: str) -> float:
    """Distinct 3-residue words / all 3-residue words. Low = repetitive."""
    n = len(seq) - 2
    if n <= 0:
        return 1.0
    return len({seq[i : i + 3] for i in range(n)}) / n


def read_alphabets(path: Path) -> dict[str, dict[str, str]]:
    """Residue -> class letter for each reduced alphabet, plus identity for protein20."""
    t = pl.read_csv(path, separator="\t")
    out = {"protein20": {a: a for a in AMINO_ACIDS}}
    for name, letters in CLASS_LETTERS.items():
        sub = t.filter(pl.col("alphabet") == name)
        out[name] = {
            r: letters[i] for r, i in zip(sub["residue"], sub["class_index"])
        }
        if set(out[name]) != set(AMINO_ACIDS):
            raise ValueError(f"{name}: class table does not cover the 20 amino acids")
    return out


def translate(seq: str, mapping: dict[str, str]) -> str:
    """Map residues to class letters, dropping anything outside the 20 amino acids."""
    return "".join(mapping[a] for a in seq if a in mapping)


def clean(seq: str) -> str:
    """Keep only the 20 standard amino acids (X, B, Z, U, O dropped)."""
    return "".join(a for a in seq if a in AMINO_ACIDS)


def iter_fasta(path: Path):
    """Yield (accession, header, sequence) from a gzipped UniProt FASTA."""
    acc = header = None
    parts: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, header, "".join(parts)
                header = line[1:].rstrip()
                acc = header.split("|")[1]
                parts = []
            else:
                parts.append(line.strip())
    if acc is not None:
        yield acc, header, "".join(parts)


def header_fields(header: str) -> tuple[str, str]:
    """Protein name and organism from a UniProt FASTA header."""
    m = re.match(r"\S+ (.*?) OS=(.*?) OX=", header)
    return (m.group(1), m.group(2)) if m else (header, "")


def stable_fraction(key: str, seed: int) -> float:
    """A number in [0, 1) fixed by key and seed, independent of file order."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:12]
    return int(h, 16) / 16**12


# --------------------------------------------------------------------------------------
# UniProt fetches, cached with their release
# --------------------------------------------------------------------------------------


def fetch_uniprot(cfg: dict, name: str, query: str, fields: str) -> pl.DataFrame:
    """Stream a UniProt query to raw/<name>.tsv once; check the release on every read."""
    raw = cfg["_out"] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path, rel_path = raw / f"{name}.tsv", raw / f"{name}.tsv.release"
    url = f"{UNIPROT_STREAM}?" + urllib.parse.urlencode(
        {"query": query, "fields": fields, "format": "tsv"}
    )
    if not path.exists():
        with urllib.request.urlopen(url, timeout=300) as resp:
            body = resp.read().decode()
            release = resp.headers.get("X-UniProt-Release", "")
        if not body.startswith("Entry"):
            sys.exit(f"{name}: UniProt did not return a TSV table; first bytes {body[:80]!r}")
        path.write_text(body)
        rel_path.write_text(release)
        with open(raw / "fetch_log.tsv", "a") as fh:
            fh.write(f"{name}\t{query}\t{url}\t{time.strftime('%Y-%m-%d')}\t{release}\n")
    release = rel_path.read_text().strip()
    if release != cfg["inputs"]["uniprot_release"]:
        sys.exit(f"{name}: cached release {release!r}, config wants "
                 f"{cfg['inputs']['uniprot_release']}")
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


# --------------------------------------------------------------------------------------
# Step 1: manifest of the truth set
# --------------------------------------------------------------------------------------


@dataclass
class FlatRecord:
    """The parts of one UniProt flat-file record the manifest needs."""

    accession: str
    sequence: str
    features: list[tuple[str, str]]
    function_text: str


def parse_flat_file(path: Path) -> dict[str, FlatRecord]:
    """Accession -> sequence, FT location lines, and the FUNCTION comment."""
    out = {}
    for rec in path.read_text().split("\n//"):
        m = re.search(r"^AC   (\w+);", rec, re.MULTILINE)
        if not m:
            continue
        lines = rec.splitlines()
        seq = "".join(
            ln.replace(" ", "")
            for ln in lines[next(i for i, ln in enumerate(lines) if ln.startswith("SQ")) + 1 :]
            if ln.startswith("     ")
        )
        feats = re.findall(r"^FT   (\w+)\s+(\S+)$", rec, re.MULTILINE)
        func, inside = [], False
        for ln in lines:
            if ln.startswith("CC   -!- FUNCTION:"):
                inside = True
                func.append(ln[len("CC   -!- FUNCTION:") :])
            elif inside and ln.startswith("CC       "):
                func.append(ln[9:])
            elif inside:
                inside = False
        out[m.group(1)] = FlatRecord(m.group(1), seq, feats, " ".join(func).strip())
    return out


def parse_span(loc: str) -> tuple[int, int] | None:
    """'21..261' or '<1..5' -> (21, 261), 1-based inclusive; None for '?' ends."""
    m = re.match(r"[<>]?(\d+)\.\.[<>]?(\d+)$", loc)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"(\d+)$", loc)
    return (int(m.group(1)), int(m.group(1))) if m else None


def mature_span(rec: FlatRecord) -> tuple[int, int, str]:
    """The region aligned: the longest CHAIN, else the span of all PEPTIDEs, else all.

    Signal peptides and propeptides are outside every CHAIN in these records, so the
    longest CHAIN is the mature protein. Returns 1-based inclusive start, end, and source.
    """
    chains = [parse_span(loc) for kind, loc in rec.features if kind == "CHAIN"]
    chains = [c for c in chains if c]
    if chains:
        s, e = max(chains, key=lambda c: c[1] - c[0])
        return s, e, "longest CHAIN"
    peps = [parse_span(loc) for kind, loc in rec.features if kind == "PEPTIDE"]
    peps = [p for p in peps if p]
    if peps:
        return min(p[0] for p in peps), max(p[1] for p in peps), "span of PEPTIDE features"
    return 1, len(rec.sequence), "whole sequence (no CHAIN or PEPTIDE)"


def assay_terms(function_text: str) -> str:
    """Assays named in the UniProt FUNCTION comment; '' when it names none."""
    t = function_text.lower()
    found = []
    if "thermal hysteresis" in t:
        found.append("thermal hysteresis")
    if "recrystallization" in t:
        found.append("ice recrystallization inhibition")
    if re.search(r"ice crystal|surface of ice|binds? (to )?ice", t):
        found.append("ice-crystal binding or growth")
    if "freezing point" in t:
        found.append("freezing-point depression")
    return "; ".join(found)


def taxon_family(lineage: str) -> str:
    """The '(family)' rank from a UniProt lineage string."""
    for part in lineage.split(", "):
        if part.endswith("(family)"):
            return part.removesuffix(" (family)")
    return ""


def assign_origin(row: dict, origins: pl.DataFrame) -> dict:
    """The one origin rule every non-empty field of which matches; else unresolved."""
    hits = []
    for o in origins.iter_rows(named=True):
        if o["uniprot_family"] and row["uniprot_family"] != o["uniprot_family"]:
            continue
        if o["taxon_family"] and not re.fullmatch(o["taxon_family"], row["taxon_family"]):
            continue
        if o["name_regex"] and not re.search(o["name_regex"], row["protein_name"]):
            continue
        hits.append(o)
    if len(hits) > 1:
        raise ValueError(f"{row['accession']}: matches origins {[h['origin_id'] for h in hits]}")
    if not hits:
        return {
            "origin_id": "unresolved",
            "status": "positive_unresolved_origin",
            "origin_evidence": "no origin rule matches",
            "origin_citation_doi": "",
            "origin_note": "left out of every metric",
        }
    o = hits[0]
    return {
        "origin_id": o["origin_id"],
        "status": o["positive_status"],
        "origin_evidence": o["origin_evidence"],
        "origin_citation_doi": o["citation_doi"],
        "origin_note": o["note"],
    }


def step_manifest(cfg: dict) -> pl.DataFrame:
    """Write truth_manifest.tsv: one row per truth-set entry, mature sequence and labels."""
    truth = pl.read_csv(REPO / cfg["inputs"]["truth_set"], separator="\t", infer_schema_length=0)
    flat = parse_flat_file(REPO / cfg["inputs"]["truth_flat_file"])
    origins = pl.read_csv(
        cfg["_out"] / "origins.tsv", separator="\t", infer_schema_length=0
    ).fill_null("")
    rows = []
    for t in truth.iter_rows(named=True):
        rec = flat[t["Entry"]]
        if len(rec.sequence) != int(t["Length"]):
            raise ValueError(f"{t['Entry']}: flat-file length differs from the truth table")
        s, e, src = mature_span(rec)
        func = rec.function_text
        row = {
            "seq_id": t["Entry"],
            "accession": t["Entry"],
            "kind": "real",
            "protein_name": t["Protein names"],
            "species": t["Organism"],
            "taxon_family": taxon_family(t["Taxonomic lineage"]),
            "uniprot_family": t["Protein families"] or "",
            "truth_lineage_label": t["lineage"],
            "full_length": len(rec.sequence),
            "mature_start": s,
            "mature_end": e,
            "boundary_source": src,
            "sequence": clean(rec.sequence[s - 1 : e]),
            "n_experimental_evidence_lines": int(t["n_experimental_evidence"]),
            "function_has_experimental_eco": "ECO:0000269" in func,
            "ice_binding_evidence": (
                "experimental (ECO:0000269 on the FUNCTION comment)"
                if "ECO:0000269" in func
                else ("annotation, no experimental code on FUNCTION" if func else
                      "keyword or name only, no FUNCTION comment")
            ),
            "assay_named_in_function": assay_terms(func),
            "function_pubmed": ";".join(sorted(set(re.findall(r"PubMed:(\d+)", func)))),
            "evidence_source": f"UniProt {cfg['inputs']['uniprot_release']} flat file",
        }
        row.update(assign_origin(row, origins))
        rows.append(row)
    man = pl.DataFrame(rows).sort("origin_id", "accession")
    man.write_csv(cfg["_out"] / "truth_manifest.tsv", separator="\t")
    return man


# --------------------------------------------------------------------------------------
# Step 2: controls, dedupe and splits
# --------------------------------------------------------------------------------------


def identity_coverage(aligner: Align.PairwiseAligner, a: str, b: str,
                      with_columns: bool = False):
    """Identity over alignment columns, and the aligned span of the shorter sequence over
    its length; with_columns also returns the number of alignment columns."""
    if not a or not b or (aln := best_alignment(aligner, a, b)) is None:
        return (0.0, 0.0, 0) if with_columns else (0.0, 0.0)
    blocks = ident = 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        blocks += a1 - a0
        ident += sum(x == y for x, y in zip(a[a0:a1], b[b0:b1]))
    c = aln.coordinates
    span_a, span_b = int(c[0, -1] - c[0, 0]), int(c[1, -1] - c[1, 0])
    cols = span_a + span_b - blocks
    cov = (span_a / len(a)) if len(a) <= len(b) else (span_b / len(b))
    if with_columns:
        return (ident / cols if cols else 0.0), float(cov), cols
    return (ident / cols if cols else 0.0), float(cov)


def best_alignment(aligner: Align.PairwiseAligner, a: str, b: str):
    """The first optimal local alignment, or None when no pair of residues scores above 0."""
    try:
        return aligner.align(a, b)[0]
    except IndexError:
        return None


def blosum62_aligner() -> Align.PairwiseAligner:
    """BLOSUM62, gap open 11 extend 1 (NCBI convention), local: used for dedupe."""
    return make_aligner(substitution_matrices.load("BLOSUM62"), 11, 1)


def make_aligner(matrix, gap_open: float, gap_extend: float) -> Align.PairwiseAligner:
    """Local aligner; a gap of length L costs gap_open + L * gap_extend."""
    return Align.PairwiseAligner(
        mode="local",
        substitution_matrix=matrix,
        open_gap_score=-(gap_open + gap_extend),
        extend_gap_score=-gap_extend,
    )


def exclusion_accessions(cfg: dict) -> set[str]:
    """Reviewed entries that are, or may be, ice-related: never used as controls."""
    q = (
        'reviewed:true AND (keyword:KW-0047 OR go:0050825 OR '
        'family:"type-I AFP family" OR family:"type-III AFP family" OR '
        'family:"ice-binding protein family" OR family:"thermal hysteresis protein family" OR '
        'family:"insect antifreeze protein family" OR protein_name:antifreeze OR '
        'protein_name:"ice-binding" OR protein_name:"ice binding" OR '
        'protein_name:"ice-structuring" OR protein_name:"thermal hysteresis" OR '
        'protein_name:"ice nucleation" OR protein_name:"ice-nucleation")'
    )
    t = fetch_uniprot(cfg, "exclusion_ice_related", q, "accession,protein_name,protein_families")
    return set(t["Entry"])


def annotated_controls(cfg: dict) -> dict[str, pl.DataFrame]:
    """Hard controls chosen by UniProt annotation (similar fold or homolog of a positive)."""
    queries = {
        "c_type_lectin": 'reviewed:true AND ft_domain:"C-type lectin" AND length:[100 TO 250]',
        "sialic_acid_synthase": 'reviewed:true AND (protein_name:"N-acetylneuraminate synthase" '
        'OR protein_name:"sialic acid synthase")',
        "apolipoprotein_a1_a4_e": 'reviewed:true AND family:"apolipoprotein A1/A4/E family"',
        "secreted": "reviewed:true AND keyword:KW-0964 AND length:[10 TO 1000]",
    }
    return {
        k: fetch_uniprot(cfg, f"control_{k}", q, "accession,protein_name,length")
        for k, q in queries.items()
    }


def control_mature_spans(cfg: dict, accs: list[str]) -> dict[str, tuple[int, int, str]]:
    """Mature region of each control from UniProt CHAIN / SIGNAL / PROPEP features.

    Same rule as the positives: the longest CHAIN; else everything after the signal
    peptide; else the whole sequence. Fetched in chunks, cached by the accession list.
    """
    spans = {}
    accs = sorted(accs)
    for i in range(0, len(accs), 100):
        chunk = accs[i : i + 100]
        tag = hashlib.md5(",".join(chunk).encode()).hexdigest()[:10]
        t = fetch_uniprot(cfg, f"control_features_{tag}",
                          " OR ".join(f"accession:{a}" for a in chunk),
                          "accession,length,ft_chain,ft_signal")
        for r in t.iter_rows(named=True):
            chains = [parse_span(x) for x in re.findall(r"CHAIN ([<>?\d.]+)", r["Chain"] or "")]
            chains = [c for c in chains if c]
            sig = [parse_span(x) for x in re.findall(r"SIGNAL ([<>?\d.]+)", r["Signal peptide"] or "")]
            sig = [x for x in sig if x]
            if chains:
                st, en = max(chains, key=lambda c: c[1] - c[0])
                spans[r["Entry"]] = (st, en, "longest CHAIN")
            elif sig:
                spans[r["Entry"]] = (sig[0][1] + 1, int(r["Length"]), "after SIGNAL")
            else:
                spans[r["Entry"]] = (1, int(r["Length"]), "whole sequence (no CHAIN or SIGNAL)")
    return spans


def scan_swissprot(cfg: dict, exclude: set[str], wanted: set[str]) -> tuple[list, dict, dict]:
    """One pass over Swiss-Prot: the random pool, composition extremes, wanted sequences."""
    p = cfg["panel"]
    fasta = Path(cfg["inputs"]["swissprot_fasta"])
    md5 = hashlib.md5(fasta.read_bytes()).hexdigest()
    if md5 != cfg["inputs"]["swissprot_md5"]:
        sys.exit(f"{fasta}: md5 {md5}, config wants {cfg['inputs']['swissprot_md5']}")
    n_total = 575_748  # entries in Swiss-Prot 2026_03 (PR 141 README)
    keep_frac = p["background_pool_size"] / n_total
    over = 10 * p["hard_extreme_per_category"]
    extremes = {k: [] for k in ("ala_rich", "ala_thr_rich", "cys_rich", "low_distinct_3mer")}
    pool, found = [], {}
    for acc, header, raw in iter_fasta(fasta):
        if acc in exclude:
            continue
        name, organism = header_fields(header)
        if ICE_NAME.search(name):
            continue
        seq = clean(raw)
        L = len(seq)
        if acc in wanted:
            found[acc] = (name, organism, seq)
        if p["pool_min_length"] <= L <= p["pool_max_length"] and \
                stable_fraction(acc, cfg["seeds"]["background_pool"]) < keep_frac:
            pool.append((acc, name, organism, seq))
        if p["hard_min_length"] <= L <= p["hard_max_length"]:
            a, t, c = seq.count("A") / L, seq.count("T") / L, seq.count("C") / L
            vals = {"ala_rich": a, "ala_thr_rich": a + t, "cys_rich": c}
            for k, v in vals.items():
                extremes[k].append((v, acc, name, organism, seq))
            extremes["low_distinct_3mer"].append(
                (-distinct_3mer_fraction(seq), acc, name, organism, seq))
            # Trim now and then so memory stays small.
            if len(extremes["ala_rich"]) > 20 * over:
                for k in extremes:
                    extremes[k] = sorted(extremes[k], reverse=True)[:over]
    for k in extremes:
        extremes[k] = sorted(extremes[k], key=lambda r: (-r[0], r[1]))[:over]
    return pool, extremes, found


def step_panel(cfg: dict) -> pl.DataFrame:
    """Build controls, shuffles, dedupe within split, remove leaks, write the manifest."""
    out = cfg["_out"]
    p, sp = cfg["panel"], cfg["split"]
    truth = pl.read_csv(out / "truth_manifest.tsv", separator="\t")
    dev_o, eval_o = set(sp["dev_origins"]), set(sp["eval_origins"])
    split_of_origin = {**{o: "dev" for o in dev_o}, **{o: "eval" for o in eval_o}}
    split_of_origin["AFP4_apolipoprotein_like"] = "eval"

    exclude = exclusion_accessions(cfg) | set(truth["accession"])
    ann = annotated_controls(cfg)
    wanted = set()
    for k, t in ann.items():
        wanted |= set(t["Entry"])
    pool, extremes, found = scan_swissprot(cfg, exclude, wanted)

    ded = blosum62_aligner()
    removed: list[dict] = []

    # Positives: dedupe within each origin (never across origins: AFGPs of the two origins
    # are near identical by convergence, and that pair is the point of the test).
    pos_rows = []
    for origin, grp in truth.group_by("origin_id", maintain_order=True):
        grp = grp.sort(
            ["function_has_experimental_eco", "n_experimental_evidence_lines", "full_length",
             "accession"], descending=[True, True, True, False])
        kept: list[dict] = []
        for r in grp.iter_rows(named=True):
            hit = next((k for k in kept if _redundant(ded, r["sequence"], k["sequence"],
                       cfg["dedupe"]["within_split_identity"],
                       cfg["dedupe"]["within_split_coverage"])), None)
            if hit:
                removed.append({"seq_id": r["seq_id"], "reason": "redundant positive",
                                "kept_instead": hit["seq_id"]})
            else:
                kept.append(r)
        for r in kept:
            r = dict(r)
            r["split"] = split_of_origin.get(r["origin_id"], "excluded")
            r["control_category"] = ""
            r["source_seq_id"] = ""
            pos_rows.append(r)
    pos = pl.DataFrame(pos_rows)
    queries = pos.filter(pl.col("status") == "positive")

    rows = []

    def control(acc, name, organism, seq, split, category, source=""):
        rows.append({
            "seq_id": acc, "accession": acc, "kind": "real", "protein_name": name,
            "species": organism, "split": split, "status": REAL_NEGATIVE_STATUS,
            "control_category": category, "source_seq_id": source,
            "full_length": len(seq), "mature_start": 1, "mature_end": len(seq),
            "boundary_source": "whole sequence", "sequence": seq,
            "ice_binding_evidence": "none known; not tested (not annotated as ice-related)",
            "origin_id": "",
        })

    # Matched background: length within the log2 window, closest composition.
    pool_comp = np.array([composition(s) for *_, s in pool])
    pool_len = np.array([len(s) for *_, s in pool], dtype=float)
    used: set[str] = set()
    for q in queries.iter_rows(named=True):
        qc, ql = composition(q["sequence"]), len(q["sequence"])
        ok = np.abs(np.log2(pool_len / ql)) <= p["matched_max_abs_log2_length_ratio"]
        dist = np.sqrt(0.5 * ((np.sqrt(pool_comp) - np.sqrt(qc)) ** 2).sum(axis=1))
        dist[~ok] = np.inf
        n = 0
        for i in np.argsort(dist, kind="stable"):
            if not np.isfinite(dist[i]) or n == p["matched_per_positive"]:
                break
            acc, name, organism, seq = pool[i]
            if acc in used:
                continue
            used.add(acc)
            control(acc, name, organism, seq, q["split"], "matched_background", q["seq_id"])
            n += 1

    # Composition extremes and repeats, split by a seeded coin, half per split. An
    # evaluation candidate that is a true leak (is_leak) to a development pick is skipped,
    # so a paralog family is not split across the two sides.
    dev_picks: list[str] = []
    per_split = p["hard_extreme_per_category"] // 2
    for cat, cands in extremes.items():
        n = {"dev": 0, "eval": 0}
        for _, acc, name, organism, seq in cands:
            if acc in used or min(n.values()) == per_split:
                continue
            split = "dev" if stable_fraction(acc, cfg["seeds"]["hard_control_split"]) < 0.5 else "eval"
            if n[split] == per_split:
                split = "eval" if split == "dev" else "dev"
            if split == "eval" and any(is_leak(cfg, ded, seq, d) for d in dev_picks):
                continue
            used.add(acc)
            if split == "dev":
                dev_picks.append(seq)
            control(acc, name, organism, seq, split, f"hard_{cat}")
            n[split] += 1

    # Annotated controls.
    rng = random.Random(cfg["seeds"]["hard_control_split"])
    for cat, split in sp["hard_homolog_split"].items():
        accs = sorted(a for a in ann[cat]["Entry"] if a in found and a not in used)
        rng.shuffle(accs)
        for acc in accs[: p["hard_annotated_per_category"]]:
            used.add(acc)
            control(acc, *found[acc], split, f"hard_homolog_{cat}")
    # Secreted: per split, length-matched to that split's queries (nearest length, seeded).
    sec = sorted(a for a in ann["secreted"]["Entry"] if a in found and a not in used)
    rng.shuffle(sec)
    for split in ("dev", "eval"):
        qlens = queries.filter(pl.col("split") == split)["sequence"].str.len_chars().to_list()
        n = 0
        for i, acc in enumerate(sec):
            if n == p["secreted_per_split"]:
                break
            if acc in used:
                continue
            L = len(found[acc][2])
            target = qlens[n % len(qlens)]
            if abs(math.log2(L / target)) <= p["matched_max_abs_log2_length_ratio"]:
                used.add(acc)
                control(acc, *found[acc], split, "hard_secreted")
                n += 1

    ctrl = pl.DataFrame(rows)
    candidates = [{k: r[k] for k in ("seq_id", "split", "control_category", "protein_name")}
                  for r in rows]
    spans = control_mature_spans(cfg, ctrl["accession"].to_list())
    ctrl = pl.DataFrame([
        {**r, "mature_start": spans[r["accession"]][0], "mature_end": spans[r["accession"]][1],
         "boundary_source": spans[r["accession"]][2],
         "sequence": r["sequence"][spans[r["accession"]][0] - 1 : spans[r["accession"]][1]]}
        for r in ctrl.iter_rows(named=True)
    ])

    # Dedupe controls within split, against positives first, then each other.
    kept_ctrl = []
    for split in ("dev", "eval"):
        ref = [r for r in pos.filter(pl.col("split") == split).iter_rows(named=True)]
        for r in ctrl.filter(pl.col("split") == split).iter_rows(named=True):
            hit = next((k for k in ref if _redundant(ded, r["sequence"], k["sequence"],
                       cfg["dedupe"]["within_split_identity"],
                       cfg["dedupe"]["within_split_coverage"])), None)
            if hit:
                removed.append({"seq_id": r["seq_id"], "reason": f"redundant {split} control",
                                "kept_instead": hit["seq_id"]})
                continue
            ref.append(r)
            kept_ctrl.append(r)
    ctrl = pl.DataFrame(kept_ctrl)

    # Leakage: drop an evaluation control close to any development sequence; report
    # positive pairs across splits (kept, since their origins are separate by citation).
    dev_seqs = pl.concat([pos.filter(pl.col("split") == "dev").select("seq_id", "sequence"),
                          ctrl.filter(pl.col("split") == "dev").select("seq_id", "sequence")])
    leak_rows, keep, composition_only = [], [], []
    for r in pl.concat([pos.filter(pl.col("split") == "eval").select("seq_id", "sequence", "status"),
                        ctrl.filter(pl.col("split") == "eval").select("seq_id", "sequence", "status")]
                       ).iter_rows(named=True):
        worst = (0.0, 0.0, "", 0.0)
        for d in dev_seqs.iter_rows(named=True):
            ident, cov, cols = identity_coverage(ded, r["sequence"], d["sequence"], True)
            if ident >= cfg["dedupe"]["leak_identity"] and cov >= cfg["dedupe"]["leak_coverage"] \
                    and cols >= cfg["dedupe"]["leak_min_columns"]:
                z = shuffle_z(ded, r["sequence"], d["sequence"], cfg["dedupe"]["leak_shuffles"],
                              f"{cfg['seeds']['shuffles']}:leak:{d['seq_id']}")
                if z >= cfg["dedupe"]["leak_shuffle_z"] and (ident, cov) > worst[:2]:
                    worst = (ident, cov, d["seq_id"], z)
                elif z < cfg["dedupe"]["leak_shuffle_z"]:
                    composition_only.append({"eval_seq_id": r["seq_id"], "dev_seq_id": d["seq_id"],
                                             "identity": ident, "coverage": cov, "columns": cols,
                                             "shuffle_z": z})
        if worst[2]:
            leak_rows.append({"eval_seq_id": r["seq_id"], "eval_status": r["status"],
                              "closest_dev_seq_id": worst[2], "identity": worst[0],
                              "coverage": worst[1], "shuffle_z": worst[3],
                              "action": "removed" if r["status"] == REAL_NEGATIVE_STATUS
                              else "kept (positive of a separate cited origin)"})
            if r["status"] == REAL_NEGATIVE_STATUS:
                continue
        keep.append(r["seq_id"])
    ctrl = ctrl.filter((pl.col("split") == "dev") | pl.col("seq_id").is_in(keep))
    fate = {r["seq_id"]: r["reason"] for r in removed}
    fate |= {r["eval_seq_id"]: "leak to development (removed)" for r in leak_rows
             if r["action"] == "removed"}
    pl.DataFrame(candidates).with_columns(
        fate=pl.col("seq_id").replace_strict(fate, default="kept")
    ).write_csv(out / "control_candidates.tsv", separator="\t")
    pl.DataFrame(leak_rows, schema={"eval_seq_id": pl.Utf8, "eval_status": pl.Utf8,
                                    "closest_dev_seq_id": pl.Utf8, "identity": pl.Float64,
                                    "coverage": pl.Float64, "shuffle_z": pl.Float64,
                                    "action": pl.Utf8}
                 ).write_csv(out / "leakage.tsv", separator="\t")
    pl.DataFrame(composition_only).write_csv(out / "leakage_composition_only.tsv", separator="\t")
    pl.DataFrame(removed).write_csv(out / "dedupe_removed.tsv", separator="\t")

    # Shuffles of each query positive, same split, reported apart from real proteins.
    shuf_rows = []
    for q in queries.iter_rows(named=True):
        for k in range(p["shuffles_per_positive"]):
            rng_k = random.Random(f"{cfg['seeds']['shuffles']}:{q['seq_id']}:{k}")
            letters = list(q["sequence"])
            rng_k.shuffle(letters)
            shuf_rows.append({
                "seq_id": f"{q['seq_id']}__shuf{k}", "accession": "", "kind": "shuffle",
                "protein_name": f"shuffle {k} of {q['seq_id']}", "species": q["species"],
                "split": q["split"], "status": "shuffle_control",
                "control_category": "shuffle_of_positive", "source_seq_id": q["seq_id"],
                "full_length": len(letters), "mature_start": 1, "mature_end": len(letters),
                "boundary_source": "shuffle of mature sequence", "sequence": "".join(letters),
                "ice_binding_evidence": "composition-preserving shuffle",
                "origin_id": q["origin_id"],
            })
    shuf = pl.DataFrame(shuf_rows)

    cols = ["seq_id", "accession", "kind", "split", "status", "origin_id", "control_category",
            "source_seq_id", "protein_name", "species", "full_length", "mature_start",
            "mature_end", "boundary_source", "ice_binding_evidence", "sequence"]
    extra = [c for c in pos.columns if c not in cols]
    man = pl.concat([
        pos.select(cols + extra),
        ctrl.select(cols).with_columns([pl.lit(None).alias(c) for c in extra]).select(cols + extra),
        shuf.select(cols).with_columns([pl.lit(None).alias(c) for c in extra]).select(cols + extra),
    ], how="vertical_relaxed").with_columns(length=pl.col("sequence").str.len_chars())
    man.write_csv(out / "manifest.tsv", separator="\t")
    for split in ("dev", "eval"):
        man.filter(pl.col("split") == split).select("seq_id", "status", "origin_id",
            "control_category").write_csv(out / f"split_{split}.tsv", separator="\t")
    return man


def is_leak(cfg: dict, aligner, a: str, b: str) -> bool:
    """The leakage rule of the config: close by identity, long, and beyond composition."""
    d = cfg["dedupe"]
    ident, cov, cols = identity_coverage(aligner, a, b, True)
    if ident < d["leak_identity"] or cov < d["leak_coverage"] or cols < d["leak_min_columns"]:
        return False
    return shuffle_z(aligner, a, b, d["leak_shuffles"], f"{cfg['seeds']['shuffles']}:leak") \
        >= d["leak_shuffle_z"]


def shuffle_z(aligner, a: str, b: str, n: int, seed: str) -> float:
    """How many SDs the a-b score sits above a aligned to n shuffles of b."""
    real = best_alignment(aligner, a, b)
    real = real.score if real is not None else 0.0
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        letters = list(b)
        rng.shuffle(letters)
        aln = best_alignment(aligner, a, "".join(letters))
        null.append(aln.score if aln is not None else 0.0)
    sd = float(np.std(null)) or 1.0
    return (real - float(np.mean(null))) / sd


def _redundant(aligner, a: str, b: str, min_ident: float, min_cov: float) -> bool:
    ident, cov = identity_coverage(aligner, a, b)
    return ident >= min_ident and cov >= min_cov


# --------------------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------------------


def swissprot_background(cfg: dict) -> np.ndarray:
    """Residue frequencies of Swiss-Prot 2026_03, cached in background_frequencies.tsv."""
    path = cfg["_out"] / "background_frequencies.tsv"
    if path.exists():
        t = pl.read_csv(path, separator="\t")
        return np.array([dict(zip(t["residue"], t["frequency"]))[a] for a in AMINO_ACIDS])
    counts = np.zeros(20)
    idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    for _, _, seq in iter_fasta(Path(cfg["inputs"]["swissprot_fasta"])):
        for a, n in zip(*np.unique(list(seq), return_counts=True)):
            if a in idx:
                counts[idx[a]] += n
    q = counts / counts.sum()
    pl.DataFrame({"residue": list(AMINO_ACIDS), "frequency": q}).write_csv(path, separator="\t")
    return q


def collapse_matrix(source: str, mapping: dict[str, str], q: np.ndarray):
    """Half-bit log-odds matrix over class letters, from a BLOSUM matrix and background q.

    Joint frequencies are recovered as p_ab = q_a q_b 2^(s_ab / 2), normalised to sum 1,
    summed over the residues of each class pair; scores are 2 log2(P_AB / (P_A P_B)).
    For protein20 (identity mapping) the source matrix is returned unchanged.
    """
    m = substitution_matrices.load(source)
    letters = sorted(set(mapping.values()), key=lambda c: list(mapping.values()).index(c))
    if letters == list(AMINO_ACIDS) and all(mapping[a] == a for a in AMINO_ACIDS):
        return m
    s = np.array([[m[a][b] for b in AMINO_ACIDS] for a in AMINO_ACIDS], dtype=float)
    p = np.outer(q, q) * 2 ** (s / 2)
    p /= p.sum()
    k = len(letters)
    cls = np.array([letters.index(mapping[a]) for a in AMINO_ACIDS])
    P = np.zeros((k, k))
    for i in range(20):
        for j in range(20):
            P[cls[i], cls[j]] += p[i, j]
    marg = P.sum(axis=1)
    S = 2 * np.log2(P / np.outer(marg, marg))
    arr = substitution_matrices.Array("".join(letters), dims=2, data=S)
    return arr


def matrix_table(cfg: dict) -> pl.DataFrame:
    """Every matrix in the grid as long rows, and its expected score under the background."""
    alph = read_alphabets(REPO / cfg["inputs"]["alphabet_classes"])
    q = swissprot_background(cfg)
    rows = []
    for name, mapping in alph.items():
        for src in sorted({g["matrix"] for g in cfg["grid"] if g["alphabet"] == name}):
            m = collapse_matrix(src, mapping, q)
            letters = list(m.alphabet) if name != "protein20" else list(AMINO_ACIDS)
            cq = {c: sum(q[i] for i, a in enumerate(AMINO_ACIDS) if mapping[a] == c) for c in letters}
            expected = sum(cq[a] * cq[b] * m[a][b] for a in letters for b in letters)
            for a in letters:
                for b in letters:
                    rows.append({"alphabet": name, "source_matrix": src, "a": a, "b": b,
                                 "score_half_bits": float(m[a][b]),
                                 "expected_score_half_bits": expected})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Step 3: alignment
# --------------------------------------------------------------------------------------


def settings_for(cfg: dict, which: str) -> list[dict]:
    """The grid (development) or the selection plus reference settings (evaluation)."""
    if which == "grid":
        return cfg["grid"]
    sel = cfg["_out"] / "selection.tsv"
    if not sel.exists():
        sys.exit("evaluation needs selection.tsv; run `select` on the development split first")
    chosen = set(pl.read_csv(sel, separator="\t")["setting"])
    chosen |= set(cfg["selection"]["reference_settings"])
    return [g for g in cfg["grid"] if g["setting"] in chosen]


def align_split(cfg: dict, split: str) -> pl.DataFrame:
    """Every query positive of `split` against every other sequence of `split`."""
    out = cfg["_out"]
    man = pl.read_csv(out / "manifest.tsv", separator="\t", infer_schema_length=0)
    man = man.filter(pl.col("split") == split)
    queries = man.filter(pl.col("status") == "positive")
    alph = read_alphabets(REPO / cfg["inputs"]["alphabet_classes"])
    q = swissprot_background(cfg)
    which = "grid" if split == "dev" else "selected"
    settings = settings_for(cfg, which)
    if split == "eval":
        sel_md5 = hashlib.md5((out / "selection.tsv").read_bytes()).hexdigest()
        (out / "eval_guard.json").write_text(json.dumps({
            "selection_md5": sel_md5, "config_md5": cfg["_md5"],
            "settings": [s["setting"] for s in settings],
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    targets = man.select("seq_id", "sequence").rows()
    rows = []
    for g in settings:
        t0 = time.perf_counter()
        mapping = alph[g["alphabet"]]
        aligner = make_aligner(collapse_matrix(g["matrix"], mapping, q),
                               g["gap_open"], g["gap_extend"])
        tr = {sid: translate(seq, mapping) for sid, seq in targets}
        raw = dict(targets)
        n = 0
        for qid in queries["seq_id"]:
            qs = tr[qid]
            for tid, _ in targets:
                if tid == qid:
                    continue
                aln = best_alignment(aligner, qs, tr[tid])
                if aln is None:
                    rows.append((split, g["setting"], qid, tid, 0.0, 0, 0, 0, 0, 0, 0, 0,
                                 len(qs), len(tr[tid])))
                    n += 1
                    continue
                c = aln.coordinates
                ident_res = ident_cls = 0
                for (a0, a1), (b0, b1) in zip(*aln.aligned):
                    ident_res += sum(x == y for x, y in zip(raw[qid][a0:a1], raw[tid][b0:b1]))
                    ident_cls += a1 - a0
                q0, q1, t0_, t1 = int(c[0, 0]), int(c[0, -1]), int(c[1, 0]), int(c[1, -1])
                cols = (q1 - q0) + (t1 - t0_) - ident_cls
                rows.append((split, g["setting"], qid, tid, float(aln.score), q0, q1, t0_, t1,
                             cols, ident_res, ident_cls, len(qs), len(tr[tid])))
                n += 1
        log_runtime(cfg, "align", time.perf_counter() - t0, f"{split} {g['setting']} {n} pairs")
    df = pl.DataFrame(rows, orient="row", schema=[
        "split", "setting", "query", "target", "score", "q_start", "q_end", "t_start", "t_end",
        "aln_columns", "identical_residues", "identical_classes", "q_len", "t_len"])
    df.write_csv(out / f"pairs_{split}.tsv.gz", separator="\t")
    return df


def baseline_scores(cfg: dict, split: str) -> pl.DataFrame:
    """Alignment-free similarity of every query and target. Higher = more similar."""
    out = cfg["_out"]
    man = pl.read_csv(out / "manifest.tsv", separator="\t", infer_schema_length=0)
    man = man.filter(pl.col("split") == split)
    alph = read_alphabets(REPO / cfg["inputs"]["alphabet_classes"])
    seqs = dict(man.select("seq_id", "sequence").rows())
    feats = {}
    for sid, s in seqs.items():
        feats[sid] = {
            "len": len(s),
            "c20": composition(s),
            "chp": composition(translate(s, alph["hp_lehninger2"]), "HP"),
            "cday": composition(translate(s, alph["dayhoff6"]), "abcdef"),
            "d3": distinct_3mer_fraction(s),
            "at": (s.count("A") + s.count("T")) / len(s),
        }
    rows = []
    for qid in man.filter(pl.col("status") == "positive")["seq_id"]:
        fq = feats[qid]
        for tid, ft in feats.items():
            if tid == qid:
                continue
            vals = {
                "length": -abs(math.log(fq["len"] / ft["len"])),
                "composition20": -hellinger(fq["c20"], ft["c20"]),
                "composition_hp_lehninger2": -hellinger(fq["chp"], ft["chp"]),
                "composition_dayhoff6": -hellinger(fq["cday"], ft["cday"]),
                "distinct_3mer_fraction": -abs(fq["d3"] - ft["d3"]),
                "target_ala_thr_fraction": ft["at"],
            }
            for name, v in vals.items():
                rows.append((split, f"baseline_{name}", qid, tid, v))
    return pl.DataFrame(rows, orient="row",
                        schema=["split", "setting", "query", "target", "similarity"])


# --------------------------------------------------------------------------------------
# Calibration and metrics
# --------------------------------------------------------------------------------------


def fit_gumbel(scores: np.ndarray, mn: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood lambda and K of P(S >= x) = 1 - exp(-K m n e^(-lambda x))."""

    def nll(theta):
        lam, lnk = math.exp(theta[0]), theta[1]
        z = lam * scores - lnk - np.log(mn)
        return -np.sum(math.log(lam) - z - np.exp(-z))

    lam0 = 1.0 / max(np.std(scores) * math.sqrt(6) / math.pi, 1e-6)
    res = minimize(nll, x0=[math.log(lam0), -1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    return math.exp(res.x[0]), math.exp(res.x[1])


def label_pairs(pairs: pl.DataFrame, man: pl.DataFrame) -> pl.DataFrame:
    """Add the query and target origin and the pair's role in each task."""
    info = man.select(
        pl.col("seq_id"), pl.col("status"), pl.col("origin_id").fill_null(""),
        pl.col("control_category").fill_null(""), pl.col("source_seq_id").fill_null(""))
    df = pairs.join(info.rename({"seq_id": "query", "origin_id": "q_origin"}).select(
        "query", "q_origin"), on="query").join(info.rename({
            "seq_id": "target", "status": "t_status", "origin_id": "t_origin",
            "control_category": "t_category", "source_seq_id": "t_source"}), on="target")
    same = pl.col("q_origin") == pl.col("t_origin")
    return df.with_columns(
        role=pl.when((pl.col("t_status") == "positive") & ~same).then(pl.lit("cross_origin_positive"))
        .when((pl.col("t_status") == "positive") & same).then(pl.lit("same_origin_positive"))
        .when(pl.col("t_status") == REAL_NEGATIVE_STATUS).then(pl.lit("real_negative"))
        .when(pl.col("t_status") == "shuffle_control").then(pl.lit("shuffle"))
        .otherwise(pl.lit("ignored")))


def recall_at_budget(pos: np.ndarray, neg: np.ndarray, rate: float) -> float:
    """Fraction of positives scoring above the (b+1)-th best negative, b = floor(rate n).

    Higher score = better. Ties with that negative count as missed.
    """
    if len(pos) == 0:
        return float("nan")
    if len(neg) == 0:
        return 1.0
    b = int(math.floor(rate * len(neg)))
    if b >= len(neg):
        return 1.0
    thr = np.sort(neg)[::-1][b]
    return float(np.mean(pos > thr))


def pooled_recall_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    """Same rule as recall_at_budget, applied to all queries' pairs pooled."""
    return recall_at_budget(pos, neg, fpr)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive outscores random negative), ties half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def task_metrics(df: pl.DataFrame, positive_role: str, rates: list[float]) -> pl.DataFrame:
    """Per query: AP and recall at each per-query budget. `df` has a 'sim' column."""
    rows = []
    for (setting, query, q_origin), g in df.group_by(["setting", "query", "q_origin"],
                                                     maintain_order=True):
        pos = g.filter(pl.col("role") == positive_role)["sim"].to_numpy()
        neg = g.filter(pl.col("role") == "real_negative")["sim"].to_numpy()
        if len(pos) == 0:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        row = {"setting": setting, "query": query, "q_origin": q_origin,
               "task": positive_role, "n_positives": len(pos), "n_negatives": len(neg),
               "average_precision": float(average_precision_score(y, np.r_[pos, neg])),
               "best_positive_rank": int(1 + np.sum(neg >= pos.max()))}
        for r in rates:
            row[f"recall_fpr_{r}"] = recall_at_budget(pos, neg, r)
        rows.append(row)
    return pl.DataFrame(rows)


def with_similarity(pairs: pl.DataFrame, params: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Add E-value and sim = -log10(E) (higher = better) from the development fit."""
    lam = pl.col("setting").replace_strict({k: v[0] for k, v in params.items()}, default=None)
    k = pl.col("setting").replace_strict({k: v[1] for k, v in params.items()}, default=None)
    e = k * pl.col("q_len") * pl.col("t_len") * (-lam * pl.col("score")).exp()
    return pairs.with_columns(evalue=e).with_columns(
        sim=-(pl.col("evalue").clip(lower_bound=1e-300)).log10())


def step_select(cfg: dict) -> pl.DataFrame:
    """Fit lambda, K per setting on development nulls; pick one setting per alphabet."""
    out = cfg["_out"]
    man = pl.read_csv(out / "manifest.tsv", separator="\t", infer_schema_length=0)
    pairs = label_pairs(pl.read_csv(out / "pairs_dev.tsv.gz", separator="\t"), man)
    params, cal = {}, []
    for g in cfg["grid"]:
        nul = pairs.filter((pl.col("setting") == g["setting"]) & (pl.col("role") == "real_negative"))
        lam, K = fit_gumbel(nul["score"].to_numpy().astype(float),
                            (nul["q_len"] * nul["t_len"]).to_numpy().astype(float))
        params[g["setting"]] = (lam, K)
        cal.append({"setting": g["setting"], "alphabet": g["alphabet"], "lambda": lam, "K": K,
                    "n_null_pairs": nul.height, "fit_on": "dev real_negative pairs"})
    pl.DataFrame(cal).write_csv(out / "calibration.tsv", separator="\t")
    scored = with_similarity(pairs, params)
    rates = cfg["budgets"]["per_query_false_hit_rates"]
    cross = task_metrics(scored, "cross_origin_positive", rates)
    within = task_metrics(scored, "same_origin_positive", rates)
    grid_rows = []
    for i, g in enumerate(cfg["grid"]):
        c = cross.filter(pl.col("setting") == g["setting"])
        w = within.filter(pl.col("setting") == g["setting"])
        grid_rows.append({
            "grid_order": i, "setting": g["setting"], "alphabet": g["alphabet"],
            "matrix": g["matrix"], "gap_open": g["gap_open"], "gap_extend": g["gap_extend"],
            "mean_cross_origin_average_precision": c["average_precision"].mean(),
            "mean_within_origin_recall_fpr_0.01": w["recall_fpr_0.01"].mean(),
            "n_dev_queries": c.height})
    grid = pl.DataFrame(grid_rows)
    grid.write_csv(out / "dev_grid_metrics.tsv", separator="\t")
    chosen = (grid.sort(["alphabet", cfg["selection"]["primary"], cfg["selection"]["tie_break"],
                         "grid_order"], descending=[False, True, True, False])
              .group_by("alphabet", maintain_order=True).first())
    chosen.write_csv(out / "selection.tsv", separator="\t")
    return chosen


def step_evaluate(cfg: dict) -> None:
    """All evaluation tables: metrics, pooled PR, controls, shuffles, subgroups, decision."""
    out = cfg["_out"]
    man = pl.read_csv(out / "manifest.tsv", separator="\t", infer_schema_length=0)
    cal = pl.read_csv(out / "calibration.tsv", separator="\t")
    params = {r["setting"]: (r["lambda"], r["K"]) for r in cal.iter_rows(named=True)}
    rates = cfg["budgets"]["per_query_false_hit_rates"]
    fprs = cfg["budgets"]["pooled_false_positive_rates"]
    all_scored = []
    for split in ("dev", "eval"):
        pairs = label_pairs(pl.read_csv(out / f"pairs_{split}.tsv.gz", separator="\t"), man)
        if split == "dev":
            keep = set(pl.read_csv(out / "selection.tsv", separator="\t")["setting"]) | set(
                cfg["selection"]["reference_settings"])
            pairs = pairs.filter(pl.col("setting").is_in(keep))
        sc = with_similarity(pairs, params)
        base = label_pairs(baseline_scores(cfg, split), man).rename({"similarity": "sim"})
        all_scored.append(sc.select(base.columns + ["score", "evalue", "q_start", "q_end",
                                                    "t_start", "t_end", "aln_columns",
                                                    "identical_residues", "identical_classes",
                                                    "q_len", "t_len"]))
        all_scored.append(base)
    scored = pl.concat(all_scored, how="diagonal_relaxed")
    selected = set(pl.read_csv(out / "selection.tsv", separator="\t")["setting"])
    scored = scored.with_columns(
        selected=pl.col("setting").is_in(selected),
        method=pl.when(pl.col("setting").str.starts_with("baseline_")).then(pl.lit("baseline"))
        .otherwise(pl.lit("alignment")))

    # Ranked lists, one row per query and target, rank 1 = most similar.
    ranked = scored.with_columns(
        rank=pl.col("sim").rank("min", descending=True).over(["split", "setting", "query"])
    ).sort("split", "setting", "query", "rank")
    ranked.drop("t_source").write_csv(out / "ranked_results.tsv.gz", separator="\t")

    perq = []
    for split in ("dev", "eval"):
        s = scored.filter(pl.col("split") == split)
        for task in ("cross_origin_positive", "same_origin_positive"):
            m = task_metrics(s, task, rates)
            if m.height:
                perq.append(m.with_columns(split=pl.lit(split)))
    perq = pl.concat(perq)
    perq.write_csv(out / "metrics_per_query.tsv", separator="\t")

    # Per query origin: mean of per-query values, and the number of queries behind it.
    per_origin = perq.group_by(["split", "setting", "task", "q_origin"]).agg(
        n_queries=pl.len(), mean_average_precision=pl.col("average_precision").mean(),
        **{f"mean_recall_fpr_{r}": pl.col(f"recall_fpr_{r}").mean() for r in rates},
        median_best_positive_rank=pl.col("best_positive_rank").median()
    ).sort("split", "task", "setting", "q_origin")
    per_origin.write_csv(out / "metrics_per_origin.tsv", separator="\t")

    # Pooled over all queries of a split.
    pooled, curves, controls = [], [], []
    for (split, setting), g in scored.group_by(["split", "setting"], maintain_order=True):
        neg = g.filter(pl.col("role") == "real_negative")["sim"].to_numpy()
        for task in ("cross_origin_positive", "same_origin_positive"):
            pos = g.filter(pl.col("role") == task)["sim"].to_numpy()
            if len(pos) == 0:
                continue
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            sc_all = np.r_[pos, neg]
            row = {"split": split, "setting": setting, "task": task, "n_positive_pairs": len(pos),
                   "n_negative_pairs": len(neg),
                   "pooled_average_precision": float(average_precision_score(y, sc_all)),
                   "pooled_auc": auc(pos, neg)}
            for f in fprs:
                row[f"pooled_recall_fpr_{f}"] = pooled_recall_at_fpr(pos, neg, f)
            pooled.append(row)
            order = np.argsort(-sc_all, kind="stable")
            tp = np.cumsum(y[order]); fp = np.cumsum(1 - y[order])
            idx = np.unique(np.linspace(0, len(order) - 1, 200).astype(int))
            for i in idx:
                curves.append({"split": split, "setting": setting, "task": task,
                               "recall": tp[i] / len(pos), "precision": tp[i] / (tp[i] + fp[i]),
                               "fpr": fp[i] / max(len(neg), 1)})
        # True cross-origin targets against each control kind.
        pos = g.filter(pl.col("role") == "cross_origin_positive")
        for cat, c in g.filter(pl.col("role").is_in(["real_negative", "shuffle"])).group_by(
                "t_category", maintain_order=True):
            controls.append({"split": split, "setting": setting, "control": cat[0],
                             "n_control_pairs": c.height,
                             "auc_cross_origin_vs_control": auc(pos["sim"].to_numpy(),
                                                                c["sim"].to_numpy()),
                             "median_control_sim": float(c["sim"].median()),
                             "median_cross_origin_sim": float(pos["sim"].median())
                             if pos.height else float("nan")})
        # Paired with shuffles: same query, shuffles of the very targets that are positives.
        sh = g.filter(pl.col("role") == "shuffle").join(
            pos.select(pl.col("query"), pl.col("target").alias("t_source")), on=["query", "t_source"])
        controls.append({"split": split, "setting": setting,
                         "control": "shuffles_of_the_cross_origin_targets",
                         "n_control_pairs": sh.height,
                         "auc_cross_origin_vs_control": auc(pos["sim"].to_numpy(), sh["sim"].to_numpy()),
                         "median_control_sim": float(sh["sim"].median()) if sh.height else float("nan"),
                         "median_cross_origin_sim": float(pos["sim"].median()) if pos.height else float("nan")})
    pooled = pl.DataFrame(pooled)
    pooled.write_csv(out / "metrics_pooled.tsv", separator="\t")
    pl.DataFrame(curves).write_csv(out / "pr_curves.tsv", separator="\t")
    controls = pl.DataFrame(controls)
    controls.write_csv(out / "control_comparison.tsv", separator="\t")

    # Calibration check on evaluation negatives: observed vs expected tail.
    cal_rows = []
    for (split, setting), g in scored.filter(pl.col("method") == "alignment").group_by(
            ["split", "setting"]):
        for role in ("real_negative", "shuffle"):
            e = g.filter(pl.col("role") == role)["evalue"].to_numpy()
            pval = 1 - np.exp(-e)
            for a in (0.001, 0.01, 0.05, 0.1):
                cal_rows.append({"split": split, "setting": setting, "pairs": role,
                                 "p_threshold": a, "observed_fraction": float(np.mean(pval <= a)),
                                 "n_pairs": len(e)})
    pl.DataFrame(cal_rows).sort("split", "setting", "pairs", "p_threshold").write_csv(
        out / "calibration_check.tsv", separator="\t")

    # Subgroups.
    sub_rows = []
    for sg in cfg["subgroup"]:
        o = set(sg["origins"])
        s = scored.filter(pl.col("q_origin").is_in(o) & (
            (pl.col("role") == "real_negative") | pl.col("t_origin").is_in(o)))
        for (split, setting), g in s.group_by(["split", "setting"], maintain_order=True):
            pos_df = g.filter(pl.col("role") == "cross_origin_positive")
            pos, neg = pos_df["sim"].to_numpy(), g.filter(pl.col("role") == "real_negative")["sim"].to_numpy()
            if len(pos) == 0:
                continue
            sh = g.filter(pl.col("role") == "shuffle").join(
                pos_df.select("query", pl.col("target").alias("t_source")), on=["query", "t_source"])
            sub_rows.append({"subgroup": sg["name"], "split": split, "setting": setting,
                             "n_positive_pairs": len(pos), "n_negative_pairs": len(neg),
                             "pooled_recall_fpr_0.01": pooled_recall_at_fpr(pos, neg, 0.01),
                             "pooled_auc": auc(pos, neg),
                             "auc_vs_shuffles_of_targets": auc(pos, sh["sim"].to_numpy()),
                             "basis": sg["basis"]})
    subs = pl.DataFrame(sub_rows)
    subs.write_csv(out / "subgroup_metrics.tsv", separator="\t")
    decide(cfg, pooled, controls, subs, selected)
    representative_alignments(cfg, scored, man, selected)


def separates(d: dict, rec: float, best_base: float, shuf_auc: float) -> tuple[bool, str]:
    """The predeclared three-part rule; returns the verdict and the reason."""
    parts = [
        (rec >= d["min_pooled_recall"], f"recall@1%FPR {rec:.3f} vs {d['min_pooled_recall']}"),
        (rec - best_base >= d["min_margin_over_baseline"],
         f"margin over best baseline {rec - best_base:+.3f} vs {d['min_margin_over_baseline']}"),
        (shuf_auc >= d["min_auc_vs_shuffles"],
         f"AUC vs shuffles {shuf_auc:.3f} vs {d['min_auc_vs_shuffles']}"),
    ]
    return all(p for p, _ in parts), "; ".join(("pass " if p else "fail ") + r for p, r in parts)


def decide(cfg: dict, pooled: pl.DataFrame, controls: pl.DataFrame, subs: pl.DataFrame,
           selected: set[str]) -> None:
    """Apply the decision rule from the config to the evaluation split; write decision.tsv."""
    d = cfg["decision"]
    ev = pooled.filter((pl.col("split") == "eval") & (pl.col("task") == "cross_origin_positive"))
    base = ev.filter(pl.col("setting").str.starts_with("baseline_"))
    best_base = float(base["pooled_recall_fpr_0.01"].max())
    best_base_name = base.sort("pooled_recall_fpr_0.01", descending=True)["setting"][0]
    rows = []
    for r in ev.filter(~pl.col("setting").str.starts_with("baseline_")).iter_rows(named=True):
        sh = controls.filter((pl.col("split") == "eval") & (pl.col("setting") == r["setting"]) & (
            pl.col("control") == "shuffles_of_the_cross_origin_targets"))["auc_cross_origin_vs_control"][0]
        ok, why = separates(d, r["pooled_recall_fpr_0.01"], best_base, sh)
        rows.append({"level": "broad", "name": r["setting"], "selected": r["setting"] in selected,
                     "alphabet": r["setting"].split("_B")[0], "separates": ok, "reason": why,
                     "best_baseline": best_base_name})
    sub_ev = subs.filter(pl.col("split") == "eval") if subs.height else subs
    for sg in cfg["subgroup"]:
        s = sub_ev.filter(pl.col("subgroup") == sg["name"]) if sub_ev.height else sub_ev
        if not s.height:
            rows.append({"level": "subgroup", "name": sg["name"], "selected": False, "alphabet": "",
                         "separates": False, "reason": "no evaluation pairs in this subgroup "
                         "(its origins are in the development split)", "best_baseline": ""})
            continue
        sb = s.filter(pl.col("setting").str.starts_with("baseline_"))
        bb = float(sb["pooled_recall_fpr_0.01"].max())
        for r in s.filter(pl.col("setting").is_in(selected)).iter_rows(named=True):
            ok, why = separates(d, r["pooled_recall_fpr_0.01"], bb, r["auc_vs_shuffles_of_targets"])
            rows.append({"level": "subgroup", "name": f"{sg['name']} | {r['setting']}",
                         "selected": True, "alphabet": r["setting"].split("_B")[0],
                         "separates": ok, "reason": why,
                         "best_baseline": sb.sort("pooled_recall_fpr_0.01", descending=True)["setting"][0]})
    dec = pl.DataFrame(rows)
    broad = dec.filter((pl.col("level") == "broad") & pl.col("selected"))
    reduced_ok = broad.filter((pl.col("alphabet") != "protein20") & pl.col("separates")).height > 0
    prot_ok = broad.filter((pl.col("alphabet") == "protein20") & pl.col("separates")).height > 0
    sub_ok = dec.filter((pl.col("level") == "subgroup") & pl.col("separates")).height > 0
    if reduced_ok:
        outcome = "reduced alignment works; kmerseek's retrieval may be the bottleneck"
    elif prot_ok:
        outcome = "full-sequence signal exists, reduced signal does not"
    elif sub_ok:
        outcome = "only a narrow subgroup shows signal"
    else:
        outcome = "no tested linear scoring method separates positives from controls"
    dec = dec.with_columns(outcome=pl.lit(outcome))
    dec.write_csv(cfg["_out"] / "decision.tsv", separator="\t")


def representative_alignments(cfg: dict, scored: pl.DataFrame, man: pl.DataFrame,
                              selected: set[str]) -> None:
    """For each evaluation query origin and selected setting: the best cross-origin target
    and the best real negative, printed residue by residue."""
    alph = read_alphabets(REPO / cfg["inputs"]["alphabet_classes"])
    q = swissprot_background(cfg)
    seqs = dict(man.select("seq_id", "sequence").rows())
    names = dict(man.select("seq_id", "protein_name").rows())
    grid = {g["setting"]: g for g in cfg["grid"]}
    lines, rows = [], []
    ev = scored.filter((pl.col("split") == "eval") & pl.col("setting").is_in(selected))
    for (setting, q_origin), g in ev.group_by(["setting", "q_origin"], maintain_order=True):
        for role in ("cross_origin_positive", "real_negative"):
            best = g.filter(pl.col("role") == role).sort("sim", descending=True).head(1)
            if not best.height:
                continue
            r = best.row(0, named=True)
            gs = grid[setting]
            mapping = alph[gs["alphabet"]]
            aligner = make_aligner(collapse_matrix(gs["matrix"], mapping, q), gs["gap_open"], gs["gap_extend"])
            qa, ta = seqs[r["query"]], seqs[r["target"]]
            q_tr, t_tr = translate(qa, mapping), translate(ta, mapping)
            aln = aligner.align(q_tr, t_tr)[0]
            qrow, mrow, trow = gapped_rows(aln, qa, ta, q_tr, t_tr)
            rows.append({"setting": setting, "q_origin": q_origin, "role": role, "query": r["query"],
                         "target": r["target"], "t_origin": r["t_origin"], "t_category": r["t_category"],
                         "score": r["score"], "evalue": r["evalue"],
                         "q_span": f"{r['q_start'] + 1}-{r['q_end']}",
                         "t_span": f"{r['t_start'] + 1}-{r['t_end']}",
                         "residue_identity": r["identical_residues"] / max(r["aln_columns"], 1),
                         "query_row": qrow, "match_row": mrow, "target_row": trow})
            lines += [f"## {setting} | query origin {q_origin} | best {role}",
                      f"query  {r['query']} ({names[r['query']]}) {r['q_start'] + 1}-{r['q_end']}",
                      f"target {r['target']} ({names[r['target']]}) {r['t_start'] + 1}-{r['t_end']}",
                      f"score {r['score']:.1f} half bits, E = {r['evalue']:.3g}", "",
                      f"  query  {qrow}", f"         {mrow}", f"  target {trow}", ""]
    pl.DataFrame(rows).write_csv(cfg["_out"] / "representative_alignments.tsv", separator="\t")
    (cfg["_out"] / "representative_alignments.txt").write_text(
        "Real residues shown; '|' identical residue, ':' same class in the alphabet aligned, "
        "' ' different class.\n\n" + "\n".join(lines))


def gapped_rows(aln, qa: str, ta: str, q_tr: str, t_tr: str) -> tuple[str, str, str]:
    """Query, match and target rows in real residues for an alignment of the translations
    q_tr, t_tr of qa, ta. Match row: '|' same residue, ':' same class, ' ' otherwise."""
    c = aln.coordinates
    qrow, trow, mrow = [], [], []
    for k in range(c.shape[1] - 1):
        q0, q1, t0, t1 = c[0, k], c[0, k + 1], c[1, k], c[1, k + 1]
        if q1 > q0 and t1 > t0:
            for i, j in zip(range(q0, q1), range(t0, t1)):
                qrow.append(qa[i]); trow.append(ta[j])
                mrow.append("|" if qa[i] == ta[j] else (":" if q_tr[i] == t_tr[j] else " "))
        elif q1 > q0:
            qrow.append(qa[q0:q1]); trow.append("-" * (q1 - q0)); mrow.append(" " * (q1 - q0))
        else:
            qrow.append("-" * (t1 - t0)); trow.append(ta[t0:t1]); mrow.append(" " * (t1 - t0))
    return "".join(qrow), "".join(mrow), "".join(trow)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["manifest", "panel", "align", "select", "evaluate", "all"])
    ap.add_argument("--split", choices=["dev", "eval"], help="for `align`")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_out"].mkdir(parents=True, exist_ok=True)
    steps = {"manifest": [("manifest", None)], "panel": [("panel", None)],
             "align": [("align", args.split)], "select": [("select", None)],
             "evaluate": [("evaluate", None)],
             "all": [("manifest", None), ("panel", None), ("align", "dev"), ("select", None),
                     ("align", "eval"), ("evaluate", None)]}[args.step]
    for name, split in steps:
        t0 = time.perf_counter()
        if name == "manifest":
            step_manifest(cfg)
        elif name == "panel":
            step_panel(cfg)
            matrix_table(cfg).write_csv(cfg["_out"] / "matrices.tsv", separator="\t")
        elif name == "align":
            if split is None:
                sys.exit("align needs --split dev or --split eval")
            align_split(cfg, split)
        elif name == "select":
            step_select(cfg)
        elif name == "evaluate":
            step_evaluate(cfg)
        log_runtime(cfg, name, time.perf_counter() - t0, split or "")
        print(f"{name} {split or ''} done in {time.perf_counter() - t0:.1f} s, "
              f"peak RSS {peak_rss_mb():.0f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
