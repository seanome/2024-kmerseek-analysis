"""Experiment 354: find the source protein of a de novo peptide call.

Compares full-alphabet exact lookup, full-alphabet local alignment, the two combined, and
reduced-alphabet k-mer search with and without full-alphabet verification. The rules are
in FROZEN_PLAN.md (same folder); this file implements them and nothing else.

Commands (run with the 2025-kmerseek-analysis Python):
    peptide_source.py validate INPUTS.json
    peptide_source.py prepare  INPUTS.json WORKDIR
    peptide_source.py build    WORKDIR METHOD
    peptide_source.py search   WORKDIR METHOD
    peptide_source.py evaluate WORKDIR
    peptide_source.py all      INPUTS.json WORKDIR
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- constants

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
CODE = {aa: i for i, aa in enumerate(CANONICAL)}
X_CODE = 20  # any non-canonical letter in a protein (U, B, Z, O, J, X)
SEP = 21  # between proteins
N_CODES = 22
CALL_LETTERS = set(CANONICAL) | {"X"}

# Reduced alphabets, copied from kmerseek src/rust/alphabets.rs (seanome/kmerseek main
# 00395d7, 2026-10-20) by way of nb300's nextflow-runs/peptide-lookup/bin/alphabets.py.
# massspec18 merges I/L (same mass) and Q/K (0.036 Da apart); it is in kmerseek PR 156.
ALPHABETS: dict[str, list[str]] = {
    "massspec18": ["A", "C", "D", "E", "F", "G", "H", "IL", "QK", "M", "N", "P", "R", "S",
                   "T", "V", "W", "Y"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "dayhoff6": ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"],
}

# Fixed in FROZEN_PLAN.md.
REDUCED_SETTINGS = {"massspec18": 5, "sdm12": 6, "dayhoff6": 8}
EXACT_K = 5
GAP_OPEN = 9  # BLAST short-query settings: a gap of n residues costs 9 + n
GAP_EXTEND = 1
WINDOW_PAD = 10
MAX_VARIANTS = 64
STORE_FLOOR = 0.2  # candidates below this match fraction are not stored
QVALUE_MAX = 0.01
ABSENT_SHARE = 0.20
FALSE_ASSIGNMENT_CEILING = 0.01
MIN_CALLS = 500
TOP_N = 5
BOOTSTRAP = 2000
SEED = 354
DAILY_BATCHES_PER_BUILD = 30

FULL_METHODS = ["exact", "align", "exact_then_align"]
REDUCED_METHODS = [f"reduced_{a}_k{k}" for a, k in REDUCED_SETTINGS.items()]
SEARCH_METHODS = FULL_METHODS + REDUCED_METHODS


def scored_methods() -> list[str]:
    """Every row of the result tables: each reduced search is scored alone and verified."""
    out = list(FULL_METHODS)
    for m in REDUCED_METHODS:
        out += [f"{m}_alone", f"{m}_verified"]
    return out


def index_families(method: str) -> list[str]:
    """Which indexes a search method loads."""
    if method == "exact_then_align":
        return ["exact", "align"]
    return [method]


def pam30_profile_table() -> np.ndarray:
    """PAM30 as a 20 x N_CODES table: query residue x target code."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("PAM30")
    t = np.zeros((20, N_CODES), dtype=np.int64)
    for a in CANONICAL:
        for b in CANONICAL:
            t[CODE[a], CODE[b]] = int(m[a, b])
        t[CODE[a], X_CODE] = int(m[a, "X"])
    t[:, SEP] = -1000
    return t


PAM30 = pam30_profile_table()

# ----------------------------------------------------------------------------- input


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """(accession, sequence). Accession is the second |-field of a UniProt header, else
    the first word."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records, name, parts = [], None, []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                word = line[1:].split()[0]
                fields = word.split("|")
                name = fields[1] if len(fields) >= 3 else word
                parts = []
            elif line:
                parts.append(line.upper())
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def parse_alternatives(text: str, length: int) -> dict[int, str]:
    """'3:KQ;7:N' -> {2: 'KQ', 6: 'N'} (0-based). Raises on a bad entry."""
    out: dict[int, str] = {}
    if not isinstance(text, str) or not text.strip():
        return out
    for entry in text.split(";"):
        pos, _, res = entry.partition(":")
        p = int(pos) - 1
        if not 0 <= p < length:
            raise ValueError(f"alternative position {pos} outside 1..{length}")
        if not res or set(res) - set(CANONICAL):
            raise ValueError(f"alternative residues {res!r} are not canonical")
        out[p] = out.get(p, "") + res
    return out


def parse_residue_scores(text, length: int) -> list[float] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != length:
        raise ValueError(f"{len(vals)} residue scores for {length} residues")
    return vals


@dataclass
class Inputs:
    data_origin: str
    proteins: list[tuple[str, str, str]]  # accession, sequence, role
    calls: pd.DataFrame
    accepted: pd.DataFrame
    checksums: dict[str, str]
    problems: list[str] = field(default_factory=list)


def load_inputs(inputs_json: Path) -> Inputs:
    """Read and check every input against the schema in FROZEN_PLAN.md. Problems that
    make the data unusable raise; problems in single rows are listed."""
    spec = json.loads(Path(inputs_json).read_text())
    base = Path(inputs_json).resolve().parent
    origin = spec["data_origin"]
    if origin not in ("experimental", "synthetic_fixture"):
        raise ValueError(f"data_origin must be experimental or synthetic_fixture, not {origin}")
    checksums, proteins, seen = {}, [], set()
    for part in spec["proteins"]:
        path = base / part["path"]
        checksums[part["path"]] = sha256_file(path)
        for acc, seq in read_fasta(path):
            if acc in seen:
                raise ValueError(f"accession {acc} appears twice in the collection")
            seen.add(acc)
            proteins.append((acc, seq, part["role"]))
    calls = pd.read_csv(base / spec["calls"], sep="\t", dtype=str, keep_default_na=False)
    accepted = pd.read_csv(base / spec["accepted"], sep="\t", dtype=str, keep_default_na=False)
    checksums[spec["calls"]] = sha256_file(base / spec["calls"])
    checksums[spec["accepted"]] = sha256_file(base / spec["accepted"])

    for col in ["call_id", "run_id", "scan_id", "called_sequence", "call_score"]:
        if col not in calls.columns:
            raise ValueError(f"calls.tsv lacks column {col}")
    for col in ["call_id", "accepted_peptide", "evidence_tool", "evidence_qvalue"]:
        if col not in accepted.columns:
            raise ValueError(f"accepted.tsv lacks column {col}")
    if calls["call_id"].duplicated().any():
        raise ValueError("calls.tsv has duplicate call_id")
    if accepted["call_id"].duplicated().any():
        raise ValueError("accepted.tsv has duplicate call_id")
    for col in ["residue_scores", "alternatives"]:
        if col not in calls.columns:
            calls[col] = ""

    problems = []
    for row in calls.itertuples():
        seq = row.called_sequence
        if not seq or set(seq) - CALL_LETTERS:
            raise ValueError(f"call {row.call_id}: called_sequence {seq!r} has letters "
                             "outside the 20 residues and X")
        float(row.call_score)
        parse_residue_scores(row.residue_scores, len(seq))
        parse_alternatives(row.alternatives, len(seq))
    for row in accepted.itertuples():
        if set(row.accepted_peptide) - set(CANONICAL):
            raise ValueError(f"accepted peptide for {row.call_id} is not a bare sequence")
        float(row.evidence_qvalue)
    missing = set(calls["call_id"]) - set(accepted["call_id"])
    if missing:
        problems.append(f"{len(missing)} calls have no accepted peptide")
    extra = set(accepted["call_id"]) - set(calls["call_id"])
    if extra:
        raise ValueError(f"{len(extra)} accepted rows name calls not in calls.tsv")
    return Inputs(origin, proteins, calls, accepted, checksums, problems)


# ----------------------------------------------------------------------------- database


@dataclass
class Database:
    accessions: list[str]
    codes: np.ndarray  # int8, SEP before, between and after proteins
    starts: np.ndarray  # first residue of each protein
    ends: np.ndarray  # one past its last residue

    def protein_of(self, pos: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.starts, pos, side="right") - 1


def encode_protein(seq: str) -> np.ndarray:
    lut = np.full(256, X_CODE, dtype=np.int8)
    for aa, c in CODE.items():
        lut[ord(aa)] = c
    return lut[np.frombuffer(seq.encode(), dtype=np.uint8)]


def build_database(proteins: list[tuple[str, str]]) -> Database:
    pieces, starts, ends, pos = [np.array([SEP], dtype=np.int8)], [], [], 1
    for _, seq in proteins:
        starts.append(pos)
        pieces.append(encode_protein(seq))
        pos += len(seq)
        ends.append(pos)
        pieces.append(np.array([SEP], dtype=np.int8))
        pos += 1
    return Database([a for a, _ in proteins], np.concatenate(pieces),
                    np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64))


def il_text(seq: str) -> str:
    return seq.replace("I", "L")


def valid_sources(peptides: list[str], proteins: list[tuple[str, str]]) -> list[list[str]]:
    """For each peptide, every protein containing it with I = L."""
    text = "*" + "*".join(il_text(s) for _, s in proteins) + "*"
    starts = np.cumsum([1] + [len(s) + 1 for _, s in proteins[:-1]])
    out = []
    for pep in peptides:
        pep = il_text(pep)
        hits, i = set(), text.find(pep)
        while i != -1:
            hits.add(int(np.searchsorted(starts, i, side="right") - 1))
            i = text.find(pep, i + 1)
        out.append(sorted(proteins[h][0] for h in hits))
    return out


# ----------------------------------------------------------------------------- queries


@dataclass
class Query:
    case_id: str
    seq: str
    alternatives: dict[int, str]
    excluded: frozenset[str]

    def allowed(self) -> list[set[str]]:
        """Residues allowed at each position: the call, its alternatives, I = L. X allows
        all 20."""
        out = []
        for i, aa in enumerate(self.seq):
            s = set(CANONICAL) if aa == "X" else {aa} | set(self.alternatives.get(i, ""))
            if s & {"I", "L"}:
                s |= {"I", "L"}
            out.append(s)
        return out

    def profile(self) -> np.ndarray:
        """L x N_CODES scores. An X position scores 0 against everything but SEP."""
        prof = np.zeros((len(self.seq), N_CODES), dtype=np.int64)
        for i, (aa, allowed) in enumerate(zip(self.seq, self.allowed())):
            if aa == "X":
                prof[i, SEP] = -1000
            else:
                prof[i] = PAM30[[CODE[a] for a in allowed]].max(axis=0)
                if "I" in allowed:  # I = L: a target I or L scores the better of the two
                    il = max(prof[i, CODE["I"]], prof[i, CODE["L"]])
                    prof[i, CODE["I"]] = prof[i, CODE["L"]] = il
        return prof

    def self_score(self) -> int:
        """Score of the call against its best possible target: the sum of row maxima,
        all of which are reached by one ungapped all-match path."""
        return int(self.profile()[:, :20].max(axis=1).clip(min=0).sum())


def allowed_mask(sets: list[set], n_classes: int, class_of: dict[str, int]) -> np.ndarray:
    """L x (n_classes + 2) boolean: may position i match class c. The two extra columns
    (X and SEP in the target) are never allowed."""
    m = np.zeros((len(sets), n_classes + 2), dtype=bool)
    for i, s in enumerate(sets):
        for aa in s:
            m[i, class_of[aa]] = True
    return m


def expand_window(mask: np.ndarray, cap: int = MAX_VARIANTS) -> list[tuple[int, ...]] | None:
    """Every combination of allowed classes along a window, or None if more than cap."""
    choices = [np.flatnonzero(row) for row in mask]
    n = math.prod(len(c) for c in choices)
    if n == 0 or n > cap:
        return None
    return list(itertools.product(*choices))


# ----------------------------------------------------------------------------- local alignment


def sw_best_per_protein(profile: np.ndarray, codes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Smith-Waterman with affine gaps (open GAP_OPEN, extend GAP_EXTEND) of one query
    profile against every protein in a SEP-separated array; returns each protein's best
    local score.

    Rows run over the query; each row is computed for all target positions at once. The
    gap along the target is a segmented running maximum, which is exact because opening a
    gap from a cell that already ends in a target gap never beats extending that gap."""
    n = len(codes)
    jj = np.arange(n, dtype=np.int64)
    is_sep = codes == SEP
    big = n + 10_000
    segkey = np.cumsum(is_sep).astype(np.int64) * big
    neg = -(10**9)
    h_prev = np.zeros(n, dtype=np.int64)
    e_prev = np.full(n, neg, dtype=np.int64)
    best = np.zeros(n, dtype=np.int64)
    open_cost = GAP_OPEN + GAP_EXTEND
    for row in profile:
        s = row[codes]
        diag = np.empty(n, dtype=np.int64)
        diag[0] = s[0]
        diag[1:] = h_prev[:-1] + s[1:]
        e = np.maximum(h_prev - open_cost, e_prev - GAP_EXTEND)
        hpre = np.maximum(np.maximum(diag, e), 0)
        hpre[is_sep] = 0
        run = np.maximum.accumulate(hpre + GAP_EXTEND * jj + segkey)
        f = np.full(n, neg, dtype=np.int64)
        f[1:] = run[:-1] - segkey[1:] - GAP_OPEN - GAP_EXTEND * jj[1:]
        h = np.maximum(hpre, f)
        h[is_sep] = 0
        e[is_sep] = neg
        np.maximum(best, h, out=best)
        h_prev, e_prev = h, e
    return np.maximum.reduceat(best, starts) if len(starts) else np.zeros(0, dtype=np.int64)


def verify_windows(query: Query, db: Database, prot: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """The shared verifier: match fraction of the call against each protein window
    [anchor - WINDOW_PAD, anchor + L + WINDOW_PAD), clipped to the protein."""
    if len(prot) == 0:
        return np.zeros(0)
    L = len(query.seq)
    lo = np.maximum(anchor - WINDOW_PAD, db.starts[prot])
    hi = np.minimum(anchor + L + WINDOW_PAD, db.ends[prot])
    pieces, wstarts, pos = [np.array([SEP], dtype=np.int8)], [], 1
    for a, b in zip(lo, hi):
        wstarts.append(pos)
        pieces.append(db.codes[a:b])
        pieces.append(np.array([SEP], dtype=np.int8))
        pos += b - a + 1
    scores = sw_best_per_protein(query.profile(), np.concatenate(pieces), np.array(wstarts))
    return scores / max(query.self_score(), 1)


# ----------------------------------------------------------------------------- k-mer index


@dataclass
class KmerIndex:
    k: int
    n_classes: int
    class_codes: np.ndarray  # per database position; n_classes = X, n_classes+1 = SEP
    positions: np.ndarray
    offsets: np.ndarray

    def lookup(self, kmer: tuple[int, ...]) -> np.ndarray:
        code = 0
        for c in kmer:
            code = code * self.n_classes + int(c)
        return self.positions[self.offsets[code]:self.offsets[code + 1]]


def class_map(alphabet: str | None) -> tuple[dict[str, int], int]:
    """Residue -> class. None is the 20-letter alphabet with I = L (19 classes used)."""
    if alphabet is None:
        m = {aa: CODE[aa] for aa in CANONICAL}
        m["I"] = CODE["L"]
        return m, 20
    return {aa: i for i, cl in enumerate(ALPHABETS[alphabet]) for aa in cl}, len(ALPHABETS[alphabet])


def build_kmer_index(db: Database, alphabet: str | None, k: int) -> KmerIndex:
    cmap, n_cls = class_map(alphabet)
    lut = np.full(N_CODES, n_cls, dtype=np.int16)
    for aa, c in cmap.items():
        lut[CODE[aa]] = c
    lut[SEP] = n_cls + 1
    cc = lut[db.codes]
    n = len(cc) - k + 1
    kcode = np.zeros(n, dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    for i in range(k):
        part = cc[i:i + n].astype(np.int64)
        ok &= part < n_cls
        kcode = kcode * n_cls + np.where(part < n_cls, part, 0)
    pos = np.flatnonzero(ok)
    keys = kcode[pos]
    order = np.argsort(keys, kind="stable")
    positions = pos[order].astype(np.int64)
    counts = np.bincount(keys, minlength=n_cls**k)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return KmerIndex(k, n_cls, cc, positions, offsets)


# ----------------------------------------------------------------------------- search methods
#
# Each returns a dict accession -> (score, extra_score). score is the verifier's match
# fraction (or, for "align", the same alignment run on the whole protein). extra_score is
# the reduced-alphabet match fraction for reduced methods, else nan. Proteins in
# query.excluded are dropped before anything is ranked or decided.


@dataclass
class Timer:
    verify_seconds: float = 0.0


def search_exact(q: Query, db: Database, idx: KmerIndex, timer: Timer) -> dict:
    cmap, n_cls = class_map(None)
    mask = allowed_mask(q.allowed(), n_cls, cmap)
    L, k = len(q.seq), idx.k
    best_window, best_cost = None, None
    for w in range(L - k + 1):
        combos = expand_window(mask[w:w + k])
        if combos is None:
            continue
        cost = sum(len(idx.lookup(c)) for c in combos)
        if best_cost is None or cost < best_cost:
            best_window, best_cost = (w, combos), cost
    if best_window is not None:
        w, combos = best_window
        starts = np.concatenate([idx.lookup(c) for c in combos] or [np.zeros(0, np.int64)]) - w
    else:  # nothing to seed with: test every start
        starts = np.arange(1, len(db.codes) - L)
    starts = starts[(starts >= 0) & (starts + L <= len(db.codes))]
    ok = np.ones(len(starts), dtype=bool)
    for i in range(L):
        ok &= mask[i][idx.class_codes[starts + i]]
    starts = np.unique(starts[ok])
    prot = db.protein_of(starts)
    keep = np.array([db.accessions[p] not in q.excluded for p in prot], dtype=bool)
    starts, prot = starts[keep], prot[keep]
    t0 = time.perf_counter()
    frac = verify_windows(q, db, prot, starts)
    timer.verify_seconds += time.perf_counter() - t0
    out: dict[str, tuple[float, float]] = {}
    for p, f in zip(prot, frac):
        acc = db.accessions[p]
        if acc not in out or f > out[acc][0]:
            out[acc] = (float(f), math.nan)
    return out


def search_align(q: Query, db: Database, timer: Timer) -> dict:
    t0 = time.perf_counter()
    frac = sw_best_per_protein(q.profile(), db.codes, db.starts) / max(q.self_score(), 1)
    timer.verify_seconds += time.perf_counter() - t0
    out = {}
    for p in np.flatnonzero(frac >= STORE_FLOOR):
        acc = db.accessions[p]
        if acc not in q.excluded:
            out[acc] = (float(frac[p]), math.nan)
    return out


def search_reduced(q: Query, db: Database, idx: KmerIndex, alphabet: str, timer: Timer) -> dict:
    cmap, n_cls = class_map(alphabet)
    mask = allowed_mask(q.allowed(), n_cls, cmap)
    L, k = len(q.seq), idx.k
    diags = []
    for w in range(L - k + 1):
        combos = expand_window(mask[w:w + k])
        if combos is None:
            continue
        for c in combos:
            hits = idx.lookup(c)
            if len(hits):
                diags.append(np.stack([hits - w, hits]))
    if not diags:
        return {}
    d = np.concatenate(diags, axis=1)
    diag, seedpos = d[0], d[1]
    diag, first = np.unique(diag, return_index=True)
    prot = db.protein_of(seedpos[first])
    keep = np.array([db.accessions[p] not in q.excluded for p in prot], dtype=bool)
    diag, prot = diag[keep], prot[keep]
    if len(diag) == 0:
        return {}
    cols = diag[:, None] + np.arange(L)[None, :]
    inside = (cols >= db.starts[prot][:, None]) & (cols < db.ends[prot][:, None])
    cls = idx.class_codes[np.clip(cols, 0, len(db.codes) - 1)]
    match = mask[np.arange(L)[None, :], cls] & inside
    red = match.sum(axis=1) / L
    # best diagonal per protein by reduced score
    order = np.lexsort((-red, prot))
    prot_s, red_s, diag_s = prot[order], red[order], diag[order]
    firsts = np.concatenate([[True], prot_s[1:] != prot_s[:-1]])
    prot_b, red_b, diag_b = prot_s[firsts], red_s[firsts], diag_s[firsts]
    t0 = time.perf_counter()
    frac = verify_windows(q, db, prot_b, diag_b)
    timer.verify_seconds += time.perf_counter() - t0
    return {db.accessions[p]: (float(f), float(r)) for p, f, r in zip(prot_b, frac, red_b)}


# ----------------------------------------------------------------------------- prepare


def low_complexity(seq: str) -> bool:
    counts = sorted((seq.count(a) for a in set(seq)), reverse=True)
    return sum(counts[:2]) >= len(seq) / 2


def length_bin(n: int) -> str:
    for lo, hi in [(7, 9), (10, 12), (13, 16), (17, 20)]:
        if n <= hi:
            return f"{lo}-{hi}" if n >= lo else f"<{lo}"
    return "21+"


def unit_hash(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16) / 16**15


def group_proteins(source_sets: list[list[str]]) -> dict[str, str]:
    """Union-find: protein -> smallest accession of its group."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in source_sets:
        for a in s:
            find(a)
        for a, b in zip(s, s[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {a: find(a) for a in parent}


def shuffle_composition(seq: str, scores, rng: random.Random):
    idx = list(range(len(seq)))
    rng.shuffle(idx)
    return "".join(seq[i] for i in idx), ([scores[i] for i in idx] if scores else None)


def shuffle_pairs(seq: str, rng: random.Random) -> str:
    """A random walk through the call's adjacent-residue pairs that uses each pair once
    (an Euler path, Hierholzer's method with shuffled edge order). Keeps every pair count
    and the first and last residue. Not a uniform draw over such sequences."""
    if len(seq) < 3:
        return seq
    edges: dict[str, list[str]] = {}
    for a, b in zip(seq, seq[1:]):
        edges.setdefault(a, []).append(b)
    for v in edges.values():
        rng.shuffle(v)
    stack, path = [seq[0]], []
    while stack:
        v = stack[-1]
        if edges.get(v):
            stack.append(edges[v].pop())
        else:
            path.append(stack.pop())
    return "".join(reversed(path))


def prepare(inputs_json: Path, workdir: Path) -> dict:
    """Validate, find valid sources, split, mark source-absent calls, make shuffled
    controls. Writes cases.tsv, checksums.tsv, proteins.fasta and prepare.json."""
    workdir.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(inputs_json)
    report: dict = {"data_origin": inp.data_origin, "problems": inp.problems,
                    "n_proteins": len(inp.proteins),
                    "n_contaminants": sum(r == "contaminant" for *_, r in inp.proteins),
                    "n_calls_in": len(inp.calls)}
    df = inp.calls.merge(inp.accepted, on="call_id", how="left")
    no_acc = df["accepted_peptide"].isna() | (df["accepted_peptide"] == "")
    df = df[~no_acc].copy()
    report["excluded_no_accepted_peptide"] = int(no_acc.sum())
    high_q = df["evidence_qvalue"].astype(float) > QVALUE_MAX
    report["excluded_qvalue_above_0.01"] = int(high_q.sum())
    df = df[~high_q].copy()
    plain = [(a, s) for a, s, _ in inp.proteins]
    df["valid_sources"] = valid_sources(df["accepted_peptide"].tolist(), plain)
    none = df["valid_sources"].str.len() == 0
    report["excluded_peptide_not_in_collection"] = int(none.sum())
    df = df[~none].copy()
    report["n_calls_used"] = len(df)

    group = group_proteins(df["valid_sources"].tolist())
    df["group"] = [group[s[0]] for s in df["valid_sources"]]
    df["side"] = ["development" if unit_hash("354:" + g) < 0.5 else "evaluation"
                  for g in df["group"]]
    df["absent"] = False
    for side in ["development", "evaluation"]:
        sub = df[df["side"] == side]
        n_abs = int(math.floor(ABSENT_SHARE * len(sub)))
        chosen = sorted(sub["call_id"], key=lambda c: unit_hash("354-absent:" + c))[:n_abs]
        df.loc[df["call_id"].isin(chosen), "absent"] = True
    report["n_by_side_and_kind"] = {f"{s}/{'absent' if a else 'present'}": int(n)
                                    for (s, a), n in df.groupby(["side", "absent"]).size().items()}

    cases = []
    for r in df.itertuples():
        cases.append(dict(case_id=r.call_id, call_id=r.call_id, side=r.side,
                          kind="source_absent" if r.absent else "source_present",
                          query=r.called_sequence, alternatives=r.alternatives,
                          residue_scores=r.residue_scores, call_score=float(r.call_score),
                          accepted_peptide=r.accepted_peptide,
                          valid_sources=";".join(r.valid_sources),
                          excluded=";".join(r.valid_sources) if r.absent else ""))
    # shuffled controls of evaluation calls; alternatives dropped because positions move
    text = "*" + "*".join(il_text(s) for _, s in plain) + "*"
    rng = random.Random(SEED)
    dropped = {"composition": 0, "pairs": 0}
    for r in df[df["side"] == "evaluation"].itertuples():
        for kind in ["composition", "pairs"]:
            for _ in range(20):
                if kind == "composition":
                    q, _s = shuffle_composition(r.called_sequence, None, rng)
                else:
                    q = shuffle_pairs(r.called_sequence, rng)
                if il_text(q) not in text:
                    break
            else:
                dropped[kind] += 1
                continue
            cases.append(dict(case_id=f"{r.call_id}__shuffle_{kind}", call_id=r.call_id,
                              side="evaluation", kind=f"control_{kind}", query=q,
                              alternatives="", residue_scores="", call_score=float(r.call_score),
                              accepted_peptide="", valid_sources="", excluded=""))
    report["controls_dropped_still_in_collection"] = dropped
    cases_df = pd.DataFrame(cases)
    cases_df.to_csv(workdir / "cases.tsv", sep="\t", index=False)
    with open(workdir / "proteins.fasta", "w") as f:
        for acc, seq in plain:
            f.write(f">{acc}\n{seq}\n")
    pd.Series(inp.checksums, name="sha256").rename_axis("file").to_csv(
        workdir / "checksums.tsv", sep="\t")
    report["checksums"] = inp.checksums
    report["frozen_plan_sha256"] = sha256_file(HERE / "FROZEN_PLAN.md")
    (workdir / "prepare.json").write_text(json.dumps(report, indent=2))
    return report


# ----------------------------------------------------------------------------- build and search


def usage() -> dict:
    r = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux
    return {"cpu": r.ru_utime + r.ru_stime, "maxrss_bytes": r.ru_maxrss * scale}


def load_db(workdir: Path) -> Database:
    return build_database(read_fasta(workdir / "proteins.fasta"))


def index_path(workdir: Path, family: str) -> Path:
    return workdir / "index" / f"{family}.npz"


def build(workdir: Path, family: str) -> dict:
    """Build one index and save it; timing goes to timing/build_<family>.json."""
    t0, u0 = time.perf_counter(), usage()
    db = load_db(workdir)
    path = index_path(workdir, family)
    path.parent.mkdir(exist_ok=True)
    if family == "align":
        np.savez(path, codes=db.codes, starts=db.starts, ends=db.ends)
    else:
        alphabet, k = (None, EXACT_K) if family == "exact" else (
            family.split("_")[1], int(family.rsplit("_k", 1)[1]))
        idx = build_kmer_index(db, alphabet, k)
        np.savez(path, codes=db.codes, starts=db.starts, ends=db.ends, k=k,
                 n_classes=idx.n_classes, class_codes=idx.class_codes,
                 positions=idx.positions, offsets=idx.offsets)
    u1 = usage()
    out = {"family": family, "wall_seconds": time.perf_counter() - t0,
           "cpu_seconds": u1["cpu"] - u0["cpu"], "peak_rss_bytes": u1["maxrss_bytes"],
           "disk_bytes": path.stat().st_size}
    (workdir / "timing").mkdir(exist_ok=True)
    (workdir / "timing" / f"build_{family}.json").write_text(json.dumps(out, indent=2))
    return out


def load_index(workdir: Path, family: str, accessions: list[str]):
    z = np.load(index_path(workdir, family))
    db = Database(accessions, z["codes"], z["starts"], z["ends"])
    if family == "align":
        return db, None
    return db, KmerIndex(int(z["k"]), int(z["n_classes"]), z["class_codes"], z["positions"],
                         z["offsets"])


def read_cases(workdir: Path) -> pd.DataFrame:
    return pd.read_csv(workdir / "cases.tsv", sep="\t", dtype=str, keep_default_na=False)


def to_query(r) -> Query:
    excl = frozenset(a for a in r.excluded.split(";") if a)
    return Query(r.case_id, r.query, parse_alternatives(r.alternatives, len(r.query)), excl)


def run_method(method: str, queries: list[Query], loaded: dict, timer: Timer) -> list[dict]:
    out = []
    for q in queries:
        if method == "exact":
            res = search_exact(q, *loaded["exact"], timer)
        elif method == "align":
            res = search_align(q, loaded["align"][0], timer)
        elif method == "exact_then_align":
            res = search_exact(q, *loaded["exact"], timer)
            if not res:
                res = search_align(q, loaded["align"][0], timer)
        else:
            alphabet = method.split("_")[1]
            res = search_reduced(q, *loaded[method], alphabet, timer)
        out.append(res)
    return out


def search(workdir: Path, method: str) -> dict:
    """Load indexes in this new process (cold), run every case, run them again (warm).
    Writes candidates/<method>.tsv.gz and timing/search_<method>.json."""
    cases = read_cases(workdir)
    queries = [to_query(r) for r in cases.itertuples()]
    accessions = [a for a, _ in read_fasta(workdir / "proteins.fasta")]
    t0, u0 = time.perf_counter(), usage()
    loaded = {f: load_index(workdir, f, accessions) for f in index_families(method)}
    t_load = time.perf_counter() - t0
    timings = {"method": method, "n_queries": len(queries), "load_seconds": t_load}
    results = None
    for label in ["cold", "warm"]:
        timer = Timer()
        ta, ua = time.perf_counter(), usage()
        results = run_method(method, queries, loaded, timer)
        ub = usage()
        timings[f"{label}_wall_seconds"] = time.perf_counter() - ta + (t_load if label == "cold" else 0)
        timings[f"{label}_cpu_seconds"] = ub["cpu"] - ua["cpu"] + (ua["cpu"] - u0["cpu"] if label == "cold" else 0)
        timings[f"{label}_verify_seconds"] = timer.verify_seconds
    timings["peak_rss_bytes"] = usage()["maxrss_bytes"]
    rows = []
    for q, res in zip(queries, results):
        for acc, (score, red) in res.items():
            rows.append((q.case_id, acc, score, red))
    (workdir / "candidates").mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["case_id", "accession", "match_fraction", "reduced_fraction"]
                 ).to_csv(workdir / "candidates" / f"{method}.tsv.gz", sep="\t", index=False)
    (workdir / "timing" / f"search_{method}.json").write_text(json.dumps(timings, indent=2))
    return timings


# ----------------------------------------------------------------------------- evaluate


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class CaseScores:
    """Per case: best valid-source score and every candidate score, both after masking."""
    best_valid: float  # nan when no valid source is among the candidates
    wrong: np.ndarray  # scores of candidates that are not valid sources, sorted descending
    all_scores: np.ndarray


def case_scores(cases: pd.DataFrame, cand: pd.DataFrame, column: str) -> dict[str, CaseScores]:
    valid = {r.case_id: set(filter(None, r.valid_sources.split(";")))
             for r in cases.itertuples()}
    present = {r.case_id: r.kind == "source_present" for r in cases.itertuples()}
    out = {cid: CaseScores(float("nan"), np.zeros(0), np.zeros(0)) for cid in cases["case_id"]}
    for cid, g in cand.groupby("case_id"):
        s = g[column].to_numpy(dtype=float)
        is_valid = g["accession"].isin(valid[cid]).to_numpy() if present[cid] else np.zeros(len(g), bool)
        bv = s[is_valid].max() if is_valid.any() else float("nan")
        out[cid] = CaseScores(bv, np.sort(s[~is_valid])[::-1], np.sort(s)[::-1])
    return out


def rank_of_valid(cs: CaseScores) -> float:
    if math.isnan(cs.best_valid):
        return float("inf")
    return 1 + int((cs.wrong >= cs.best_valid).sum())


def fit_threshold(absent_dev: list[CaseScores]) -> float:
    """Smallest t with at most 1% of development source-absent cases holding any candidate
    at or above t."""
    tops = sorted((cs.all_scores[0] if len(cs.all_scores) else -1.0 for cs in absent_dev),
                  reverse=True)
    allowed = int(math.floor(FALSE_ASSIGNMENT_CEILING * len(tops)))
    if allowed >= len(tops) or tops[allowed] < STORE_FLOOR:
        return STORE_FLOOR
    return float(np.nextafter(tops[allowed], 2.0))


def recovered(cs: CaseScores, t: float) -> bool:
    return rank_of_valid(cs) <= TOP_N and cs.best_valid >= t


def confident(cs: CaseScores, t: float) -> bool:
    return len(cs.all_scores) > 0 and cs.all_scores[0] >= t


def n_surviving(cs: CaseScores, t: float) -> int:
    return int((cs.all_scores >= t).sum())


def strata(row) -> dict[str, str]:
    q = row.query
    return {
        "low_complexity": "yes" if low_complexity(q) else "no",
        "ambiguous": "yes" if ("X" in q or row.alternatives) else "no",
        "call_matches_accepted": "yes" if il_text(q) == il_text(row.accepted_peptide) else "no",
        "length": length_bin(len(q)),
    }


def evaluate(workdir: Path) -> dict:
    """Thresholds from development cases, every measure on evaluation cases, verdict."""
    cases = read_cases(workdir)
    prep = json.loads((workdir / "prepare.json").read_text())
    cases["call_score"] = cases["call_score"].astype(float)
    dev = cases[cases["side"] == "development"]
    edges = np.quantile(dev[dev["kind"].str.startswith("source")]["call_score"], [0.25, 0.5, 0.75])
    st = {r.case_id: strata(r) | {"call_score_quartile": f"Q{1 + int(np.searchsorted(edges, r.call_score, side='right'))}"}
          for r in cases.itertuples()}
    rows, curves, counts, per_query, scored = [], [], [], [], {}
    grid = np.round(np.arange(0.2, 1.0001, 0.005), 3)
    for method in scored_methods():
        base = method.rsplit("_", 1)[0] if method.endswith(("_alone", "_verified")) else method
        column = "reduced_fraction" if method.endswith("_alone") else "match_fraction"
        cand = pd.read_csv(workdir / "candidates" / f"{base}.tsv.gz", sep="\t")
        cs = case_scores(cases, cand, column)
        scored[method] = cs
        side = {s: cases[cases["side"] == s] for s in ["development", "evaluation"]}
        t = fit_threshold([cs[c] for c in side["development"].query("kind == 'source_absent'")["case_id"]])
        ev = side["evaluation"]
        pres = [cs[c] for c in ev.query("kind == 'source_present'")["case_id"]]
        absn = [cs[c] for c in ev.query("kind == 'source_absent'")["case_id"]]
        k_rec = sum(recovered(c, t) for c in pres)
        k_fa = sum(confident(c, t) for c in absn)
        surv = [n_surviving(c, t) for c in pres]
        row = dict(method=method, threshold=t, n_present=len(pres), recovered=k_rec,
                   recovery=k_rec / max(len(pres), 1),
                   recovery_ci=wilson(k_rec, len(pres)),
                   n_absent=len(absn), false_assignments=k_fa,
                   false_assignment_rate=k_fa / max(len(absn), 1),
                   false_assignment_ci=wilson(k_fa, len(absn)),
                   median_surviving=float(np.median(surv)) if surv else float("nan"),
                   max_surviving=int(max(surv)) if surv else 0)
        for kind in ["control_composition", "control_pairs"]:
            ctl = [cs[c] for c in ev.query("kind == @kind")["case_id"]]
            kk = sum(confident(c, t) for c in ctl)
            row[f"{kind}_confident"] = kk
            row[f"{kind}_n"] = len(ctl)
        rows.append(row)
        for tt in grid:
            curves.append(dict(method=method, threshold=tt,
                               recovery=sum(recovered(c, tt) for c in pres) / max(len(pres), 1),
                               false_assignment_rate=sum(confident(c, tt) for c in absn) / max(len(absn), 1)))
        for cid, c in zip(ev.query("kind == 'source_present'")["case_id"], surv):
            counts.append(dict(method=method, case_id=cid, n_surviving=c))
        # strata
        for name in ["low_complexity", "ambiguous", "call_matches_accepted", "length", "call_score_quartile"]:
            for kind, pool in [("source_present", ev.query("kind == 'source_present'")),
                               ("source_absent", ev.query("kind == 'source_absent'"))]:
                for value, g in pool.groupby(pool["case_id"].map(lambda c: st[c][name])):
                    hit = sum(recovered(cs[c], t) if kind == "source_present" else confident(cs[c], t)
                              for c in g["case_id"])
                    per_query.append(dict(method=method, stratum=name, value=value, kind=kind,
                                          count=hit, n=len(g)))
        # per-query candidate lists: everything at or above t, the best valid source, top 10
        valid = dict(zip(cases["case_id"], cases["valid_sources"]))
        cand = cand.merge(cases[["case_id", "side", "kind"]], on="case_id")
        cand["score"] = cand[column]
        cand = cand.sort_values(["case_id", "score"], ascending=[True, False])
        cand["rank_in_list"] = cand.groupby("case_id").cumcount() + 1
        cand["is_valid_source"] = [a in valid[c].split(";") and k == "source_present"
                                   for a, c, k in zip(cand["accession"], cand["case_id"], cand["kind"])]
        keep = (cand["score"] >= t) | (cand["rank_in_list"] <= 10) | cand["is_valid_source"]
        out = cand[keep].assign(method=method, at_or_above_threshold=cand["score"] >= t)
        out_path = workdir / "results" / "candidate_lists" / f"{method}.tsv.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out[["method", "case_id", "side", "kind", "rank_in_list", "accession", "score",
             "is_valid_source", "at_or_above_threshold"]].to_csv(out_path, sep="\t", index=False)

    res = pd.DataFrame(rows)
    res_dir = workdir / "results"
    res.to_csv(res_dir / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(curves).to_csv(res_dir / "recall_vs_false_assignment.tsv", sep="\t", index=False)
    pd.DataFrame(counts).to_csv(res_dir / "surviving_candidates.tsv", sep="\t", index=False)
    pd.DataFrame(per_query).to_csv(res_dir / "strata.tsv", sep="\t", index=False)
    runtime = runtime_table(workdir)
    runtime.to_csv(res_dir / "runtime.tsv", sep="\t", index=False)
    verdict = decide(prep, res, runtime, scored, cases)
    (res_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    return verdict


def runtime_table(workdir: Path) -> pd.DataFrame:
    rows = []
    for method in scored_methods():
        base = method.rsplit("_", 1)[0] if method.endswith(("_alone", "_verified")) else method
        s = json.loads((workdir / "timing" / f"search_{base}.json").read_text())
        builds = [json.loads((workdir / "timing" / f"build_{f}.json").read_text())
                  for f in index_families(base)]
        skip_verify = method.endswith("_alone")
        warm = s["warm_wall_seconds"] - (s["warm_verify_seconds"] if skip_verify else 0)
        cold = s["cold_wall_seconds"] - (s["cold_verify_seconds"] if skip_verify else 0)
        build_wall = sum(b["wall_seconds"] for b in builds)
        rows.append(dict(
            method=method, n_queries=s["n_queries"],
            build_wall_seconds=build_wall,
            build_cpu_seconds=sum(b["cpu_seconds"] for b in builds),
            build_peak_rss_bytes=max(b["peak_rss_bytes"] for b in builds),
            index_disk_bytes=sum(b["disk_bytes"] for b in builds),
            cold_batch_wall_seconds=cold, warm_batch_wall_seconds=warm,
            cold_batch_cpu_seconds=s["cold_cpu_seconds"], warm_batch_cpu_seconds=s["warm_cpu_seconds"],
            verify_seconds_warm=s["warm_verify_seconds"],
            search_peak_rss_bytes=s["peak_rss_bytes"],
            amortized_daily_seconds=build_wall / DAILY_BATCHES_PER_BUILD + warm,
            cpu_hours_total=(sum(b["cpu_seconds"] for b in builds) + s["cold_cpu_seconds"]
                             + s["warm_cpu_seconds"]) / 3600))
    return pd.DataFrame(rows)


def paired_bootstrap(a: list[bool], b: list[bool]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    a, b = np.array(a, float), np.array(b, float)
    if len(a) == 0:
        return (float("nan"), math.nan)
    idx = rng.integers(0, len(a), size=(BOOTSTRAP, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))


def decide(prep: dict, res: pd.DataFrame, runtime: pd.DataFrame, scored: dict,
           cases: pd.DataFrame) -> dict:
    """The pass rule of FROZEN_PLAN.md, computed in full even when the data force
    INSUFFICIENT_EVIDENCE, so the software is checked end to end."""
    r = res.set_index("method")
    rt = runtime.set_index("method")
    pres_ids = cases.query("side == 'evaluation' and kind == 'source_present'")["case_id"]
    full_best = max(FULL_METHODS, key=lambda m: r.loc[m, "recovery"])
    per_setting = {}
    for m in REDUCED_METHODS:
        v = f"{m}_verified"
        gates = {
            "gate1_recovery_ge_0.95": bool(r.loc[v, "recovery"] >= 0.95),
            "gate2_false_assignment_le_0.01": bool(r.loc[v, "false_assignment_rate"] <= 0.01),
            "gate3_median_surviving_le_5": bool(r.loc[v, "median_surviving"] <= 5),
        }
        delta = r.loc[v, "recovery"] - r.loc[full_best, "recovery"]
        rec_v = [recovered(scored[v][c], r.loc[v, "threshold"]) for c in pres_ids]
        rec_f = [recovered(scored[full_best][c], r.loc[full_best, "threshold"]) for c in pres_ids]
        matched = [f for f in FULL_METHODS
                   if r.loc[f, "recovery"] >= r.loc[v, "recovery"] - 0.01
                   and r.loc[f, "false_assignment_rate"] <= 0.01]
        fastest = min(matched, key=lambda f: rt.loc[f, "amortized_daily_seconds"]) if matched else None
        speedup = (rt.loc[fastest, "amortized_daily_seconds"] / rt.loc[v, "amortized_daily_seconds"]
                   if fastest else math.nan)
        crit = {"A_recovery_gain_ge_10pp": bool(delta >= 0.10),
                "B_runtime_5x_lower_at_matched_recovery": bool(fastest is not None and speedup >= 5)}
        gain_vanishes = bool(r.loc[f"{m}_alone", "recovery"] > r.loc[full_best, "recovery"]
                             and delta <= 0)
        per_setting[m] = dict(gates=gates, criteria=crit,
                              recovery_gain_vs_strongest_full=delta,
                              recovery_gain_ci=paired_bootstrap(rec_v, rec_f),
                              matched_full_method=fastest, speedup=speedup,
                              gain_vanishes_after_verification=gain_vanishes,
                              passes=all(gates.values()) and any(crit.values()))
    full_meets_need = {f: bool(r.loc[f, "recovery"] >= 0.95 and r.loc[f, "false_assignment_rate"] <= 0.01
                               and r.loc[f, "median_surviving"] <= 5) for f in FULL_METHODS}
    computed = "PASS" if any(s["passes"] for s in per_setting.values()) else "FAIL"
    reasons = []
    if prep["data_origin"] != "experimental":
        reasons.append(f"data_origin is {prep['data_origin']}: synthetic calls cannot show usefulness")
    if prep["n_calls_used"] < MIN_CALLS:
        reasons.append(f"{prep['n_calls_used']} calls left after exclusions, fewer than {MIN_CALLS}")
    n_abs = int(r.loc[f"{REDUCED_METHODS[0]}_verified", "n_absent"])
    return dict(
        verdict="INSUFFICIENT_EVIDENCE" if reasons else computed,
        reasons=reasons,
        verdict_if_data_were_real=computed,
        strongest_full_method=full_best,
        full_method_meets_gates_1_to_3=full_meets_need,
        settings=per_setting,
        source_absent_note=(f"{n_abs} source-absent evaluation calls; 0 false assignments "
                            f"would still allow a true rate up to {wilson(0, n_abs)[1]:.1%} "
                            "(Wilson 95% upper bound)"),
    )


# ----------------------------------------------------------------------------- driver


def run_all(inputs_json: Path, workdir: Path) -> dict:
    """Every step, each build and search in its own process so cold timing is a fresh load."""
    prepare(inputs_json, workdir)
    me = [sys.executable, str(Path(__file__).resolve())]
    families = sorted({f for m in SEARCH_METHODS for f in index_families(m)})
    for f in families:
        subprocess.run(me + ["build", str(workdir), f], check=True)
    for m in SEARCH_METHODS:
        subprocess.run(me + ["search", str(workdir), m], check=True)
    return evaluate(workdir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").add_argument("inputs", type=Path)
    for name in ["prepare", "all"]:
        sp = sub.add_parser(name)
        sp.add_argument("inputs", type=Path)
        sp.add_argument("workdir", type=Path)
    for name in ["build", "search"]:
        sp = sub.add_parser(name)
        sp.add_argument("workdir", type=Path)
        sp.add_argument("target")
    sub.add_parser("evaluate").add_argument("workdir", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "validate":
        inp = load_inputs(a.inputs)
        print(json.dumps({"data_origin": inp.data_origin, "n_proteins": len(inp.proteins),
                          "n_calls": len(inp.calls), "problems": inp.problems,
                          "checksums": inp.checksums}, indent=2))
    elif a.cmd == "prepare":
        print(json.dumps(prepare(a.inputs, a.workdir), indent=2))
    elif a.cmd == "build":
        print(json.dumps(build(a.workdir, a.target)))
    elif a.cmd == "search":
        print(json.dumps(search(a.workdir, a.target)))
    elif a.cmd == "evaluate":
        print(json.dumps(evaluate(a.workdir), indent=2, default=str))
    elif a.cmd == "all":
        print(json.dumps(run_all(a.inputs, a.workdir), indent=2, default=str))


if __name__ == "__main__":
    main()
