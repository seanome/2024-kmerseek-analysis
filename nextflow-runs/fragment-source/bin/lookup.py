"""Source-protein lookup for short peptide calls: the two workflows of notebook 354 and the
full-alphabet check they share.

Workflow K (kmerseek): index the database with a reduced alphabet, search the peptides,
take every target kmerseek reports as a candidate.
Workflow C (comparison): an exact full-alphabet k-mer index of the database, which finds
every protein that holds the whole peptide, plus blastp-short against the same database.
Its candidates are the union of the two.

Both workflows get the same peptide-call uncertainty: isoleucine and leucine cannot be told
apart by mass, and glutamine and lysine differ by 0.036 Da. massspec18 merges both pairs.
Workflow C and the check read every sequence with I written as L and Q written as K
(`normalise`), which is the same information.

The check (`verify`) aligns each candidate with the peptide in the full alphabet:
Smith-Waterman local alignment, PAM30, a gap of length g costing 9 + g (blastp-short's
defaults, gap open 9 and extend 1; parasail counts the first gap position in `open`, so
open = 10, extend = 1). Its score S ranks the candidates. Ties are broken by accession.
"""

from __future__ import annotations

import gzip
import os
import random
import resource
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import parasail

PARASAIL_OPEN = 10  # = BLAST gap open 9 + extend 1 for the first gap position
PARASAIL_EXTEND = 1
MATRIX = parasail.pam30
EXACT_K = 5  # word length of workflow C's exact index; shorter peptides are scanned directly
TOP_N = 5  # pass rule 1: accepted source within the top five
BLAST_EVALUE = 1000.0  # report weak blastp-short hits too; the check decides
BLAST_MAX_TARGETS = 500

_NORMALISE = str.maketrans({"I": "L", "Q": "K", "U": "X", "O": "X", "B": "X", "Z": "X",
                            "J": "X"})


def normalise(seq: str) -> str:
    """Write I as L and Q as K, the two mass-spec confusions massspec18 merges.

    The few non-standard letters in UniProt (U selenocysteine, O pyrrolysine, B, Z, J) become
    X, which PAM30 scores as unknown.
    """
    return seq.translate(_NORMALISE)


def read_fasta(path: str | Path) -> dict[str, str]:
    """{first header token: sequence}, gzip or plain."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, str] = {}
    name, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        out[name] = "".join(parts)
    return out


def write_fasta(records: dict[str, str], path: str | Path) -> None:
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n{seq}\n")


# ---------------------------------------------------------------- shuffles (controls)


def composition_shuffle(seq: str, rng: random.Random) -> str:
    """Same residues, random order."""
    letters = list(seq)
    rng.shuffle(letters)
    return "".join(letters)


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985, Mol Biol Evol 2:526).

    Copied unchanged from nextflow-runs/shared/decoys.py (seanome/2024-kmerseek-analysis
    PR 138, branch olgabot/shared-dipeptide-shuffle), itself notebook 256's copy.
    """
    if len(seq) < 3:
        return seq
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out[a].append(b)
    last = seq[-1]
    while True:
        final = {v: rng.choice(ends) for v, ends in out.items() if v != last}
        ok = True
        for v in final:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = final[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, ends in out.items():
        rest = list(ends)
        if v in final:
            rest.remove(final[v])
        rng.shuffle(rest)
        order[v] = rest + ([final[v]] if v in final else [])
    walk, u = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        u = order[u].pop(0)
        walk.append(u)
    return "".join(walk)


# ---------------------------------------------------------------- timing


@dataclass
class Timer:
    """Wall and CPU seconds of named steps. CPU counts this process and its children."""

    steps: dict[str, dict[str, float]] = field(default_factory=dict)

    def run(self, name: str, fn, *args, **kwargs):
        wall0 = time.perf_counter()
        self_cpu0 = time.process_time()
        child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        out = fn(*args, **kwargs)
        child1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = (child1.ru_utime - child0.ru_utime) + (child1.ru_stime - child0.ru_stime)
        step = self.steps.setdefault(name, {"wall_s": 0.0, "cpu_s": 0.0})
        step["wall_s"] += time.perf_counter() - wall0
        step["cpu_s"] += time.process_time() - self_cpu0 + child_cpu
        return out


def run_cmd(cmd: list[str], threads: int = 1) -> subprocess.CompletedProcess:
    """Run a command with RAYON_NUM_THREADS = threads; raise with its stderr on failure."""
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {proc.returncode}:\n{proc.stderr[-4000:]}")
    return proc


# ---------------------------------------------------------------- workflow K: kmerseek


def kmerseek_index(kmerseek: str, db_fasta: str, index_dir: str, alphabet: str, ksize: int,
                   threads: int = 1) -> None:
    run_cmd([kmerseek, "index", "--input", db_fasta, "--output", index_dir,
             "--ksize", str(ksize), "--alphabet", alphabet, "--scaled", "1",
             "--ka-queries", "0"], threads)


def kmerseek_search(kmerseek: str, queries_fasta: str, index_dir: str, alphabet: str,
                    ksize: int, out_csv: str, threads: int = 1) -> None:
    run_cmd([kmerseek, "search", "--query", queries_fasta, "--target", index_dir,
             "--alphabet", alphabet, "--ksize", str(ksize), "--output", out_csv,
             "--threshold", "0", "--min-shared-kmers", "1", "--max-query-pvalue", "1",
             "--extend-mismatch-penalty", "0"], threads)


def kmerseek_candidates(out_csv: str, query_col: str, target_col: str) -> dict[str, set[str]]:
    """{query: {target}} from a kmerseek search CSV."""
    import csv

    cands: dict[str, set[str]] = defaultdict(set)
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            cands[row[query_col].split()[0]].add(row[target_col].split()[0])
    return cands


# ---------------------------------------------------------------- workflow C: exact + blastp-short


def build_exact_index(db: dict[str, str], k: int = EXACT_K) -> dict[str, list[str]]:
    """{normalised k-mer: [accessions holding it]} over the whole database."""
    index: dict[str, set[str]] = defaultdict(set)
    for acc, seq in db.items():
        s = normalise(seq)
        for i in range(len(s) - k + 1):
            index[s[i:i + k]].add(acc)
    return {kmer: sorted(accs) for kmer, accs in index.items()}


def exact_lookup(peptides: dict[str, str], db: dict[str, str], index: dict[str, list[str]],
                 k: int = EXACT_K) -> dict[str, set[str]]:
    """{query: {accessions holding the whole normalised peptide}}."""
    norm_db_cache: dict[str, str] = {}
    out: dict[str, set[str]] = defaultdict(set)
    for qid, pep in peptides.items():
        p = normalise(pep)
        if len(p) < k:
            shortest = list(db)  # too short for the index: check every protein
        else:
            # The rarest of the peptide's k-mers gives the shortest list to check.
            lists = [index.get(p[i:i + k], []) for i in range(len(p) - k + 1)]
            shortest = min(lists, key=len)
        for acc in shortest:
            s = norm_db_cache.get(acc)
            if s is None:
                s = norm_db_cache[acc] = normalise(db[acc])
            if p in s:
                out[qid].add(acc)
    return out


def makeblastdb(db_fasta_normalised: str, blast_db: str) -> None:
    run_cmd(["makeblastdb", "-in", db_fasta_normalised, "-dbtype", "prot", "-out", blast_db,
             "-parse_seqids"])


def blastp_short(queries_fasta_normalised: str, blast_db: str, out_tsv: str, threads: int = 1) -> None:
    run_cmd(["blastp", "-task", "blastp-short", "-query", queries_fasta_normalised,
             "-db", blast_db, "-evalue", str(BLAST_EVALUE),
             "-max_target_seqs", str(BLAST_MAX_TARGETS), "-num_threads", str(threads),
             "-outfmt", "6 qseqid sseqid evalue bitscore", "-out", out_tsv])


def blast_candidates(out_tsv: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with open(out_tsv) as fh:
        for line in fh:
            q, s = line.split("\t")[:2]
            # -parse_seqids turns a bare accession into itself, or "lcl|acc"/"sp|acc|".
            s = s.split("|")[1] if s.startswith(("lcl|", "sp|", "tr|")) else s
            out[q].add(s)
    return out


# ---------------------------------------------------------------- the shared check


def verify(peptides: dict[str, str], cands: dict[str, set[str]], db: dict[str, str]) -> list[tuple[str, str, int]]:
    """Full-alphabet Smith-Waterman score of every (peptide, candidate) pair."""
    rows = []
    norm_db_cache: dict[str, str] = {}
    for qid, accs in cands.items():
        if not accs:
            continue
        profile = parasail.profile_create_16(normalise(peptides[qid]), MATRIX)
        for acc in accs:
            s = norm_db_cache.get(acc)
            if s is None:
                s = norm_db_cache[acc] = normalise(db[acc])
            score = parasail.sw_striped_profile_16(profile, s, PARASAIL_OPEN, PARASAIL_EXTEND).score
            rows.append((qid, acc, int(score)))
    return rows


def align(peptide: str, target: str) -> dict:
    """The check's alignment of one pair, in the original residues, for printing.

    Returns 1-based inclusive coordinates on both sequences, the three alignment lines, and
    how many aligned columns hold the same residue (n_identical of n_columns). `match` marks
    `|` for the same residue, `:` for I against L or Q against K (the same after
    `normalise`), and a space otherwise.
    """
    res = parasail.sw_trace_striped_16(normalise(peptide), normalise(target),
                                       PARASAIL_OPEN, PARASAIL_EXTEND, MATRIX)
    # parasail's traceback holds the aligned (normalised) strings with "-" for gaps, and
    # end_query / end_ref are the 0-based last aligned positions. Its cigar's begin fields
    # stay 0 for a local alignment, so the starts are counted back from the ends.
    tb_q, tb_t = res.traceback.query, res.traceback.ref
    while tb_q and (tb_q[0] == "-" or tb_t[0] == "-"):
        tb_q, tb_t = tb_q[1:], tb_t[1:]
    while tb_q and (tb_q[-1] == "-" or tb_t[-1] == "-"):
        tb_q, tb_t = tb_q[:-1], tb_t[:-1]
    q_start0 = res.end_query - sum(c != "-" for c in tb_q) + 1
    t_start0 = res.end_ref - sum(c != "-" for c in tb_t) + 1
    qi, ti = q_start0, t_start0
    q_line, t_line, m_line = [], [], []
    n_ident = n_cols = 0
    for cq, ct in zip(tb_q, tb_t):
        a = peptide[qi] if cq != "-" else "-"
        b = target[ti] if ct != "-" else "-"
        q_line.append(a)
        t_line.append(b)
        if a != "-" and b != "-":
            if a == b:
                m_line.append("|")
                n_ident += 1
            elif normalise(a) == normalise(b):
                m_line.append(":")
            else:
                m_line.append(" ")
        else:
            m_line.append(" ")
        qi += cq != "-"
        ti += ct != "-"
        n_cols += 1
    return {
        "score": int(res.score),
        "query_start": q_start0 + 1, "query_end": qi,
        "target_start": t_start0 + 1, "target_end": ti,
        "query_line": "".join(q_line), "match_line": "".join(m_line),
        "target_line": "".join(t_line),
        "n_identical": n_ident, "n_columns": n_cols,
    }
