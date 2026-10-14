"""The k-mer offset test for rearranged protein pairs (notebook 293).

Two proteins built from the same parts in a different order (a circular permutation, or
two domains swapped) share k-mers whose position differences fall into two groups. For
every k-mer the two proteins share, offset = position in B minus position in A. Colinear
homologs put their shared k-mers on one offset (or a few close ones, after indels). A
rearranged pair puts them on two offsets, and the k-mers on the two offsets come in one
order along A and the reverse order along B. This is a dot plot read through k-mers: each
offset is a diagonal of the dot plot. The idea is old; notebook 293 tests it at scale and
in reduced alphabets.

Terms used in the code and the tables:
    setting   one alphabet, one k, one n_block (minimum shared k-mers in a block)
    block     a run of adjacent offset bins (BIN_WIDTH residues each) that together hold
              at least n_block shared k-mer occurrences
    called    at least two blocks, whose k-mer ranges overlap by at most MAX_OVERLAP
              residues in A and in B, and whose order along B is the reverse of their
              order along A
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

# Alphabets: the three that sourmash also sketches (protein, dayhoff, hp), which kmerseek
# calls protein20, dayhoff6 and hp_lehninger2 (kmerseek src/rust/alphabets.rs,
# canonical_moltype). Tables read from sourmash 4.9.4 sourmash_aa_to_dayhoff and
# sourmash_aa_to_hp on 2026-10-14.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
DAYHOFF6 = {
    "C": "a",
    **dict.fromkeys("AGPST", "b"),
    **dict.fromkeys("DENQ", "c"),
    **dict.fromkeys("HKR", "d"),
    **dict.fromkeys("ILMV", "e"),
    **dict.fromkeys("FWY", "f"),
}
HP_LEHNINGER2 = {
    **dict.fromkeys("AFGILMPVWY", "h"),
    **dict.fromkeys("CDEHKNQRST", "p"),
}
PROTEIN20 = {a: a for a in CANONICAL_AA}
ALPHABETS = {
    "protein20": PROTEIN20,
    "dayhoff6": DAYHOFF6,
    "hp_lehninger2": HP_LEHNINGER2,
}

# k per alphabet, from the experiment brief.
KSIZES = {
    "protein20": [5, 6, 7],
    "dayhoff6": [8, 10, 12],
    "hp_lehninger2": [15, 20, 25],
}
N_BLOCKS = [3, 5, 8]
BIN_WIDTH = 3  # residues per offset bin
MAX_OCCURRENCES = 5  # a k-mer seen more often than this in either protein is dropped
# Two blocks (or two MMseqs2 alignments) may overlap by this many residues in each protein
# and still count as separate. MMseqs2's two local alignments of a circular permutation run
# a few residues past the junction (1 to 6 residues on the concanavalin A pairs of T1), so
# a rule of no overlap at all would miss them. The same allowance is used for both tests.
MAX_OVERLAP = 10


def encode(seq: str, alphabet: str) -> str:
    """Encode a protein; residues outside the 20 canonical ones become '*'.

    A k-mer containing '*' is never used, so X, U, B, Z and O break k-mers.
    """
    table = ALPHABETS[alphabet]
    return "".join(table.get(a, "*") for a in seq.upper())


def kmer_positions(encoded: str, k: int) -> dict[str, list[int]]:
    """0-based start positions of every k-mer without '*'."""
    pos: dict[str, list[int]] = defaultdict(list)
    for i in range(len(encoded) - k + 1):
        kmer = encoded[i : i + k]
        if "*" not in kmer:
            pos[kmer].append(i)
    return pos


def shared_kmer_hits(
    seq_a: str,
    seq_b: str,
    alphabet: str,
    k: int,
    max_occurrences: int = MAX_OCCURRENCES,
) -> tuple[list[tuple[int, int]], int]:
    """Every occurrence pair (position in A, position in B) of every shared k-mer.

    K-mers seen more than `max_occurrences` times in either protein are dropped (low
    complexity). Returns the hits, sorted, and how many distinct shared k-mers were dropped.
    """
    pa = kmer_positions(encode(seq_a, alphabet), k)
    pb = kmer_positions(encode(seq_b, alphabet), k)
    hits = []
    n_dropped = 0
    for kmer, ia in pa.items():
        ib = pb.get(kmer)
        if ib is None:
            continue
        if len(ia) > max_occurrences or len(ib) > max_occurrences:
            n_dropped += 1
            continue
        hits.extend((i, j) for i in ia for j in ib)
    hits.sort()
    return hits, n_dropped


@dataclass
class Block:
    """A run of adjacent offset bins holding at least n_block shared k-mer occurrences."""

    offset_lo: int  # smallest offset in the block (residues, position in B minus in A)
    offset_hi: int
    n_kmers: int  # shared k-mer occurrences in the block
    a_start: (
        int  # 0-based, end-exclusive residue range in A covered by the block's k-mers
    )
    a_end: int
    b_start: int
    b_end: int
    offset_median: int


def find_blocks(hits: list[tuple[int, int]], k: int, n_block: int) -> list[Block]:
    """Group hits by offset bin; merge adjacent non-empty bins; keep runs of >= n_block."""
    by_bin: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, j in hits:
        by_bin[(j - i) // BIN_WIDTH].append((i, j))
    blocks = []
    run: list[tuple[int, int]] = []
    prev = None
    for b in sorted(by_bin):
        if prev is not None and b != prev + 1:
            blocks.append(run)
            run = []
        run.extend(by_bin[b])
        prev = b
    if run:
        blocks.append(run)
    out = []
    for r in blocks:
        if len(r) < n_block:
            continue
        offsets = sorted(j - i for i, j in r)
        out.append(
            Block(
                offset_lo=offsets[0],
                offset_hi=offsets[-1],
                n_kmers=len(r),
                a_start=min(i for i, _ in r),
                a_end=max(i for i, _ in r) + k,
                b_start=min(j for _, j in r),
                b_end=max(j for _, j in r) + k,
                offset_median=offsets[len(offsets) // 2],
            )
        )
    return out


def reversed_pair(blocks: list[Block]) -> tuple[Block, Block] | None:
    """The strongest pair of blocks in reverse order, or None.

    Two blocks qualify when their A ranges overlap by at most MAX_OVERLAP residues, their
    B ranges likewise, and the block that starts first along A starts second along B. Among qualifying pairs,
    the one whose weaker block has the most k-mers wins (ties: more k-mers in total, then
    the earlier A start). Block 1 of the returned pair is the one that comes first along A.
    """
    best = None
    best_key = None
    for x in range(len(blocks)):
        for y in range(x + 1, len(blocks)):
            p, q = blocks[x], blocks[y]
            if p.a_start > q.a_start:
                p, q = q, p
            if p.a_end - q.a_start > MAX_OVERLAP:  # overlap in A
                continue
            if (
                q.b_start >= p.b_start or q.b_end - p.b_start > MAX_OVERLAP
            ):  # q before p along B
                continue
            key = (min(p.n_kmers, q.n_kmers), p.n_kmers + q.n_kmers, -p.a_start)
            if best_key is None or key > best_key:
                best, best_key = (p, q), key
    return best


def gapless_identity(seq_a: str, seq_b: str, block: Block) -> tuple[int, int, int, int]:
    """Identical residues in the gapless comparison of a block's A range with B.

    B is read at the block's median offset, so residue A[i] is compared with
    B[i + offset_median]. Positions that fall outside B are left out.
    Returns (n_identical, n_compared, b_start, b_end) with the B range used.
    """
    off = block.offset_median
    a0 = max(block.a_start, -off)
    a1 = min(block.a_end, len(seq_b) - off)
    if a1 <= a0:
        return 0, 0, 0, 0
    same = sum(seq_a[i] == seq_b[i + off] for i in range(a0, a1))
    return same, a1 - a0, a0 + off, a1 + off


def offset_test(
    seq_a: str, seq_b: str, alphabet: str, k: int, n_blocks: list[int] = N_BLOCKS
) -> list[dict]:
    """Run the offset test for one pair at one alphabet and k, for every n_block.

    One row per n_block. Every quantity is its own column; coordinates are 0-based,
    end-exclusive. Block 1 is the block that comes first along A.
    """
    hits, n_dropped = shared_kmer_hits(seq_a, seq_b, alphabet, k)
    rows = []
    for n_block in n_blocks:
        blocks = find_blocks(hits, k, n_block)
        pair = reversed_pair(blocks)
        row = {
            "alphabet": alphabet,
            "ksize": k,
            "n_block": n_block,
            "n_shared_kmer_hits": len(hits),
            "n_kmers_dropped_low_complexity": n_dropped,
            "n_blocks": len(blocks),
            "called": pair is not None,
        }
        for tag, blk in zip(["block1", "block2"], pair or (None, None)):
            if blk is None:
                for col in [
                    "a_start",
                    "a_end",
                    "b_start",
                    "b_end",
                    "n_kmers",
                    "offset_median",
                    "identical",
                    "compared",
                ]:
                    row[f"{tag}_{col}"] = None
                continue
            same, n, _, _ = gapless_identity(seq_a, seq_b, blk)
            row.update(
                {
                    f"{tag}_a_start": blk.a_start,
                    f"{tag}_a_end": blk.a_end,
                    f"{tag}_b_start": blk.b_start,
                    f"{tag}_b_end": blk.b_end,
                    f"{tag}_n_kmers": blk.n_kmers,
                    f"{tag}_offset_median": blk.offset_median,
                    f"{tag}_identical": same,
                    f"{tag}_compared": n,
                }
            )
        rows.append(row)
    return rows


def read_fasta(path) -> dict[str, str]:
    """{accession: sequence}; UniProt headers sp|ACC|NAME give ACC, others the first word."""
    import gzip

    seqs: dict[str, list[str]] = {}
    name = None
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                word = line[1:].split()[0]
                name = word.split("|")[1] if "|" in word else word
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def all_settings() -> list[tuple[str, int]]:
    """(alphabet, k) for every alphabet and k in the brief; n_block is applied inside."""
    return [(a, k) for a, ks in KSIZES.items() for k in ks]


def shuffled(seq: str, seed: int) -> str:
    """The residues of `seq` in a random order (composition kept), fixed seed."""
    residues = list(seq)
    random.Random(seed).shuffle(residues)
    return "".join(residues)


def match_line(a: str, b: str) -> str:
    """'|' where two equal-length strings agree, ' ' where they differ."""
    return "".join("|" if x == y else " " for x, y in zip(a, b))


def format_block_comparison(
    name_a: str,
    seq_a: str,
    name_b: str,
    seq_b: str,
    block: Block | dict,
    width: int = 60,
) -> str:
    """Gapless comparison of one block, with 1-based coordinates and 'x of y identical'.

    `block` is a Block or a dict with a_start, a_end and offset_median (0-based,
    end-exclusive), as stored in the result tables.
    """
    if isinstance(block, dict):
        block = Block(
            0, 0, 0, block["a_start"], block["a_end"], 0, 0, block["offset_median"]
        )
    same, n, b0, b1 = gapless_identity(seq_a, seq_b, block)
    off = block.offset_median
    a0 = b0 - off
    lines = []
    pad = max(len(name_a), len(name_b))
    for s in range(0, n, width):
        sa = seq_a[a0 + s : a0 + min(s + width, n)]
        sb = seq_b[b0 + s : b0 + min(s + width, n)]
        lines.append(f"{name_a:>{pad}} {a0 + s + 1:>5} {sa} {a0 + s + len(sa)}")
        lines.append(f"{'':>{pad}} {'':>5} {match_line(sa, sb)}")
        lines.append(f"{name_b:>{pad}} {b0 + s + 1:>5} {sb} {b0 + s + len(sb)}")
        lines.append("")
    lines.append(f"{same} of {n} identical")
    return "\n".join(lines)


def format_sequence_with_blocks(
    name: str, seq: str, ranges: list[tuple[int, int, str]], width: int = 60
) -> str:
    """A protein in lines of `width` residues, 1-based coordinates, blocks marked below.

    `ranges` holds (start, end, symbol), 0-based end-exclusive; residues inside a range get
    `symbol` on the marker line, others a space.
    """
    marks = [" "] * len(seq)
    for start, end, symbol in ranges:
        for i in range(max(0, start), min(len(seq), end)):
            marks[i] = symbol
    lines = [f"{name} ({len(seq)} aa)"]
    for s in range(0, len(seq), width):
        chunk = seq[s : s + width]
        lines.append(f"{s + 1:>5} {chunk} {s + len(chunk)}")
        lines.append(f"{'':>5} {''.join(marks[s : s + width])}")
    return "\n".join(lines)


def offset_histogram(hits: list[tuple[int, int]]) -> dict[int, int]:
    """Shared k-mer occurrences per offset bin, keyed by the bin's lowest offset (residues)."""
    counts: dict[int, int] = defaultdict(int)
    for i, j in hits:
        counts[((j - i) // BIN_WIDTH) * BIN_WIDTH] += 1
    return dict(sorted(counts.items()))


def format_local_alignment(
    name_a: str,
    name_b: str,
    qaln: str,
    taln: str,
    qstart: int,
    tstart: int,
    width: int = 60,
) -> str:
    """An MMseqs2 alignment as MMseqs2 reported it (its qaln and taln strings, 1-based
    starts), with a match line and 'x of y aligned columns identical'."""
    same = sum(x == y and x != "-" for x, y in zip(qaln, taln))
    lines = []
    ia, ib = qstart, tstart
    pad = max(len(name_a), len(name_b))
    for s in range(0, len(qaln), width):
        ta, tb = qaln[s : s + width], taln[s : s + width]
        na, nb = sum(c != "-" for c in ta), sum(c != "-" for c in tb)
        lines.append(f"{name_a:>{pad}} {ia:>5} {ta} {ia + na - 1}")
        lines.append(f"{'':>{pad}} {'':>5} {match_line(ta, tb)}")
        lines.append(f"{name_b:>{pad}} {ib:>5} {tb} {ib + nb - 1}")
        lines.append("")
        ia, ib = ia + na, ib + nb
    lines.append(f"{same} of {len(qaln)} aligned columns identical")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Comparators and metrics (polars imported here so the test itself needs only the stdlib).

IDENTITY_BIN_EDGES = [0.3, 0.5, 0.9]
IDENTITY_BIN_LABELS = ["under 30%", "30 to 50%", "50 to 90%", "over 90%"]


def mmseqs_calls(mm):
    """One row per pair and copy: what MMseqs2 C1 and C2 report.

    C1 (one alignment): coverage of the shorter protein by the best alignment, its
    identity; it never calls a rearrangement.
    C2 (two alignments, --alt-ali 2): called when the two best alignments overlap by at
    most MAX_OVERLAP residues in A and in B, and come in opposite orders along A and B.
    c2_called_no_overlap is the same with no overlap allowed.
    Coordinates are MMseqs2's, 1-based and inclusive.
    """
    import polars as pl

    keys = ["pair_set", "accession_a", "accession_b", "copy"]
    c1 = (
        mm.filter(pl.col("method") == "C1")
        .select(
            *keys,
            "fident",
            (
                (pl.col("qend") - pl.col("qstart") + 1)
                / pl.min_horizontal("qlen", "tlen")
            )
            .clip(upper_bound=1.0)
            .alias("c1_coverage_of_shorter"),
            pl.col("fident").alias("c1_identity"),
        )
        .drop("fident")
    )
    c2 = (
        mm.filter((pl.col("method") == "C2") & (pl.col("alignment_rank") <= 1))
        .sort(*keys, "alignment_rank")
        .group_by(keys, maintain_order=True)
        .agg(
            pl.len().alias("c2_n_alignments"),
            pl.col("qstart"),
            pl.col("qend"),
            pl.col("tstart"),
            pl.col("tend"),
        )
    )

    def reversed_alignments(r, allowed: int) -> bool:
        if r["c2_n_alignments"] < 2:
            return False
        (q1s, q2s), (q1e, q2e) = r["qstart"], r["qend"]
        (t1s, t2s), (t1e, t2e) = r["tstart"], r["tend"]
        if q1s > q2s:  # make alignment 1 the one that starts first along A
            q1s, q2s, q1e, q2e, t1s, t2s, t1e, t2e = (
                q2s,
                q1s,
                q2e,
                q1e,
                t2s,
                t1s,
                t2e,
                t1e,
            )
        # 1-based inclusive: residues shared = end - start + 1
        if q1e - q2s + 1 > allowed:
            return False
        return t2s < t1s and t2e - t1s + 1 <= allowed

    def first(col, i):
        return pl.col(col).list.get(i, null_on_oob=True)

    c2 = c2.with_columns(
        # The two alignments as 0-based end-exclusive ranges, alignment 1 = the best.
        *[
            (first(c, i) - (1 if c.endswith("start") else 0)).alias(
                f"c2_aln{i + 1}_{side}_{c[1:]}"
            )
            for i in range(2)
            for c, side in [
                ("qstart", "a"),
                ("qend", "a"),
                ("tstart", "b"),
                ("tend", "b"),
            ]
        ],
    )
    c2 = c2.with_columns(
        pl.struct(c2.columns)
        .map_elements(lambda r: reversed_alignments(r, MAX_OVERLAP), pl.Boolean)
        .alias("c2_called"),
        pl.struct(c2.columns)
        .map_elements(lambda r: reversed_alignments(r, 0), pl.Boolean)
        .alias("c2_called_no_overlap"),
    ).select(
        *keys,
        "c2_n_alignments",
        "c2_called",
        "c2_called_no_overlap",
        *[c for c in c2.columns if c.startswith("c2_aln")],
    )
    # Every pair and copy appears in mm (a left join from the pair list); pairs with no
    # alignment get C1 coverage 0 and no C2 call.
    base = mm.select(keys).unique(maintain_order=True)
    out = base.join(c1, on=keys, how="left").join(c2, on=keys, how="left")
    return out.with_columns(
        pl.col("c1_coverage_of_shorter").fill_null(0.0),
        pl.col("c2_n_alignments").fill_null(0),
        pl.col("c2_called").fill_null(False),
        pl.col("c2_called_no_overlap").fill_null(False),
    )


def identity_bin(identity):
    """Polars expression: the brief's identity bins from a 0-1 identity column."""
    import polars as pl

    return (
        pl.when(identity.is_null())
        .then(pl.lit("no alignment"))
        .when(identity < IDENTITY_BIN_EDGES[0])
        .then(pl.lit(IDENTITY_BIN_LABELS[0]))
        .when(identity < IDENTITY_BIN_EDGES[1])
        .then(pl.lit(IDENTITY_BIN_LABELS[1]))
        .when(identity < IDENTITY_BIN_EDGES[2])
        .then(pl.lit(IDENTITY_BIN_LABELS[2]))
        .otherwise(pl.lit(IDENTITY_BIN_LABELS[3]))
    )


def rate_with_ci(
    values, n_boot: int = 2000, seed: int = 293
) -> tuple[float, float, float, int]:
    """Fraction of True, with a bootstrap 95% CI over pairs (percentile), and n."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        int(x.size),
    )


def inside_fraction(start, end, lo, hi):
    """Polars expression: share of the range [start, end) that lies inside [lo, hi)."""
    import polars as pl

    inter = (pl.min_horizontal(end, hi) - pl.max_horizontal(start, lo)).clip(
        lower_bound=0
    )
    return inter / (end - start)


def called_in_chain_region(called, ranges_a, ranges_b, min_inside: float = 0.5):
    """Polars expression for T1: the call is made, and every range of it lies at least
    `min_inside` inside the UniProt range its PDB chain covers.

    ranges_a / ranges_b are lists of (start_col, end_col) names on A and on B, 0-based
    end-exclusive. The chain ranges come from SIFTS (min_uniprot_position_a ..
    max_uniprot_position_a, 1-based inclusive, converted here).
    """
    import polars as pl

    lo_a, hi_a = pl.col("min_uniprot_position_a") - 1, pl.col("max_uniprot_position_a")
    lo_b, hi_b = pl.col("min_uniprot_position_b") - 1, pl.col("max_uniprot_position_b")
    conds = [called]
    conds += [
        inside_fraction(pl.col(s), pl.col(e), lo_a, hi_a) >= min_inside
        for s, e in ranges_a
    ]
    conds += [
        inside_fraction(pl.col(s), pl.col(e), lo_b, hi_b) >= min_inside
        for s, e in ranges_b
    ]
    out = conds[0]
    for c in conds[1:]:
        out = out & c.fill_null(False)
    # pl.min_horizontal / max_horizontal skip nulls, so a row with no chain range would
    # pass every inside test; such rows (every set but T1) get null instead.
    return pl.when(pl.col("min_uniprot_position_a").is_null()).then(None).otherwise(out)


def same_family_place(arch: dict, acc_a: str, acc_b: str, ranges: list) -> bool | None:
    """For a T2 pair: does every range pair (a_start, a_end, b_start, b_end; 0-based,
    end-exclusive) lie at least half on a Pfam domain of one family in A and on a domain
    of the same family in B? `arch` maps accession -> list of (family, start, end) with
    Pfam's 1-based inclusive envelope. None when a range is missing."""

    def families(acc, start, end):
        out = set()
        for fam, s, e in arch.get(acc, []):
            inter = min(end, e) - max(start, s - 1)
            if inter >= 0.5 * (end - start):
                out.add(fam)
        return out

    for a0, a1, b0, b1 in ranges:
        if a0 is None or b0 is None:
            return None
        if not (families(acc_a, a0, a1) & families(acc_b, b0, b1)):
            return False
    return True


def load_results(data_dir):
    """Offset-test rows and MMseqs2 calls for T1, T2, N1, N2, with identity bins and the T1
    chain-region check. Returns (offset, mmseqs): offset has one row per pair x copy x
    setting; mmseqs one row per pair x copy. identity = fident of the best MMseqs2
    alignment of the real pair (C1), null when MMseqs2 reports none at E <= 1e-3."""
    from pathlib import Path

    import polars as pl

    d = Path(data_dir)
    sets = ["t1", "t2", "n1", "n2"]
    offset = pl.concat(
        [pl.read_parquet(d / f"offset_test_{s}.parquet") for s in sets],
        how="diagonal_relaxed",
    )
    mm = pl.concat(
        [
            ku_mm
            for s in sets
            for ku_mm in [mmseqs_calls(pl.read_parquet(d / f"mmseqs_{s}.parquet"))]
        ],
        how="diagonal_relaxed",
    )
    keys = ["pair_set", "accession_a", "accession_b"]
    ident = mm.filter(pl.col("copy") == -1).select(
        *keys, pl.col("c1_identity").alias("identity")
    )
    # N1 was matched to T2 on the identities of the N1 pool run. An MMseqs2 E-value
    # depends on how many sequences are in the run, so a pair near E = 1e-3 can have an
    # alignment in one run and none in another; N1's identity is read from the run it was
    # selected on, so its bins match T2's as built.
    pool = (
        mmseqs_calls(pl.read_parquet(d / "mmseqs_n1_pool.parquet"))
        .filter(pl.col("copy") == -1)
        .select(
            "accession_a", "accession_b", pl.col("c1_identity").alias("pool_identity")
        )
    )
    ident = (
        ident.join(pool, on=["accession_a", "accession_b"], how="left")
        .with_columns(
            pl.when(pl.col("pair_set") == "N1")
            .then(pl.col("pool_identity"))
            .otherwise(pl.col("identity"))
            .alias("identity")
        )
        .drop("pool_identity")
        .with_columns(identity_bin(pl.col("identity")).alias("identity_bin"))
    )
    chains = (
        pl.read_parquet(d / "t1_pairs.parquet")
        .select(
            "accession_a",
            "accession_b",
            "chain_a",
            "chain_b",
            "tm_align",
            "tm_align_cp",
            "min_uniprot_position_a",
            "max_uniprot_position_a",
            "min_uniprot_position_b",
            "max_uniprot_position_b",
        )
        .with_columns(pl.lit("T1").alias("pair_set"))
    )
    offset = offset.join(ident, on=keys, how="left").join(chains, on=keys, how="left")
    offset = offset.with_columns(
        called_in_chain_region(
            pl.col("called"),
            [("block1_a_start", "block1_a_end"), ("block2_a_start", "block2_a_end")],
            [("block1_b_start", "block1_b_end"), ("block2_b_start", "block2_b_end")],
        ).alias("called_in_chain_region")
    )
    arch_df = pl.read_parquet(d / "swissprot_architectures.parquet")
    arch = {
        acc: list(zip(f, st, en))
        for acc, f, st, en in arch_df.select(
            "accession", "families_in_order", "domain_starts", "domain_ends"
        ).iter_rows()
    }

    def t2_place(df, called, cols):
        vals = []
        for r in df.select(
            "pair_set", "accession_a", "accession_b", called, *cols
        ).iter_rows(named=True):
            if r["pair_set"] != "T2" or not r[called]:
                vals.append(None if r["pair_set"] != "T2" else False)
                continue
            vals.append(
                same_family_place(
                    arch,
                    r["accession_a"],
                    r["accession_b"],
                    [tuple(r[c] for c in cols[i : i + 4]) for i in (0, 4)],
                )
            )
        return pl.Series(vals, dtype=pl.Boolean)

    block_cols = [
        "block1_a_start",
        "block1_a_end",
        "block1_b_start",
        "block1_b_end",
        "block2_a_start",
        "block2_a_end",
        "block2_b_start",
        "block2_b_end",
    ]
    offset = offset.with_columns(
        t2_place(offset, "called", block_cols).alias("called_on_same_family")
    )
    mm = mm.join(ident, on=keys, how="left").join(chains, on=keys, how="left")
    aln_cols = [
        "c2_aln1_a_start",
        "c2_aln1_a_end",
        "c2_aln1_b_start",
        "c2_aln1_b_end",
        "c2_aln2_a_start",
        "c2_aln2_a_end",
        "c2_aln2_b_start",
        "c2_aln2_b_end",
    ]
    mm = mm.with_columns(
        t2_place(mm, "c2_called", aln_cols).alias("c2_called_on_same_family")
    )
    mm = mm.with_columns(
        called_in_chain_region(
            pl.col("c2_called"),
            [
                ("c2_aln1_a_start", "c2_aln1_a_end"),
                ("c2_aln2_a_start", "c2_aln2_a_end"),
            ],
            [
                ("c2_aln1_b_start", "c2_aln1_b_end"),
                ("c2_aln2_b_start", "c2_aln2_b_end"),
            ],
        ).alias("c2_called_in_chain_region")
    )
    return offset, mm


def settings_table(offset, mm, n_boot: int = 2000):
    """Step 5 metrics, one row per method: every k-mer setting, then C2 and C1.

    Columns: sensitivity on T1 (any call, and a call inside the UniProt range the two PDB
    chains cover) and on T2 (any call, and a call whose two blocks or alignments each sit
    on the same Pfam family in A and B), false call rate on N1 and N2, and the call rate on the shuffled copies of each set,
    each with a bootstrap 95% CI over pairs (shuffled copies: over pair x copy), and n.
    """
    import polars as pl

    def block(df_real, df_shuf, called_col, in_place_col, family_col):
        out = {}
        for name, sel, col in [
            ("t1_sensitivity", df_real.filter(pl.col("pair_set") == "T1"), called_col),
            (
                "t1_sensitivity_in_chain_region",
                df_real.filter(pl.col("pair_set") == "T1"),
                in_place_col,
            ),
            ("t2_sensitivity", df_real.filter(pl.col("pair_set") == "T2"), called_col),
            (
                "t2_sensitivity_on_same_family",
                df_real.filter(pl.col("pair_set") == "T2"),
                family_col,
            ),
            ("n1_false_rate", df_real.filter(pl.col("pair_set") == "N1"), called_col),
            ("n2_false_rate", df_real.filter(pl.col("pair_set") == "N2"), called_col),
            (
                "shuffled_rate_t1",
                df_shuf.filter(pl.col("pair_set") == "T1"),
                called_col,
            ),
            (
                "shuffled_rate_t2",
                df_shuf.filter(pl.col("pair_set") == "T2"),
                called_col,
            ),
            (
                "shuffled_rate_n1",
                df_shuf.filter(pl.col("pair_set") == "N1"),
                called_col,
            ),
            ("shuffled_rate_all", df_shuf, called_col),
        ]:
            m, lo, hi, n = rate_with_ci(sel[col].to_list(), n_boot)
            out.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_n": n})
        return out

    rows = []
    for (alphabet, k, nb), g in offset.group_by(
        ["alphabet", "ksize", "n_block"], maintain_order=True
    ):
        rows.append(
            {
                "method": f"k-mer offset test, {alphabet} k={k} n_block={nb}",
                "alphabet": alphabet,
                "ksize": k,
                "n_block": nb,
                **block(
                    g.filter(pl.col("copy") == -1),
                    g.filter(pl.col("copy") >= 0),
                    "called",
                    "called_in_chain_region",
                    "called_on_same_family",
                ),
            }
        )
    real, shuf = mm.filter(pl.col("copy") == -1), mm.filter(pl.col("copy") >= 0)
    rows.append(
        {
            "method": "C2, MMseqs2 two alignments",
            **block(
                real,
                shuf,
                "c2_called",
                "c2_called_in_chain_region",
                "c2_called_on_same_family",
            ),
        }
    )
    c1 = real.with_columns(pl.lit(False).alias("c1_called"))
    rows.append(
        {
            "method": "C1, MMseqs2 one alignment",
            **block(
                c1,
                shuf.with_columns(pl.lit(False).alias("c1_called")),
                "c1_called",
                "c1_called",
                "c1_called",
            ),
        }
    )
    return pl.DataFrame(rows, infer_schema_length=None)
