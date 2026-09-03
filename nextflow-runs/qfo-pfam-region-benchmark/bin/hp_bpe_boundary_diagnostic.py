#!/usr/bin/env python3
"""Do ProtBERTa_2's learned HP token boundaries land on Pfam domain boundaries?

The decision this answers, before any compute is spent: whether the Rannon & Burstein
result belongs in this paper as an independent low-k measurement of the same gradient, or
as an unrelated pLM result that happens to use two letters.

They trained protein language models on reduced alphabets and found the 2-letter model
worst on signal peptides (ROC-AUC 0.75, PR-AUC 0.47), nearly lossless on solubility
(relative F1 ~0.97), and strong on enzyme detection (~0.90). Signal peptides are ~20
residues; solubility and enzyme class are whole-protein properties. That is a
feature-LENGTH gradient, and their BPE tokens are short. Our HP k floor is 18. If the two
are the same gradient, their negative result is the low-k arm of our own k-sweep, run
independently by another lab -- which is worth far more to the claim than another sweep of
our own.

This measures ONE thing on the way to that: whether their learned segmentation agrees with
domain architecture at all. Their tokenizer is trained on a corpus, with no annotation in
the loss, so any agreement is emergent. Boundary-position enrichment is computed against a
length- and composition-matched shuffled null rather than against a uniform expectation:
HP strings are strongly autocorrelated, so a naive "how many positions are boundaries"
baseline overstates enrichment for every alphabet by the same unknown amount.

Their split, read out of burstein-lab/BioTokenizers data_processing/get_encoded_dataset.py
(HYDROPHILIC_PHOBIC), is
    L = hydrophilic  S T N K E Q H D R  (plus the ambiguity codes Z and B)
    B = hydrophobic  A G I L M V P F W C Y  (plus J)
Over the 20 canonical residues that is not merely closest to hp_lehninger_c_nonpolar2 -- it
is IDENTICAL to it, and the two produce identical numbers below, which is a useful check
that this script does what it claims. The only difference is the ambiguity codes: they map
B, Z and J, which none of our alphabets define. Against the rest of ours it disagrees on
one to four residues -- C for hp_lehninger2, G for hp_pbotc_1st_ed2, G and P for
hp_thomas_dill2 -- and the full per-residue disagreement is printed at run time rather than
asserted here, so this paragraph cannot drift from the tables.

WHAT THIS DOES NOT MEASURE. Segmentation agreement, not end-to-end performance. A tokenizer
whose boundaries never coincide with domain boundaries can still support a model that finds
domains, and one whose boundaries agree perfectly can still be beaten by a k-mer method.
The claim available from this number is about where information sits in an HP string, not
about which method wins.

Runtime is time-boxed by --max-proteins: ~30 s of CPU at the default 2000, across all eight
alphabets and their nulls. Everything is pure Python and stdlib except polars for the
annotation table, so it runs anywhere the rest of bin/ runs -- no `tokenizers` dependency
and therefore no new container for a script that otherwise needs nothing. The BPE itself is
hand-written; --self-test checks it against a textbook reference implementation.
"""

import argparse
import json
import random
import sys
import tarfile
import tempfile
import time
from heapq import heappop, heappush
from pathlib import Path

import polars as pl

# Verbatim from kmerseek src/rust/hp_alphabets.rs (the h_residues argument of build_hp);
# everything not listed is polar, and build_hp() there asserts the two sets cover all 20.
# Names are the post-PR-#43 spellings that appear in variant strings and in ALL_ENCODINGS.
# hp_lehninger_hpc3 is deliberately absent: it has a third class for cysteine, and their
# vocabulary has two symbols, so there is nothing to map it onto without inventing one.
HP_ALPHABETS = {
    "hp_lehninger2":            "AFGILMPVWY",
    "hp_thomas_dill2":          "ACFILMVWY",
    "hp_kyte_doolittle2":       "ACFILMV",
    "hp_thomas_dill_no_c2":     "AFILMVWY",
    "hp_lehninger_c_nonpolar2": "ACFGILMPVWY",
    "hp_pbotc_1st_ed2":         "ACFILMPVWY",
    # The negative control the sweep already carries: a random 10/10 split with the same
    # class balance. If enrichment does not drop here, the measurement is not measuring
    # hydrophobicity.
    "hp_random_control2":       "ADGKLMQRWY",
}

# From burstein-lab/BioTokenizers, data_processing/get_encoded_dataset.py.
PROTBERTA_HYDROPHOBIC = "AGILMVPFWCY"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Their symbols. The merge table is written in these two characters, so every alphabet has
# to be re-expressed in them before the same merges can be applied.
HYDROPHOBIC_SYMBOL = "B"
HYDROPHILIC_SYMBOL = "L"
# Anything outside the 20 canonical residues (X, U, O, and the ambiguity codes). Their
# pipeline leaves these untranslated and the byte-level vocabulary carries them as
# single-character tokens, so no merge ever fires on one. Same behaviour here.
UNKNOWN_SYMBOL = "X"

TOKENIZER_SUBDIR = "BPE_tokenizer_prot_5000_min_freq_2_mapping_2"


def resolve_tokenizer(path: Path) -> Path:
    """Accept the Zenodo tarball, the directory it unpacks to, or the tokenizer itself."""
    if path.is_file() and "".join(path.suffixes[-2:]) in (".tar.gz", ".tgz"):
        tmp = Path(tempfile.mkdtemp(prefix="protberta_tok_"))
        with tarfile.open(path) as tar:
            # filter="data" is the 3.12+ default-in-3.14 behaviour: refuse absolute paths,
            # links escaping the destination and odd modes. Passed explicitly so this does
            # not warn on 3.13 and does not change behaviour on 3.14, and guarded so it
            # still runs on the older interpreters some of bin/ is invoked under.
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)
        path = tmp
    if (path / "merges.txt").exists():
        return path
    hits = sorted(path.rglob(f"{TOKENIZER_SUBDIR}/merges.txt"))
    if not hits:
        raise SystemExit(
            f"no {TOKENIZER_SUBDIR}/merges.txt under {path}. Fetch the tokenizers from "
            "Zenodo doi 10.5281/zenodo.18256943 (make bpe-tokenizer)."
        )
    return hits[0].parent


def load_merges(tokenizer_dir: Path) -> dict[tuple[str, str], int]:
    """The BPE merge table, lowest rank first. Rank IS the merge order."""
    merges = {}
    for line in (tokenizer_dir / "merges.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        left, right = line.split(" ")
        merges.setdefault((left, right), len(merges))
    return merges


def encode(seq: str, hydrophobic: frozenset) -> str:
    return "".join(
        HYDROPHOBIC_SYMBOL if c in hydrophobic
        else HYDROPHILIC_SYMBOL if c in AMINO_ACIDS
        else UNKNOWN_SYMBOL
        for c in seq
    )


def bpe_cuts(encoded: str, merges: dict[tuple[str, str], int]) -> list[int]:
    """Token boundaries, as residue offsets, from applying the merge table to one sequence.

    Returns the INTERIOR cuts only: offset c means the boundary between residue c-1 and
    residue c, 0-based, so the values run 1..len-1. The two protein termini are dropped
    because every tokenizer gets them right for free and including them would dilute the
    statistic identically for every alphabet.

    A heap over pending merges rather than a pass per merge. The whole-table sweep is
    O(n_merges * len) = 4740 * ~500 per protein, which at four alphabets times four
    tokenizations of ~2000 proteins is not a time-boxed diagnostic. This is O(len log len)
    and produces the same segmentation: HuggingFace applies the globally lowest-rank pair
    first and resolves overlapping occurrences of it left to right, which is exactly the
    (rank, position) ordering of this heap.

    The tokenizer was trained by ByteLevelBPETokenizer on ungapped sequence, so its
    pre-tokenizer sees one uninterrupted run of letters and the merges apply across the
    whole protein. No word splitting is reproduced here because there is none to reproduce.
    """
    n = len(encoded)
    if n < 2:
        return []
    sym = list(encoded)
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    prev = list(range(-1, n - 1))
    alive = [True] * n

    heap: list[tuple[int, int, int]] = []
    for i in range(n - 1):
        rank = merges.get((sym[i], sym[i + 1]))
        if rank is not None:
            heappush(heap, (rank, i, i + 1))

    while heap:
        rank, i, j = heappop(heap)
        # Stale entry: either side already merged away, or they are no longer neighbours.
        if not alive[i] or not alive[j] or nxt[i] != j:
            continue
        if merges.get((sym[i], sym[j])) != rank:
            continue
        sym[i] += sym[j]
        alive[j] = False
        right = nxt[j]
        nxt[i] = right
        if right != -1:
            prev[right] = i
        for a, b in ((prev[i], i), (i, right)):
            if a == -1 or b == -1:
                continue
            new_rank = merges.get((sym[a], sym[b]))
            if new_rank is not None:
                heappush(heap, (new_rank, a, b))

    cuts, pos, i = [], 0, 0
    while nxt[i] != -1:
        pos += len(sym[i])
        cuts.append(pos)
        i = nxt[i]
    return cuts


def _reference_cuts(encoded: str, merges: dict[tuple[str, str], int]) -> list[int]:
    """The textbook BPE loop, kept only so bpe_cuts can be checked against it.

    Repeatedly find the globally lowest-rank adjacent pair and merge every non-overlapping
    occurrence of it, left to right. O(len * n_merges), which is why it is not what runs --
    but it is the definition, and the heap version is an optimisation of it rather than a
    different algorithm. --self-test compares the two.
    """
    parts = list(encoded)
    while len(parts) > 1:
        best, rank = None, None
        for a, b in zip(parts, parts[1:]):
            r = merges.get((a, b))
            if r is not None and (rank is None or r < rank):
                best, rank = (a, b), r
        if best is None:
            break
        out, i = [], 0
        while i < len(parts):
            if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                out.append(parts[i] + parts[i + 1])
                i += 2
            else:
                out.append(parts[i])
                i += 1
        parts = out
    cuts, pos = [], 0
    for token in parts[:-1]:
        pos += len(token)
        cuts.append(pos)
    return cuts


def self_test(merges: dict[tuple[str, str], int], seed: int = 7) -> int:
    """Check the heap tokenizer against the reference on edge cases and random strings.

    Worth having as a flag rather than a comment: every number this script prints rests on
    the segmentation being the one the released tokenizer produces, the merge table is
    4739 rules long, and the failure mode of a subtly wrong merge order is plausible
    numbers rather than an error. Homopolymer runs are in the fixed cases on purpose --
    overlapping occurrences of one pair, e.g. (B,B) in BBB, are where a heap and a
    left-to-right sweep can disagree.
    """
    rng = random.Random(seed)
    cases = ["", "B", "BL", "BBB", "LLLL", "BLBLBL", "B" * 40, "L" * 40, "BBLBBLLBX" * 3]
    cases += ["".join(rng.choice("BL") for _ in range(rng.randint(1, 350)))
              for _ in range(400)]
    cases += ["".join(rng.choice("BLX") for _ in range(rng.randint(1, 120)))
              for _ in range(100)]
    bad = [c for c in cases if bpe_cuts(c, merges) != _reference_cuts(c, merges)]
    print(f"self-test: {len(cases)} sequences, {len(bad)} disagree with the reference")
    for c in bad[:3]:
        print(f"  MISMATCH {c[:60]!r}")
    return 1 if bad else 0


def domain_boundaries(rows, length: int) -> tuple[list[int], list[int]]:
    """Interior N- and C-terminal cut positions implied by a protein's Pfam domains.

    Pfam coordinates are 1-based inclusive. The cut before a domain starting at s sits at
    offset s-1; the cut after a domain ending at e sits at offset e. Boundaries at the
    protein termini are dropped for the same reason bpe_cuts drops them.
    """
    starts = [s - 1 for s, _ in rows if 0 < s - 1 < length]
    ends = [e for _, e in rows if 0 < e < length]
    return starts, ends


def hit_count(boundaries: list[int], cuts: set[int], tolerance: int) -> int:
    if not boundaries:
        return 0
    if tolerance == 0:
        return sum(1 for b in boundaries if b in cuts)
    return sum(
        1 for b in boundaries
        if any((b + d) in cuts for d in range(-tolerance, tolerance + 1))
    )


def read_fasta(path: Path) -> dict[str, str]:
    """UniProt FASTA. Same accession extraction as bin/zinc_finger_hp_diagnostic.py."""
    recs, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    recs[name] = "".join(buf)
                parts = line[1:].split("|")
                name = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        recs[name] = "".join(buf)
    return recs


def measure(seqs, domains, hydrophobic, merges, tolerance):
    """One pass over the corpus for one alphabet: token stats and boundary hits."""
    n_tokens = n_residues = n_cuts = 0
    hits = {"start": 0, "end": 0}
    totals = {"start": 0, "end": 0}
    for acc, seq in seqs.items():
        cuts = bpe_cuts(encode(seq, hydrophobic), merges)
        cutset = set(cuts)
        n_tokens += len(cuts) + 1
        n_residues += len(seq)
        n_cuts += len(cuts)
        starts, ends = domain_boundaries(domains[acc], len(seq))
        for kind, bnds in (("start", starts), ("end", ends)):
            totals[kind] += len(bnds)
            hits[kind] += hit_count(bnds, cutset, tolerance)
    n_bnd = totals["start"] + totals["end"]
    return {
        "n_tokens": n_tokens,
        "mean_token_length": n_residues / n_tokens if n_tokens else 0.0,
        # The naive expectation: what share of interior positions is a boundary at all.
        # Reported beside the shuffled null so the two can be compared, never instead of it.
        "boundary_density": n_cuts / max(1, n_residues - len(seqs)),
        "n_domain_boundaries": n_bnd,
        "hit_rate_start": hits["start"] / totals["start"] if totals["start"] else 0.0,
        "hit_rate_end": hits["end"] / totals["end"] if totals["end"] else 0.0,
        "hit_rate": (hits["start"] + hits["end"]) / n_bnd if n_bnd else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not required, because --self-test needs neither. Checked after parsing instead.
    p.add_argument("--fasta", type=Path)
    p.add_argument("--annotations", type=Path,
                   help="Pfam annotation dir holding human_pfam_domains.parquet")
    p.add_argument("--tokenizer", required=True, type=Path,
                   help="ProtBERTa_tokenizers.tar.gz, the directory it unpacks to, or the "
                        f"{TOKENIZER_SUBDIR} directory itself")
    p.add_argument("--max-proteins", type=int, default=2_000,
                   help="time box. 0 means the whole proteome, which is ~10x this and is "
                        "not what a go/no-go diagnostic is for")
    p.add_argument("--n-null", type=int, default=3,
                   help="shuffled replicates. 3 is enough to see whether enrichment is far "
                        "outside the null; it is not enough for a p-value and none is given")
    p.add_argument("--tolerance", type=int, default=0,
                   help="residues of slack allowed between a token boundary and a domain "
                        "boundary. 0 is exact coincidence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true",
                   help="check the tokenizer against the reference implementation and exit. "
                        "Needs --tokenizer; --fasta and --annotations are ignored.")
    p.add_argument("--out", type=Path)
    p.add_argument("--figure", type=Path,
                   help="write the enrichment bar chart here. Needs matplotlib, which the "
                        "pipeline container does not carry; the MultiQC report draws the "
                        "same numbers from --out")
    args = p.parse_args()

    started = time.time()
    merges = load_merges(resolve_tokenizer(args.tokenizer))

    if args.self_test:
        raise SystemExit(self_test(merges, args.seed))
    if not (args.fasta and args.annotations):
        raise SystemExit("--fasta and --annotations are required unless --self-test is set")

    dom = pl.read_parquet(args.annotations / "human_pfam_domains.parquet")
    if "has_position" in dom.columns:
        dom = dom.filter(pl.col("has_position"))
    seqs = read_fasta(args.fasta)

    # Only proteins that HAVE a domain boundary to be right about. A protein with no
    # annotation contributes token statistics and no signal, so including it would move
    # mean_token_length without moving the thing being measured.
    domains: dict[str, list[tuple[int, int]]] = {}
    for acc, s, e in dom.select("accession", "domain_start", "domain_end").iter_rows():
        if acc in seqs:
            domains.setdefault(acc, []).append((s, e))

    keep = sorted(domains)
    rng = random.Random(args.seed)
    if args.max_proteins and len(keep) > args.max_proteins:
        keep = sorted(rng.sample(keep, args.max_proteins))
    seqs = {a: seqs[a] for a in keep}
    n_dom = sum(len(domains[a]) for a in keep)
    print(f"{len(seqs)} human proteins, {n_dom} Pfam domain instances, "
          f"{len(merges)} BPE merges, tolerance {args.tolerance} residues")

    alphabets = {"protberta_2": PROTBERTA_HYDROPHOBIC, **HP_ALPHABETS}

    # Their split against each of ours, computed rather than asserted.
    ref = frozenset(PROTBERTA_HYDROPHOBIC)
    print("\nresidues classed differently from ProtBERTa_2 (over the canonical 20):")
    for name, h in HP_ALPHABETS.items():
        diff = sorted(ref ^ frozenset(h))
        print(f"  {name:26s} {''.join(diff) if diff else 'none -- identical'}")
    print(f"  ambiguity codes B, Z, J are mapped by ProtBERTa_2 and by none of ours")

    # One shuffle per replicate, shared across alphabets: the null has to differ from the
    # observation only in residue ORDER, and giving each alphabet its own shuffle would add
    # a second source of variation between the columns being compared.
    nulls = []
    for rep in range(args.n_null):
        shuffled = {}
        for acc, seq in seqs.items():
            chars = list(seq)
            rng.shuffle(chars)
            shuffled[acc] = "".join(chars)
        nulls.append(shuffled)

    results = {}
    for name, h_res in alphabets.items():
        h_set = frozenset(h_res)
        obs = measure(seqs, domains, h_set, merges, args.tolerance)
        null = [measure(sh, domains, h_set, merges, args.tolerance) for sh in nulls]

        def null_mean(key):
            return sum(n[key] for n in null) / len(null)

        def ratio(key):
            m = null_mean(key)
            return obs[key] / m if m else None

        rates = [n["hit_rate"] for n in null]
        results[name] = {
            **obs,
            "null_hit_rate_mean": null_mean("hit_rate"),
            "null_hit_rate_start_mean": null_mean("hit_rate_start"),
            "null_hit_rate_end_mean": null_mean("hit_rate_end"),
            # Range, not a standard deviation: three replicates do not estimate one, and
            # printing sd from n=3 would invite a t-test this design cannot support.
            "null_hit_rate_range": max(rates) - min(rates),
            "n_null_replicates": args.n_null,
            "enrichment": ratio("hit_rate"),
            # N and C termini kept apart. A k-mer method loses the first k-1 residues at the
            # N terminus for a structural reason, and boundary_metrics already reports the
            # two offsets separately for the same reason; averaging them here would throw
            # away the half of the signal that is interpretable.
            "enrichment_start": ratio("hit_rate_start"),
            "enrichment_end": ratio("hit_rate_end"),
            "identical_to_protberta_2": frozenset(h_res) == ref,
        }
        spread = results[name]["null_hit_rate_range"]
        null_m = results[name]["null_hit_rate_mean"]
        print(f"{name:26s} mean_token_len={obs['mean_token_length']:5.2f}  "
              f"hit={obs['hit_rate']:.4f}  null={null_m:.4f}"
              f"(+/-{spread / 2:.4f})  "
              f"enrichment={results[name]['enrichment']:.3f}")

    elapsed = time.time() - started
    payload = {
        "tokenizer": str(args.tokenizer),
        "tokenizer_doi": "10.5281/zenodo.18256943",
        "code": "https://github.com/burstein-lab/BioTokenizers",
        "paper_doi": "10.64898/2026.02.08.701987",
        "n_proteins": len(seqs),
        "n_domain_instances": n_dom,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "limit": ("segmentation agreement, not end-to-end performance: a tokenizer whose "
                  "boundaries never coincide with domain boundaries can still support a "
                  "model that finds domains"),
        "alphabets": results,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n{elapsed:.1f}s")
    best = max(results.items(), key=lambda kv: kv[1]["enrichment"] or 0.0)
    print(f"most boundary-aligned: {best[0]} ({best[1]['enrichment']:.3f}x the shuffled null)")
    ctrl = results.get("hp_random_control2", {}).get("enrichment")
    if ctrl is not None:
        print(f"random 10/10 control:  {ctrl:.3f}x -- anything not clearly above this is "
              "measuring autocorrelation, not hydrophobicity")
    print("\nlimit: this is segmentation agreement, not end-to-end performance.")

    if args.figure:
        write_figure(results, args.figure)


def write_figure(results: dict, path: Path) -> None:
    """Enrichment per alphabet, with the shuffled null drawn at 1.0."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skipped {path}", file=sys.stderr)
        return

    order = sorted(results, key=lambda k: results[k]["enrichment"] or 0.0)
    vals = [results[k]["enrichment"] or 0.0 for k in order]
    colors = ["#c44e52" if k == "protberta_2" else
              "#8c8c8c" if k == "hp_random_control2" else "#4c72b0" for k in order]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.6))
    ax.barh(order, vals, color=colors)
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.set_xlabel("domain-boundary hit rate / length- and composition-matched shuffled null")
    ax.set_title("ProtBERTa_2 BPE token boundaries vs Pfam domain boundaries")
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
