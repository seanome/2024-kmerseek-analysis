"""Join probe sequences to probe intensities and cut them into k-mers."""

import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.geo import Geo2KmersError, read_series_matrix, read_table

COMPLEMENT = str.maketrans("ACGT", "TGCA")
BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS = {a + b + c: AMINO[i] for i, (a, b, c) in enumerate((a, b, c) for a in BASES for b in BASES for c in BASES)}
# sourmash's two letters: h hydrophobic, p polar (read from sourmash 4.9.4's output, see README)
HP = {a: "h" for a in "AFGILMPVWY"} | {a: "p" for a in "CDEHKNQRST"}
# sourmash's six Dayhoff groups, also read from its output
DAYHOFF = {a: letter for letter, group in zip("abcdef", ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"]) for a in group}
REDUCED = {"hp": HP, "dayhoff": DAYHOFF}
LARGEST_K = {"protein": 14, "dayhoff": 24, "hp": 63}  # a k-mer must fit in 64 bits
DETECTION_COLUMNS = ["ABS_CALL", "DETECTION", "DETECTION CALL", "DETECTION_CALL"]
DETECTION_P_COLUMNS = [
    "DETECTION P-VALUE",
    "DETECTION_PVAL",
    "DETECTION PVAL",
    "DETECTION_P_VALUE",
    "DETECTION P VALUE",
]
BACKGROUND_PERCENTILE = 5
LOG_SCALE_LIMIT = 32
MIN_STRETCH_AA = 100
MIN_SHARE_IN_PROTEOME = 0.9
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path, whole_name=False):
    """Return {name: sequence}, upper case. A name is the first word of the header, or all of it if whole_name."""
    records, name = {}, None
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith(">"):
                name = line[1:].strip() if whole_name else line[1:].split()[0]
                if name in records:
                    raise Geo2KmersError(f"{path} has the name {name} twice")
                records[name] = []
            else:
                records[name].append(line.strip())
    return {k: "".join(v).upper() for k, v in records.items()}


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def dna_kmers(seq, k):
    """Every k-mer of seq in canonical form: the smaller of it and its reverse complement.

    Windows holding a letter other than A, C, G or T are skipped.
    """
    out = []
    for piece in re.findall(r"[ACGT]+", seq):
        for i in range(len(piece) - k + 1):
            kmer = piece[i : i + k]
            rc = reverse_complement(kmer)
            out.append(min(kmer, rc))
    return out


def translate_frames(seq):
    """Translate the six frames. A codon holding any letter other than A, C, G, T becomes X."""
    frames = []
    for strand in (seq, reverse_complement(seq)):
        for start in range(3):
            codons = (strand[i : i + 3] for i in range(start, len(strand) - 2, 3))
            frames.append("".join(CODONS.get(c, "X") for c in codons))
    return frames


def letters(alphabet):
    return len(AMINO_ACIDS) if alphabet == "protein" else len(set(REDUCED[alphabet].values()))


def letter_codes(alphabet):
    """Table from byte to the letter's number. A letter outside the 20 amino acids gets 255."""
    table = np.full(256, 255, dtype=np.uint8)
    for i, letter in enumerate(AMINO_ACIDS):
        table[ord(letter)] = (
            i if alphabet == "protein" else sorted(set(REDUCED[alphabet].values())).index(REDUCED[alphabet][letter])
        )
    return table


def kmer_numbers(text, k, alphabet, table):
    """Each k-mer of text as one number, leaving out windows that hold another letter or a # joint."""
    codes = table[np.frombuffer(text.encode(), dtype=np.uint8)]
    n = len(codes) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    bad = np.convolve((codes == 255).astype(np.int64), np.ones(k, dtype=np.int64), "valid") > 0
    base = np.uint64(letters(alphabet))
    values = np.zeros(n, dtype=np.uint64)
    digits = np.where(codes == 255, 0, codes).astype(np.uint64)
    for j in range(k):
        values = values * base + digits[j : j + n]
    return values[~bad]


class ProteomeFrames:
    """Pick each probe's reading frame by matching its k-mers to a proteome.

    A stretch is the part of a frame between two stop codons, or between a stop and an end of the frame.
    A stretch passes if it has at least MIN_STRETCH_AA amino acids, not counting X, and at least
    MIN_SHARE_IN_PROTEOME of its k-mers are in the proteome. The longest passing stretch is used, and the rest of the probe is not.
    """

    def __init__(self, paths, k, alphabet):
        if k > LARGEST_K[alphabet]:
            raise Geo2KmersError(
                f"--proteome works up to k = {LARGEST_K[alphabet]} for alphabet {alphabet}, not k = {k}"
            )
        self.k, self.alphabet = k, alphabet
        self.table = letter_codes(alphabet)
        parts = []
        for path in paths:
            proteins = sorted(set(read_fasta(path).values()))
            parts.append(np.unique(kmer_numbers("#".join(proteins), k, alphabet, self.table)))
        self.known = np.unique(np.concatenate(parts))
        if not len(self.known):
            raise Geo2KmersError(f"no k-mer of k = {k} in {', '.join(map(str, paths))}")
        self.picks = {}

    def share_in_proteome(self, stretch):
        """The share of a stretch's k-mers in the proteome. 0 if it has none."""
        values = kmer_numbers(stretch, self.k, self.alphabet, self.table)
        if not len(values):
            return 0.0
        where = np.searchsorted(self.known, values).clip(max=len(self.known) - 1)
        return float((self.known[where] == values).mean())

    def pick(self, sequence):
        """Return the longest passing stretch as (frame, start, stretch, share), or None.

        Frames number as in translate_frames and start counts amino acids. A tie goes to the higher
        share, then the lower frame, then the earlier stretch, so the answer does not depend on order.
        """
        if sequence not in self.picks:
            best = None
            for number, frame in enumerate(translate_frames(sequence)):
                start = 0
                for stretch in frame.split("*"):
                    if len(stretch) - stretch.count("X") >= MIN_STRETCH_AA:
                        share = self.share_in_proteome(stretch)
                        rank = (len(stretch), share, -number, -start)
                        if share >= MIN_SHARE_IN_PROTEOME and (best is None or rank > best[0]):
                            best = (rank, (number, start, stretch, share))
                    start += len(stretch) + 1
            self.picks[sequence] = best[1] if best else None
        return self.picks[sequence]


def longest_stretch(sequence):
    """Return the longest stretch of any frame as (frame, start, stretch, None), or None if there is none.

    A tie goes to the lower frame, then the earlier stretch, as in ProteomeFrames.pick.
    """
    best = None
    for number, frame in enumerate(translate_frames(sequence)):
        start = 0
        for stretch in frame.split("*"):
            if stretch and (best is None or len(stretch) > len(best[2])):
                best = (number, start, stretch, None)
            start += len(stretch) + 1
    return best


def stretch_kmers(stretch, k, alphabet):
    """K-mers of one stretch. No k-mer crosses an X."""
    out = []
    for piece in stretch.split("X"):
        text = "".join(REDUCED[alphabet][a] for a in piece) if alphabet in REDUCED else piece
        out.extend(text[i : i + k] for i in range(len(text) - k + 1))
    return out


def channel_column(columns, channel, override=None):
    """Find the column that holds a channel's signal, leaving out background columns."""
    if override:
        if override not in columns:
            raise Geo2KmersError(f"no column named {override}; columns are: {', '.join(columns)}")
        return override
    wanted = [
        c
        for c in columns
        if c.lower().replace(" ", "").replace("_", "").startswith(channel)
        and re.search(r"intens|sig|median|mean", c, re.IGNORECASE)
        and not re.search(r"back|bkd|bkg|bg|sd|std", c, re.IGNORECASE)
    ]
    if len(wanted) != 1:
        raise Geo2KmersError(
            f"found {len(wanted)} columns for {channel} in: {', '.join(columns)}. Name one with --column."
        )
    return wanted[0]


def to_float(text):
    """A number, or None if the cell is empty, NULL or not finite."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def gsm_table(folder, sample):
    """A sample's table in a folder, plain or gzipped."""
    plain = Path(folder) / f"{sample}.txt"
    gzipped = Path(folder) / f"{sample}.txt.gz"
    return gzipped if gzipped.exists() and not plain.exists() else plain


def load_intensities(path, sample, channel=None, column=None):
    """Return (probe to intensity, probe to passed detection or None, text naming the source)."""
    path = Path(path)
    table = gsm_table(path, sample) if path.is_dir() else path
    if not path.is_dir() and "series_matrix" in path.name:
        meta, samples, rows = read_series_matrix(path)
        if sample not in samples:
            raise Geo2KmersError(f"{sample} is not in {path}")
        counts = dict(zip(samples, meta.get("Sample_channel_count", [[]])[0]))
        if counts.get(sample, "1") != "1":
            sibling = gsm_table(path.parent / "gsm", sample)
            if (channel or column) and sibling.exists():
                table = sibling
            else:
                raise Geo2KmersError(
                    f"{sample} is a two-colour array, so VALUE in the series matrix is a ratio, not an intensity. "
                    "Give --channel ch1 or ch2 and keep the GSM tables from `fetch` in gsm/ beside the matrix."
                )
        else:
            if channel or column:
                raise Geo2KmersError(f"{sample} is a one-colour array, so --channel and --column do not apply here")
            index = samples.index(sample)
            values = {p: to_float(r[index]) for p, r in rows.items()}
            check_not_ratio(values, sample)
            sibling = gsm_table(path.parent / "gsm", sample)
            detection = read_detection(*read_table(sibling, "sample")[1:]) if sibling.exists() else None
            return values, detection, f"VALUE column of {path.name}"
    meta, columns, rows = read_table(table, "sample")
    two_colour = meta.get("Sample_channel_count", ["1"])[0] != "1"
    if two_colour and not (channel or column):
        raise Geo2KmersError(f"{sample} is a two-colour array. Give --channel ch1 or ch2.")
    name = channel_column(columns, channel, column) if (channel or column) else "VALUE"
    values = {r[columns[0]]: to_float(r.get(name)) for r in rows}
    check_not_ratio(values, sample)
    return values, read_detection(columns, rows), f"column '{name}' of {table.name}"


def read_detection(columns, rows):
    """Probe to passed, from a detection call or detection p value column. None if neither exists."""
    upper = {c.upper(): c for c in columns}
    call = next((upper[c] for c in DETECTION_COLUMNS if c in upper), None)
    p_col = next((upper[c] for c in DETECTION_P_COLUMNS if c in upper), None)
    cells = {r.get(call, "").strip().upper() for r in rows} if call else set()
    if call and cells - {""} and cells <= {"", "P", "M", "A", "PRESENT", "MARGINAL", "ABSENT"}:
        return {r[columns[0]]: r.get(call, "").strip().upper() in ("P", "PRESENT") for r in rows}
    if p_col:
        return {r[columns[0]]: to_float(r.get(p_col)) is not None and to_float(r.get(p_col)) <= 0.05 for r in rows}
    return None


def check_not_ratio(values, sample):
    numbers = [v for v in values.values() if v is not None]
    if numbers and min(numbers) < 0:
        raise Geo2KmersError(
            f"{sample} has negative values, so this column is not a plain intensity. It may be a log ratio, or a "
            "difference such as Affymetrix Average Difference. This tool does not sketch either. "
            "If the sample table has an intensity column, name it with --column."
        )
    if numbers and max(numbers) < LOG_SCALE_LIMIT:
        print(f"warning: every value of {sample} is below {LOG_SCALE_LIMIT}, so they may be logarithms of intensity")


def match_names(probes, values):
    """Key probes by the name in the intensity table: the whole FASTA header, or else its first word."""
    matched = {name if name in values else name.split()[0]: sequence for name, sequence in probes.items()}
    if len(matched) < len(probes):
        raise Geo2KmersError("two FASTA headers share a first word and neither is in the intensity table")
    return matched


def add_probe(table, kmers, weight, probe_index):
    """Add one probe's k-mers. A k-mer seen twice in a probe adds its weight twice."""
    for kmer in kmers:
        row = table[kmer]
        row[0] += weight
        if row[1] != probe_index:
            row[1] = probe_index
            row[2] += 1


def build_sketch(probes, intensities, k, alphabet, mode, detection=None, proteome=None):
    """Return (table, report). table maps k-mer to [weight, last probe index, n_probes].

    Protein, dayhoff and hp k-mers come from one stretch of each probe: the one ProteomeFrames picks if
    proteome, a list of FASTA paths, is given, and otherwise the longest.
    """
    if alphabet not in ("dna", "protein", "dayhoff", "hp"):
        raise Geo2KmersError(f"unknown alphabet {alphabet}")
    if mode not in ("present", "weighted"):
        raise Geo2KmersError(f"unknown mode {mode}")
    if k < 1:
        raise Geo2KmersError(f"k must be at least 1, not {k}")
    if proteome and alphabet == "dna":
        raise Geo2KmersError("--proteome picks a reading frame, so it does not apply to alphabet dna")
    picker = ProteomeFrames(proteome, k, alphabet) if proteome else None
    measured = {p: v for p, v in intensities.items() if v is not None}
    if not measured:
        raise Geo2KmersError("no probe has a measured intensity")
    numbers = np.array(list(measured.values()), dtype=float)
    report = {
        "alphabet": alphabet,
        "k": k,
        "mode": mode,
        "probes_in_intensity_table": len(intensities),
        "probes_with_a_measured_intensity": len(measured),
        "probes_with_a_sequence": len(probes),
        "probes_with_sequence_and_intensity": len(set(probes) & set(measured)),
        "probes_with_intensity_zero": int((numbers == 0).sum()),
    }
    weights = {}
    if mode == "weighted":
        background = float(np.percentile(numbers, BACKGROUND_PERCENTILE))
        report["rule"] = (
            f"max(0, intensity - background), background = {BACKGROUND_PERCENTILE}th percentile of this array"
        )
        report["background"] = background
        weights = {p: max(0.0, v - background) for p, v in measured.items()}
    elif detection is not None:
        report["rule"] = "GEO detection call"
        weights = {p: 1.0 for p in measured if detection.get(p)}
    else:
        threshold = float(np.median(numbers))
        report["rule"] = "intensity above the median of this array"
        report["threshold"] = threshold
        weights = {p: 1.0 for p, v in measured.items() if v > threshold}
    report["probes_used"] = sum(p in probes for p in weights)

    table = defaultdict(lambda: [0.0, -1, 0])
    no_stretch = no_kmers = 0
    picked = []
    expected_sum = 0.0
    longest = 0
    for index, (probe, weight) in enumerate(sorted(weights.items())):
        if probe not in probes:
            continue
        sequence = probes[probe]
        longest = max(longest, len(sequence))
        if alphabet == "dna":
            kmers = dna_kmers(sequence, k)
        else:
            pick = picker.pick(sequence) if picker else longest_stretch(sequence)
            kmers = stretch_kmers(pick[2], k, alphabet) if pick else []
            no_stretch += pick is None
            if pick:
                picked.append((probe, pick[0], pick[1], len(pick[2]), pick[3]))
        no_kmers += not kmers
        expected_sum += weight * len(kmers)
        add_probe(table, kmers, weight, index)
    report["probes_with_no_kmer"] = no_kmers
    if alphabet != "dna":
        report["probes_with_no_stretch"] = no_stretch
        report["frame_rule"] = "longest stretch between stops"
        report["picked_stretches"] = picked
    if picker:
        report["frame_rule"] = (
            f"longest stretch between stops with at least {MIN_STRETCH_AA} amino acids and at least "
            f"{MIN_SHARE_IN_PROTEOME:.0%} of its k-mers in the proteome"
        )
        report["proteome"] = ",".join(Path(p).name for p in proteome)
        report["proteome_distinct_kmers"] = len(picker.known)
    report["distinct_kmers"] = len(table)
    if not table:
        raise Geo2KmersError(
            f"no probe gave a k-mer at k = {k} with alphabet {alphabet}: {no_stretch} of "
            f"{report['probes_used']} probes had no stretch, and the longest probe used is {longest} nt"
        )
    report["sum_of_probe_weight_times_kmers"] = expected_sum
    report["sum_of_weight"] = float(sum(row[0] for row in table.values()))
    return table, report


def rounded_abundance(weight):
    """Round half up. A weight above 0 is at least 1, so rounding never drops a k-mer."""
    return max(1, math.floor(weight + 0.5))


def write_signature(table, path, k, alphabet, mode, name):
    """Write a sourmash signature if sourmash is installed. Returns True if it wrote one."""
    try:
        import sourmash
        from sourmash.sourmash_args import SaveSignaturesToLocation
    except ImportError:
        return False
    args = {"n": 0, "ksize": k, "scaled": 1, "track_abundance": mode == "weighted"}
    if alphabet == "protein":
        args["is_protein"] = True
    elif alphabet in REDUCED:
        args[alphabet] = True
    minhash = sourmash.MinHash(**args)
    hash_of = minhash.seq_to_hashes if alphabet == "dna" else None
    abundances = {}
    for kmer, row in table.items():
        if mode == "weighted" and row[0] <= 0:
            continue
        if alphabet == "dna":
            value = hash_of(kmer)[0]
        else:
            value = sourmash.minhash.hash_murmur(kmer)
        abundances[value] = rounded_abundance(row[0])
    if mode == "weighted":
        minhash.set_abundances(abundances)
    else:
        minhash.add_many(list(abundances))
    signature = sourmash.SourmashSignature(minhash, name=name)
    with SaveSignaturesToLocation(str(path)) as save:
        save.add(signature)
    return True


def write_sketch(table, report, out, sample, channel):
    """Write the TSV, the report next to it, and the signature if sourmash is installed."""
    mode, alphabet, k = report["mode"], report["alphabet"], report["k"]
    value_name = "intensity_weight" if mode == "weighted" else "presence"
    with open(out, "w") as f:
        f.write(f"kmer\t{value_name}\tn_probes\n")
        for kmer in sorted(table):
            weight, _, n_probes = table[kmer]
            value = f"{weight:.6f}" if mode == "weighted" else "1"
            f.write(f"{kmer}\t{value}\t{n_probes}\n")
    name = f"{sample}{'_' + channel if channel else ''}"
    wrote = write_signature(table, f"{out}.sig", k, alphabet, mode, name)
    report["sample"] = sample
    report["channel"] = channel
    report["signature_written"] = wrote
    picked = report.pop("picked_stretches", None)
    if picked is not None:
        with open(f"{out}.frames.tsv", "w") as f:
            f.write("probe_id\tframe\tstart_aa\tlength_aa\tshare_in_proteome\n")
            f.writelines(
                f"{probe}\t{frame}\t{start}\t{length}\t{'' if share is None else format(share, '.3f')}\n"
                for probe, frame, start, length, share in picked
            )
    with open(f"{out}.report.json", "w") as f:
        json.dump(report, f, indent=2)
