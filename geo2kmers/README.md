# geo2kmers

geo2kmers turns microarray data from GEO (the NCBI Gene Expression Omnibus) into one k-mer table per
sample, so array data can be set beside k-mer sketches made from RNA-seq. A k-mer is a stretch of k
letters. A sketch is a list of k-mers, here with a weight for each. A probe is one spot on an array,
with a known sequence. It runs on a CPU; nothing needs a GPU.

```mermaid
flowchart LR
  A[GEO series GSE] -->|fetch| B[platform table GPL,<br/>series matrix, GSM tables]
  B -->|sequences| C[probes.fasta<br/>one sequence per probe]
  C --> D[sketch]
  B --> D
  D --> E[TSV of k-mers,<br/>report JSON,<br/>sourmash signature]
```

## What is measured and what is computed

| value | kind | from what |
|---|---|---|
| intensity | measured | GEO: the sample's VALUE column, or a channel column on two-colour arrays |
| detection call | measured | GEO: a detection column in the sample table, when there is one |
| background | computed | the 5th percentile of that array's intensities |
| threshold | computed | the median of that array's intensities |
| `intensity_weight` | computed | max(0, intensity - background), summed over the probes that hold the k-mer |

An intensity is not a count. It is on a scale of its own for each array, it stops rising when the
scanner saturates, and it includes background. So the weight column is called `intensity_weight`.

## Install

```bash
conda env create -f environment.yml
```

The commands run with `python -m geo2kmers` from this folder.

## Commands

### fetch

```bash
python -m geo2kmers fetch GSE2183 --outdir data/GSE2183
```

Downloads the platform table, the series matrix and, when needed, the GSM (sample) tables, over HTTPS
from GEO. Files already on disk are not downloaded again. `provenance.json` records the accession, the
date and each file's URL and SHA-256 checksum. `samples.tsv` lists the samples and how many channels
each has. Add `--sample-tables` to get GSM tables for a one-colour series too; only they hold detection
calls. A series with several platforms needs `--gpl`.

### sequences

```bash
export NCBI_EMAIL=you@example.org
python -m geo2kmers sequences data/GSE2183/GPL1799.txt --out data/GSE2183/probes.fasta
```

- A probe's sequence comes from the platform's SEQUENCE column if the row has one.
- Otherwise it comes from NCBI, using the row's GenBank accession (columns GB_ACC, GB_LIST and similar).
  Requests go in batches of 200 with at least 0.34 seconds between them, or 0.11 seconds if
  `NCBI_API_KEY` is set. NCBI asks for a contact address, so the tool reads it from `NCBI_EMAIL` and
  stops if it is not set. Fetched sequences are kept in `genbank_cache.fasta`.
- For an Affymetrix platform with neither column it stops and names the Bioconductor probe package
  that holds the 25-letter probes, such as `hgu133plus2probe`. It does not guess sequences.

`probe_sequence_source.tsv` has one row per probe: probe_id, source (gpl_sequence, genbank_fetch or
missing) and length_nt.

### sketch

```bash
python -m geo2kmers sketch --fasta data/GSE2183/probes.fasta \
  --intensities data/GSE2183/GSE2183_series_matrix.txt.gz \
  --sample GSM39421 --channel ch2 --k 31 --alphabet dna --mode weighted --out GSM39421.tsv
```

`--intensities` takes a series matrix, one GSM table, or a folder of GSM tables.

Alphabets:
- `dna`: every k-mer of each probe in canonical form, meaning the smaller of the k-mer and its reverse
  complement. Windows holding a letter other than A, C, G or T are skipped.
- `protein`: the probe is translated in all six frames with the standard genetic code. A frame with a
  stop codon anywhere is dropped, and each remaining frame is cut into k-mers. A codon with an unknown
  letter becomes X and no k-mer crosses it.
- `hp`: the same, with each amino acid then written as h (hydrophobic: A F G I L M P V W Y) or p (polar:
  C D E H K N Q R S T). Those groups are the ones sourmash 4.9.4 uses; `tests/test_units.py` checks our
  hashes against sourmash's own.

A 25 nt probe gives at most 8 amino acids in a frame, so protein k must be 8 or less on short-probe
arrays. Long probes do not solve the problem by themselves; see the result below.

Modes:
- `present`: the output is the set of k-mers from expressed probes. A probe is expressed if the sample
  table's detection call says so. A call of P passes, and a detection p value of 0.05 or less passes.
  With no detection column, a probe is expressed if its intensity is above that array's median. The
  command prints which rule it used.
- `weighted`: each probe's weight is max(0, intensity - background). A k-mer's `intensity_weight` is
  the sum of the weights of its probes. A k-mer that occurs twice in a probe adds that probe's weight
  twice, so the sum of `intensity_weight` over all k-mers equals the sum over probes of probe weight
  times number of k-mers in the probe. K-mers of probes with weight 0 stay in the TSV with weight 0.

Outputs: a TSV with columns kmer, intensity_weight (or presence) and n_probes, where n_probes counts
the probes holding the k-mer; `<out>.report.json` with every count and the rule used; and, if sourmash
is installed, `<out>.sig`.

The signature keeps every k-mer (scaled = 1). In `weighted` mode a sourmash abundance must be a whole
number, so each weight is rounded: abundance = max(1, floor(weight + 0.5)). A k-mer with weight 0 is
left out of the signature. These abundances are rounded intensities, not read counts.

## Two-colour arrays

A two-colour array hybridises two samples to one slide, each with its own dye. Each dye is a channel.
On these arrays the series matrix holds a log ratio between the channels, not an intensity. So the
tool refuses it, and `fetch` also downloads the GSM tables, which hold each channel's signal. Choose
one with `--channel ch1` or `--channel ch2`. If it cannot find the column, name it with `--column`. A
table with negative numbers in the chosen column is refused too, as a log ratio. The tool uses the
signal column as GEO gives it and does not subtract GEO's local background columns.

## Workflow

`main.nf` runs fetch, sequences and sketch for every sample of a series, one process per step, and one
sketch per channel on two-colour arrays. Each process runs an inline Python script through an absolute
interpreter path in its first line, because a `conda` directive does not reliably reach an inline
script. Each process has its own path: `--python_fetch`, `--python_sequences` and `--python_sketch`.
The default paths are on Sherlock, so change them on another machine.

```bash
nextflow run main.nf -profile small --gse GSE2183 --outdir results/GSE2183
```

Every process retries when the exit status is empty, `Integer.MAX_VALUE` or in 128 to 143, which mean
something outside the task killed it. Any other failure stops the run with an error.

## Result on a real series

GSE2183 is a mussel series, described in `tests/README.md`. The run in `results/GSE2183/` used a 4 CPU,
16 GB job with no GPU and no scheduler inside it. All numbers below come from the file named beside them.

- 50 tasks ran, each on the first try, with exit status 0 (`geo2kmers.trace.txt`).
- The run took 83 seconds (`wall_time.txt`), under the 10 minute limit.
- The platform has 3840 probes. 3374 got a sequence from GenBank and 466 have none. The 3374 are 1673
  distinct sequences, so most genes are spotted more than once. Lengths run from 101 to 16740 nt, median
  513 (`probe_summary.tsv`).
- Each of the 4 samples has 2 channels, so 8 channels at 3 alphabets and 2 modes make 48 sketches
  (`sketch_summary.tsv`).
- DNA k = 31, `weighted`: 709173 distinct k-mers in every sketch. The set is the same because an array
  has a fixed set of probes; only the weights differ (`sketch_summary.tsv`).
- DNA k = 31, `present`: 1744 to 1835 probes pass the median rule, giving 397918 to 459149 distinct
  k-mers (`sketch_summary.tsv`).
- The median is 0 in GSM39424 ch1, and background is 0 in 7 of the 8 channels. Between 148 and 2008 of
  the 3840 probes have intensity exactly 0 (`sketch_summary.tsv`). GEO's table does not say why.
- Protein and hp: 2608 of the 3374 probes have zero usable frames, so 766 probes supply all the k-mers
  (`sketch_summary.tsv`). A sequence around 500 nt usually has a stop codon in all six frames, so
  dropping such frames removes most of these long probes. On this array the protein and hp sketches
  see only a small part of the probes.

## Limits

- The sequence fetched for a GenBank accession is the whole record, which need not be what was spotted
  on the slide. The longest here is 16740 nt, far longer than a typical EST (a short sequence read from
  a messenger RNA).
- Probes sharing an accession add their weights, so a gene spotted twice weighs twice as much.
- Weights from different arrays are on different scales, and no normalisation between arrays is done.
- The sketches have not yet been compared with an RNA-seq sketch.

## Tests

```bash
python -m pytest tests
```

`tests/README.md` describes the test series and the known-answer checks.
