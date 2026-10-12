#!/usr/bin/env python3
"""Build notebooks/280_time_split_pfam_33_to_38.ipynb. Its first cell holds the
question, the data, the predictions and the kill rule, written before any search ran."""

from pathlib import Path

import nbformat as nbf

INTRO = r"""# 280. Time split: would kmerseek have placed Pfam domains before Pfam did?

There is no answer key for proteins nobody has characterized, so a search over them cannot
be scored. A time split makes one. We take proteins that had no Pfam domain in an old Pfam
release and gained one in a later release, search them against the reference as it was at
the old release, and score each tool against what the later release found. CAFA scores
function prediction the same way (Zhou et al. 2019, *Genome Biology* 20: 244, https://doi.org/10.1186/s13059-019-1835-8).

**Data.**

- **t0** = Pfam 33.1 (May 2020), built on UniProtKB reference proteomes 2019_08, 18,259
  Pfam-A families. Queries are Pfam 33.1 `pfamseq` sequences with no row in its
  `Pfam-A.regions.tsv.gz`.
- **t1** = Pfam 38.2 (data files dated 22 January 2026, published June 2026), built on
  reference proteomes 2025_03, 30,134 Pfam-A families. A query gained a domain when its
  identical sequence (same MD5) has a row in the 38.2 regions file.
- **Reference searched by every tool**: Swiss-Prot as of UniProt release 2019_08
  (`uniprot_sprot-only2019_08.tar.gz`), with the Pfam 33.1 domains on each protein from
  `Pfam-A.regions.uniprot.tsv.gz`.
- **kmerseek build**: `docker.io/olgabot/kmerseek:0.4.0-rc5`
  (`sha256:7855826572644933e0c71926102b90174e650c144ae90b3641b760bd9f463afb`), the build
  the dark-set 0.4 run uses.
- Code: [`nextflow-runs/time-split-pfam/`](../nextflow-runs/time-split-pfam/).

**Scoring, fixed before any search.** A tool *transfers* family F to a query region when a
hit's interval on the target overlaps a t0 domain of F on that target. An instance is
*found* when a tool transfers the same family and at least half of the call's residues lie
inside the t1 domain. Errors are counted on decoys only (one dipeptide-shuffled copy per
query, which keeps every adjacent residue pair), because t1 Pfam is not complete: a call
t1 does not confirm on a real query is not proof of an error, but any transfer to a decoy
is one. Each tool's score cutoff is set on the *tune* half of the queries so decoys receive
at most 1 family transfer per 100 decoy queries; recall is reported on the *test* half. The
halves are fixed by the first byte of SHA-1(accession): even goes to tune.

**Predictions, written before any search.**

- **P1.** Most instances sit below 30% identity to the closest t0 Swiss-Prot domain of the
  same family.
- **P2.** hmmscan with the t0 Pfam models (no gathering thresholds, `-E 0.001 --domE
  0.001`) finds the largest share of instances.
- **P3.** Over all instances at the matched error rate, kmerseek finds fewer than MMseqs2.
  Figure 3 says kmerseek stops working near 30% identity.
- **P4, the claim kmerseek needs.** For Motif, Disordered and Repeat families, and for
  regions under 60 aa, kmerseek's *unique finds* (instances found by kmerseek and by no
  other tool, each at its own matched cutoff) exceed the number expected from its decoys,
  with a 95% interval above zero, on the test half.

**Kill rule.** If P4 fails on the test half, this is written up as a negative result and
the test stops. No kmerseek settings are added, no cutoffs changed and "found" is not
redefined after the test half has been read.

**The decision this informs.** If P4 holds, the paper can claim that on proteins nobody had
characterized, kmerseek placed short features (Motif, Disordered, Repeat, or under 60 aa)
that no other tool placed, more often than chance. This is the Figure 4 candidate. If P4
fails, kmerseek adds nothing over MMseqs2 and Pfam's own models on uncharacterized proteins,
and Figure 4 reports that.
"""


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(__file__).resolve().parents[1] / "notebooks/280_time_split_pfam_33_to_38.ipynb"
    nbf.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
