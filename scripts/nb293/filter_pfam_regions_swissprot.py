"""Stream Pfam-A.regions.tsv.gz from stdin and keep rows whose accession is in Swiss-Prot.

Usage:
    gzip -dc Pfam-A.regions.tsv.gz | python3 filter_pfam_regions_swissprot.py ACCESSIONS OUT_TSV
ACCESSIONS is either a Swiss-Prot FASTA(.gz) (headers sp|ACC|NAME) or a text file with
one accession per line. Keeps the header line. Standard library only, so it runs with
Sherlock's system python3.
"""

import gzip
import sys

accessions_path, out_tsv = sys.argv[1], sys.argv[2]
opener = gzip.open if accessions_path.endswith(".gz") else open
accessions = set()
with opener(accessions_path, "rt") as fh:
    lines = fh.read().splitlines()
if lines and lines[0].startswith(">"):
    accessions = {line.split("|")[1] for line in lines if line.startswith(">")}
else:
    accessions = {line.strip() for line in lines if line.strip()}
n_in = n_out = 0
with open(out_tsv, "w") as fout:
    fout.write(sys.stdin.readline())
    for line in sys.stdin:
        n_in += 1
        if line[: line.index("\t")] in accessions:
            fout.write(line)
            n_out += 1
sys.stderr.write(
    "swissprot_accessions=%d rows_in=%d rows_out=%d\n" % (len(accessions), n_in, n_out)
)
