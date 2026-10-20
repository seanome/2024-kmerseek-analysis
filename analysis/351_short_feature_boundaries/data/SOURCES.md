# Data sources for notebook 351

Every file used by this analysis, where it came from, and when.

## ELM instance table

| | |
|---|---|
| URL | http://elm.eu.org/instances.tsv?q=* |
| Downloaded | 2026-10-20 (file header: `#ELM_Instance_Download_Date: 2026-10-20 22:10:22`, server time) |
| ELM download format | version 1.4, 4,277 instances in the header and in the file |
| Local file | `raw_elm/elm_instances.tsv` |
| Size | 944,144 bytes |
| SHA-256 | 68c1832950fc3ed30399a247059979a60c4e95974a5d18c9dbc291c8ee578592 (as downloaded, CRLF line ends) |
| SHA-256 in git | c4279592ff4a2c0c9d7c2d971a087b4b36c417262389cf6ef999e430513004db (git stores it with LF line ends) |

The URL with `&instance_logic=true+positive` returned HTTP 500 on 2026-10-20, so the full
table was downloaded and filtered to true positives locally.

## UniProt sequences of the ELM proteins

| | |
|---|---|
| URL | https://rest.uniprot.org/uniprotkb/accessions?accessions=...&format=fasta (100 accessions per request); isoforms from https://rest.uniprot.org/uniprotkb/<acc>.fasta |
| Downloaded | 2026-10-20 |
| Script | `scripts/filter_elm_instances.py` |
| Local file | `filtered/query_proteins.fasta`, 2,572 sequences |
| Size | 2,024,659 bytes |
| SHA-256 | 6be1ba67d726139e0af0b80b3c76f49629fb07984eef932b53fd60fab924c871 |

Derived files from the same script:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `filtered/instances_kept.tsv` | 291,962 | dc3812d0af57f9db9581f05366644566c147d336ee67d401524561072ad55e6d |
| `filtered/filter_counts.tsv` | 208 | 197ee90acb9eb46f7be1dc7d6cda5fb16b83954d4e20a25f54465cfd9dded85f |

## UniRef50

| | |
|---|---|
| URL | https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz |
| Release | 2026_03, 02-Sep-2026, 38,840,027 clusters (from `uniref50.release_note`, read 2026-10-20) |
| Size | 8,780,552,383 bytes (HTTP Content-Length, 2026-10-20) |
| Downloaded | not downloaded: replaced by Swiss-Prot on 2026-10-20 (report.md, change 1) |
| SHA-256 | not computed |

## Swiss-Prot and Pfam-A

Downloaded on Sherlock by `sherlock/01_prepare.sbatch`, which writes the URL, download time,
size and SHA-256 of each file to `$SCRATCH/351_short_feature_boundaries/db/sources.tsv`.
Those rows are copied here when the run finishes. Not downloaded yet.

| File | URL |
|---|---|
| uniprot_sprot.fasta.gz | https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz |
| reldate.txt (Swiss-Prot release) | https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt |
| Pfam-A.hmm.gz | https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz |
| relnotes.txt (Pfam release) | https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/relnotes.txt |

## Software

| Tool | Version | Source |
|---|---|---|
| kmerseek | v0.4.0, commit 0aed5ca | built from https://github.com/seanome/kmerseek tag v0.4.0 |
| BLAST+ | 2.17.0 | docker://quay.io/biocontainers/blast:2.17.0--hb02a186_1 |
| HMMER | 3.4 | docker://quay.io/biocontainers/hmmer:3.4--h7d74f8d_5 |
| MMseqs2 | 18.8cc5c | docker://quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0 |

The preregistration named BLAST+ 2.16 "or the current build"; 2.17.0 was the current
build on 2026-10-20.
