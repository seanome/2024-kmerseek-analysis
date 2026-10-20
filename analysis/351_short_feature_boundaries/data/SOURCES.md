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
| Downloaded | not downloaded yet; it goes straight to Sherlock, not the laptop |
| SHA-256 | not computed yet |

## Pfam-A

Not downloaded yet.
