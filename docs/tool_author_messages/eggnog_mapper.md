# Draft to the eggNOG-mapper authors (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: Carlos P. Cantalapiedra and Jaime Huerta-Cepas (eggNOG-mapper)

Subject: Checking how we ran eggNOG-mapper for a protein domain annotation leaderboard

Dear Dr Cantalapiedra and Dr Huerta-Cepas,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. eggNOG-mapper has a row in it, and before the row goes
public we would like you to check that we ran it fairly.

What we ran:

- eggNOG-mapper 2.1.15 (quay.io/biocontainers/eggnog-mapper@sha256:f70babaf681ff4b6b2fc8e8e76754bf989f01dd4910ca91f156c22aa88ea70d3)
  with emapperdb 5.0.2
- `emapper.py -i queries.fasta --itype proteins -m diamond --data_dir emapperdb-5.0.2 --cpu 8`,
  every other option at its default
- Queries: 998 human proteins.
- The call is the PFAMs column of `.emapper.annotations`: each named family becomes one
  call on the query. The column carries no coordinates, so the call claims no position:
  only the family-level score is reported, and the position scores on the card read
  "not applicable", not zero. eggNOG 5 contains human, so the row is labelled
  "reference-database tool, graded on its own models".

Would you score eggNOG-mapper with `--pfam_realign` for a task about where on the
protein a domain sits? We left it off because it is not the default; we can add it as a
second row.

The tool card, with the scores, their number of truth instances, the hardware, the run
time and the database size, will be attached once the run finishes: [card link, to fill in].

Best wishes,
Olga Botvinnik
