# Draft to the InterProScan team (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: InterPro team, EMBL-EBI (InterProScan)

Subject: Checking how we ran InterProScan for a protein domain annotation leaderboard

Dear InterPro team,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. InterProScan has a row in it, and before the row goes
public we would like you to check that we ran it fairly.

What we ran:

- InterProScan 5.78-109.0 (docker.io/interpro/interproscan@sha256:dc58b7c147fbbf00c2dd4f5ced42121fc1e8841fcbc7cc2c484380248ff76d11),
  with the data directory of the 5.78-109.0 full distribution after `setup.py`
- `interproscan.sh --input queries.fasta --formats tsv --cpu 16 --disable-precalc --iprlookup`,
  with no `-appl`, so every member database the distribution ships is run
- Queries: 998 human proteins.
- The score is on Pfam domain instances (Pfam 38.2). We score InterProScan's Pfam matches
  with their coordinates. Because InterPro's Pfam member is the source of that truth, and
  its databases contain human, the row is labelled "reference-database tool, graded on
  its own models", and it sits in its own column group, next to hmmscan against Pfam-A.
- We used `--disable-precalc` so the run time on the card is the time to compute every
  match, not the time to look them up.

The tool card, with the scores, their number of truth instances, the hardware, the run
time and the database size, will be attached once the run finishes: [card link, to fill in].

Is there anything you would run differently for this task, or a way you would rather the
row be labelled? We will rerun with your choice and show both rows.

Best wishes,
Olga Botvinnik
