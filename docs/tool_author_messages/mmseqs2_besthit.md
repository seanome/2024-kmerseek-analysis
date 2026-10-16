# Draft to the MMseqs2 authors (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: Martin Steinegger and Johannes Söding (MMseqs2)

Subject: Checking how we ran MMseqs2 for a protein domain annotation leaderboard

Dear Dr Steinegger and Dr Söding,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. One row is the step most annotation pipelines run first:
search each protein against Swiss-Prot, take the best hit, and copy that hit's Pfam domains
onto the query. MMseqs2 is one of the two search tools in that row, and before the row goes
public we would like you to check that we ran it fairly.

What we ran:

- MMseqs2 from quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef
  (the exact version string is recorded on the card from `mmseqs version`)
- `mmseqs createdb` for the queries and the reference
- `mmseqs search query/db reference/db result tmp -s 7.5 -a`, one round (no
  `--num-iterations`), other options at their defaults (E-value 0.001, `--max-seqs 300`)
- `mmseqs convertalis ... --format-output query,target,qstart,qend,tstart,tend,bits,evalue,pident,qaln,taln`
- Queries: 998 human proteins. Reference: Swiss-Prot 2026_03 with every mammal removed
  (507,740 entries), so the nearest relative left is a bird or a reptile.
- The best hit is the row with the highest bit score. Each Pfam domain on that hit (found
  by hmmscan with Pfam's gathering thresholds) that the alignment covers by at least half
  is placed on the query residue by residue through the aligned sequences.

We used one round because, in an earlier test on this project, three rounds gave shuffled
decoy queries hits at E-values near 1e-22, and one round gave none below 0.026.

The tool card, with the scores, their number of truth instances, the hardware and the
run time, will be attached here once the run finishes: [card link, to fill in].

Is there a setting you would use instead for this task? We will rerun with your choice
and show both rows.

Best wishes,
Olga Botvinnik
