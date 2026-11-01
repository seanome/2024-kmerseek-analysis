# Draft to the DIAMOND authors (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: Benjamin Buchfink (DIAMOND)

Subject: Checking how we ran DIAMOND for a protein domain annotation leaderboard

Dear Dr Buchfink,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. One row is the step most annotation pipelines run first:
search each protein against Swiss-Prot, take the best hit, and copy that hit's Pfam domains
onto the query. DIAMOND is one of the two search tools in that row, and before the row goes
public we would like you to check that we ran it fairly.

What we ran:

- DIAMOND 2.2.8 (quay.io/biocontainers/diamond@sha256:e02013280712f1295e59339dc6ea162adf8a6bbdd730e73a9c00700c03c3e556)
- `diamond makedb --in reference.fasta -d reference`
- `diamond blastp --query queries.fasta --db reference.dmnd --sensitive` and, separately,
  `--ultra-sensitive`; all other options at their defaults (E-value 0.001, 25 targets
  per query), with `--outfmt 6 qseqid sseqid qstart qend sstart send bitscore evalue
  pident qseq_gapped sseq_gapped`
- Queries: 998 human proteins. Reference: Swiss-Prot 2026_03 with every mammal removed
  (507,740 entries), so the nearest relative left is a bird or a reptile.
- The best hit is the row with the highest bit score. Each Pfam domain on that hit (found
  by hmmscan with Pfam's gathering thresholds) that the alignment covers by at least half
  is placed on the query residue by residue through the aligned sequences.

The tool card, with the scores, their number of truth instances, the hardware and the
run time, will be attached here once the run finishes: [card link, to fill in].

Is there a setting you would use instead for this task, for example a different `-k`,
`--top`, or an E-value cutoff? We will rerun with your choice and show both rows.

Best wishes,
Olga Botvinnik
