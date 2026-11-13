# Draft to the PLMSearch authors (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: Wei Liu, Hong-Bin Shen and colleagues (PLMSearch)

Subject: Checking how we ran PLMSearch for a protein domain annotation leaderboard

Dear Dr Liu and Dr Shen,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. PLMSearch has a row in it, and before the row goes public
we would like you to check that we ran it fairly.

What we ran:

- PLMSearch at commit d6df1dc (github.com/maovshao/PLMSearch), in a CUDA image built from
  our Dockerfile.plmsearch, with model/plmsearch.sav and the ESM-1b weights from
  plmsearch_data.tar.gz (Zenodo 11480660)
- `embedding_generate.py` on the queries and on each target proteome, then
  `main_similarity.py -iqe query.pkl -ite target.pkl -smp plmsearch.sav`: the
  SS-predictor search, without the PfamClan prefilter
- Queries: 998 human proteins, against nine reference proteomes (mouse to E. coli).
- The best hit's Pfam families are the call. PLMSearch reports no alignment, so the call
  claims no position on the query: only the family-level score is reported, and the
  position scores on the card read "not applicable", not zero.

We left out the PfamClan prefilter because it runs PfamScan, and our truth set is Pfam
domain instances, so the search would be partly graded on the truth's own models. Is
there a setting you would rather we run? Would you place domains with PLMAlign's
alignment? We can add either as a second row.

The tool card, with the scores, their number of truth instances, the GPU model and the
GPU hours, will be attached once the run finishes: [card link, to fill in].

Best wishes,
Olga Botvinnik
