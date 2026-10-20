# Draft to the TM-Vec and DeepBLAST authors (not sent; waiting for Olga)

Status: draft. Nothing has been sent. Per the leaderboard plan (PR 147, days 11-12), each
tool's parameters go to its authors with its card before the row is public. Olga decides
whether and when to send this, and from which address.

To: Tymor Hamamsy, Jamie Morton, Richard Bonneau (TM-Vec, DeepBLAST)

Subject: Checking how we ran TM-Vec and DeepBLAST for a protein domain annotation leaderboard

Dear Dr Hamamsy, Dr Morton and Dr Bonneau,

We are building a public leaderboard of protein annotation tools for genome projects such
as the Earth BioGenome Project. TM-Vec has a row in it, and before the row goes public we
would like you to check that we ran it fairly.

What we ran:

- tm-vec 1.0.3 and deepblast 1.0.2 from PyPI, in a CUDA image built from our
  Dockerfile.tmvec (digest on the card), one GPU per target proteome
- `tmvec-build-database` on each target proteome, then `tmvec-search
  --k-nearest-neighbors 1` with `--output-format tabular`, and again with
  `--output-format alignment --deepblast-model ...` for the structural alignment
- Model: tm_vec_cath_model with ProtT5-XL-UniRef50, and the DeepBLAST checkpoint
  [name to fill in once downloaded]. The URLs in the DeepBLAST wiki returned 404 on
  2026-10-16; we took the files from [source to fill in].
- Queries: 998 human proteins, against nine reference proteomes (mouse to E. coli).
- The best hit's Pfam domains are placed on the query through DeepBLAST's alignment, and
  the placed domain is scored by overlap with the query's own Pfam domain. A best hit
  that DeepBLAST does not align gives no placed call.

Two questions: would you use tm_vec_cath_model or tmvec_swiss_model_large for full-length
proteins, and which DeepBLAST checkpoint goes with it? We will rerun with your choice and
show both rows.

The tool card, with the scores, their number of truth instances, the GPU model and the
GPU hours, will be attached once the run finishes: [card link, to fill in].

Best wishes,
Olga Botvinnik
