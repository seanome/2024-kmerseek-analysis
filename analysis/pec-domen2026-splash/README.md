# Notebook 361: Domen et al. 2026 PEC RNA-seq, reproduced without a genome

Domen et al. 2026 (PNAS 123(22): e2610968123, https://doi.org/10.1073/pnas.2610968123) report genes
that change after pulsatile electrical current (PEC) in *Botryllus schlosseri*. They aligned reads to the
genome (STAR, HTSeq, DESeq2). This folder stages a planned analysis without a genome: SPLASH 2 will test 31-nt
anchors whose downstream sequence differs between PEC and control, an anchor-count model (PyDESeq2)
will give the abundance view, and the changed sequences will be translated and matched to the new Botryllus
proteome and to Swiss-Prot.

Status: in progress. Results and the report are added as stages finish.

## Layout

| path | what |
|---|---|
| `handoff/` | the starting package as received on 2026-10-23 (inferred design, capped downloader, first runner). Kept unchanged for provenance. |
| `design/` | the declared analysis plan and SI-driven design builder; `design/samples.tsv` lists all 64 runs |
| `environment-pec361.yml` | pinned tools |
| `pec-tools/` | Rust: six-frame translation with coordinates, and the read simulator for the known-answer control |
| `control/` | known-answer control design (3,000 parent transcripts, 3,100 sequences including paired isoforms) |
| `scripts/`, `slurm/` | stage scripts; every stage writes a done-marker and is restartable |

Runs on Sherlock under `$SCRATCH/nb361-pec`.

## Review limits

No notebook or differential-expression result is present yet. `design/samples.tsv` (64 runs) and the six `contrast_*.design.tsv` files, with pair IDs, are committed. A review checked that the contrast sizes (47 samples) and the pair counts in the design files match `design/DESIGN_NOTES.md`. It did not check the pairing rule in `freeze_design.py` against the SI workbook, which was not available. The pairs come from a rule on genotype and age, not from a pairing column in the SI, so treat the pairing as unverified until that check is done.

`handoff/` remains an unchanged historical package. Its runner contains container paths and inferred labels, skips missing databases, and can write a search success marker after failed searches. Do not run it as the production workflow. The checked-in Slurm stages are separate.

The simulator targets an expected library size, with actual counts saved in `truth_counts.tsv`. Its large-mean Poisson draws use a normal approximation. This is a software control, not a validated RNA-seq dispersion model. Protein sketch matches remain leads requiring the alignment and domain checks specified in the analysis plan.

Review checks use tiny local inputs only. Running Sherlock jobs, their data, and completion markers were not changed.
