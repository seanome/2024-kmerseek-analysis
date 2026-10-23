# Notebook 361: Domen et al. 2026 PEC RNA-seq, reproduced without a genome

Domen et al. 2026 (PNAS 123(22): e2610968123, https://doi.org/10.1073/pnas.2610968123) report genes
that change after pulsatile electrical current (PEC) in *Botryllus schlosseri*. They aligned reads to the
genome (STAR, HTSeq, DESeq2). This folder repeats the analysis with no genome: SPLASH 2 finds 27-nt
anchors whose downstream sequence differs between PEC and control, an anchor-count model (PyDESeq2)
gives the abundance view, and the changed sequences are translated and matched to the new Botryllus
proteome and to Swiss-Prot.

Status: in progress. Results and the report are added as stages finish.

## Layout

| path | what |
|---|---|
| `handoff/` | the starting package as received on 2026-10-23 (inferred design, capped downloader, first runner). Kept unchanged for provenance. |
| `design/` | the frozen sample-to-contrast table, taken from SI Dataset S1, written before any analysis |
| `env/pec361.yml` | pinned tools |
| `pec-tools/` | Rust: six-frame translation with coordinates, and the read simulator for the known-answer control |
| `control/` | known-answer control design (3,000 transcripts, planted changes) |
| `scripts/`, `slurm/` | stage scripts; every stage writes a done-marker and is restartable |

Runs on Sherlock under `$SCRATCH/nb361-pec`.
