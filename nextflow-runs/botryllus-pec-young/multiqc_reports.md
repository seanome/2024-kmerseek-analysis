# MultiQC reports for the botryllus-pec-young runs

The reports stay on Sherlock and the laptop, not in this repo, because FastQC lists
overrepresented Botryllus sequences. Made 2026-11-13.

| Report | What it covers | Where on Sherlock |
|---|---|---|
| PR 203 trimming | FastQC raw and trimmed, Trim Galore (`--length 62`), 64 libraries | `$SCRATCH/botryllus-pec-young/multiqc-extra/pr203-trimming/` |
| rnaseq | 2024 genome, PR 203 reads: STAR, samtools, Qualimap, dupRadar, Picard, strandedness | `$SCRATCH/botryllus-pec-young/rnaseq/multiqc/star_salmon/` |
| rnaseq-paper-genome | paper's genome, PR 203 reads: STAR, samtools (`skip_qc: true`) | `$SCRATCH/botryllus-pec-young/rnaseq-paper-genome/multiqc/star_salmon/` |
| rnaseq-paper-genome-length20 | paper's genome, trimmed at 20: Trim Galore, FastQC, STAR, samtools; rerun with `--skip_qc false` (job 52190636) adds RSeQC, Qualimap, dupRadar, Preseq | `$SCRATCH/botryllus-pec-young/rnaseq-paper-genome-length20/multiqc/star_salmon/` |
| star-too-short-test | STAR on 500,000 pairs of 2 libraries, 4 ways (`star_too_short_test.py`, job 52078147) | `$SCRATCH/botryllus-pec-young/multiqc-extra/star-too-short-test/` |
| star-relaxed-vs-default | STAR default vs 30-base minimum plus featureCounts, 500,000 pairs of 38 libraries (`star_relaxed_vs_default.py`, job 52189672) | `$SCRATCH/botryllus-pec-young/multiqc-extra/star-relaxed-vs-default/` |

How the extra reports were made: MultiQC from the nf-core/rnaseq 3.27.0 MULTIQC image
(`community-cr-prod.seqera.io-...-c8e346f4...-data.img` in `$SCRATCH/apptainer-cache`) with
that pipeline's `assets/multiqc_config.yml`, `--dirs` for the STAR tests. The PR 203 run
published only the text Trim Galore reports; MultiQC's Trim Galore module reads the JSON ones,
so those were linked in from its work folders
(`$SCRATCH/botryllus-shazam-PEC-20261023/nfcore/work/*/*_trimming_report.json`, newest per file).
