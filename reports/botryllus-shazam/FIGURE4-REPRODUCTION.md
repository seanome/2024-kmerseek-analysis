# Reproducing Domen et al. Figure 4

Source: supplied Domen et al. 2026 PDF, DOI 10.1073/pnas.2610968123, Figure 4 caption (page 5) and RNA-seq/signature methods (page 8). This is a reproduction specification, not a reproduced result.

## Panels A–B: differential gene expression and functional groups

1. Use the authors' exact Botryllus genome and gene annotation versions, library inclusion list and sample-to-group assignments. Their pipeline was Trim Galore → STAR → HTSeq gene counts → DESeq2. Libraries with fewer than 10,000 total gene counts were excluded.
2. Reproduce their young and old contrasts and time bins: young 2 h, 24 h, 13–49 d and 132 d; old 24 h and 41–96 d. These paper bins are separate from our preregistered 2 h, 24 h and 13 d genotype-adjusted primary contrasts. Do not merge those designs or call one a reproduction of the other.
3. The caption reports P ≤ 0.05 and absolute log2 fold change > 1. The main methods do not identify whether this P is raw or adjusted, or specify the complete DESeq2 design. Obtain the supplement/code and resolve both before declaring exact agreement. Our primary adjusted P < 0.05 rule remains unchanged.
4. Use the authors' exact gene-to-functional-hub mapping from Gene Analytics / supplementary Table S2. The six broad groups are DNA/RNA, signaling/regulation, mitochondrial/metabolic, contractile/cytoskeletal, extracellular matrix/structural, and immune. Count unique DE genes by group, time and up/down direction. GO or Pfam2GO terms do not automatically define these six paper groups.
5. Compare the underlying gene sets, directions and counts, not just a visually similar bubble plot. Record unmatched gene IDs and denominators. The caption says bubble diameter is proportional to gene count; a new plot should expose numeric counts directly.

## Panel E: predefined immune/cell-repair score

1. Recover the exact supplementary Botryllus gene lists: 112 homologous model genes expand to 13 proimmune and 133 cell-repair Botryllus gene identities because multiple Botryllus genes can map to one model gene. The source model contains 338 mammalian genes, 48 proimmune and 290 cell-repair.
2. Recover the exact expression transform, StepMiner thresholds, cluster assignments, signed weights, standard-deviation convention and score code. The prose formula alone leaves implementation details unresolved. Freeze these before looking at treatment separation; do not train weights on these samples or tune them to match the figure.
3. The control-versus-2 h score uses only the proimmune genes. The long-term comparisons use both proimmune and cell-repair genes, with negative and positive weights respectively. Scores are weighted combinations of normalized cluster averages, not counts of SPLASH anchors or DE k-mers.
4. Match the displayed group memberships and sample counts, including developmental stage and genotype. Reproduce the authors' test and ROC-AUC orientation from their code; do not infer these solely from the plotted P values.
5. Report the paper reproduction and our genotype/collection-adjusted analysis side by side. The old cohort is one genotype, so inference cannot establish an age-wide treatment response across old genotypes.

## Connecting SPLASH and DE-kupl

Map every significant SPLASH anchor and DE k-mer to a supported Botryllus gene or leave it unresolved. Join those genes to gene-level DE and the frozen paper gene sets. Show how many paper genes are recovered and whether abundance directions agree. Keep candidate splice/sequence changes separate: an anchor's significance or a sketch match does not establish differential gene abundance, differential splicing, an ortholog, or a cell-repair effect.

## Recovered inputs and outstanding requirements

Dataset S1 and the supporting appendix have now been retrieved from the Europe PMC supplementary-files service. The workbook contains 72,113 genes × 64 libraries, all six original DESeq2 tables, the 13 proimmune plus 133 cell-repair gene IDs, curated DE gene IDs, functional-group list text and all 84 declared group-count cells. These are exported under figure4/ with the workbook SHA-256 fingerprint.

The companion mammalian exercise-analysis repository is pinned at YoyoVosko/MacrophagesOntheRun commit 4cf2f16fb2191135521bdf716f61c99ed1d3f815. It does not contain a Botryllus-specific implementation. Exact original reference versions, the complete original DESeq2 formula/version, transformed Botryllus expression values and score implementation remain unresolved.

All six supplied-count condition-only refits and three genotype/collection-adjusted refits are completed and retrieved (corrected job 49387733). The adjusted models tested 27,509 / 26,219 / 3,619 genes for young 2 h / 24 h / 312 h, with 4 / 4 / 3 genotypes. None passed global BH-adjusted P < 0.05. Collection strata were balanced and summed within genotype/condition; no calendar-date collection covariate was verified.

The gene view includes 851 identifiable genes from the immune model and functional-group label mappings, including ambiguous membership candidates. The 12 unresolved labels remain separate. Exact gene IDs join each record to supplied-count expression effects and adjusted P values. Full-depth SPLASH/DE-kupl gene recovery job 49373978 remains dependent on unfinished full-depth analysis, behind read trimming job 49368820. Its pending exporter now retains individual anchor and k-mer sequences with mapping ambiguity. No completed sequence recovery or protein-to-paper-gene assignment is imported. A supplied-count redraw is not an independent raw-read reproduction.
