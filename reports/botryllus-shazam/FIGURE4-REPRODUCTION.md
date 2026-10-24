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

## Outstanding requirements

Exact Table S2 / Gene Analytics hub assignments, immune gene identities and scoring implementation, exact reference versions, sample inclusion list and original DESeq2 design are not yet recovered here. Full-depth counts and adjusted results remain pending. Kallisto-derived gene counts in our current abundance branch provide an independent analysis; they do not exactly reproduce STAR plus HTSeq.
