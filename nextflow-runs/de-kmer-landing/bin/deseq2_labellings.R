#!/usr/bin/env Rscript
# DESeq2 on one count table, once per labelling of the 6 samples.
# Same model for both paths: ~ condition, B vs A, default settings
# (notebooks/360_data/preregistered.md).
#
#   deseq2_labellings.R counts.tsv labellings.tsv out_prefix threads
#
# counts.tsv: first column the feature (a transcript or a k-mer), then sample_01..sample_06.
# Writes <out_prefix>.<labelling>.deseq2.tsv.gz with feature, baseMean, log2FoldChange,
# pvalue, padj, and <out_prefix>.deseq2_version.txt.
suppressPackageStartupMessages({
  library(DESeq2)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
counts <- read.delim(args[1], check.names = FALSE, stringsAsFactors = FALSE)
labellings <- read.delim(args[2], check.names = FALSE, stringsAsFactors = FALSE)
prefix <- args[3]
register(MulticoreParam(as.integer(args[4])))

feature <- counts[[1]]
m <- as.matrix(round(counts[, sprintf("sample_%02d", 1:6)]))
storage.mode(m) <- "integer"
rownames(m) <- feature

for (i in seq_len(nrow(labellings))) {
  lab <- labellings$labelling[i]
  cond <- factor(unlist(labellings[i, sprintf("sample_%02d", 1:6)]), levels = c("A", "B"))
  dds <- DESeqDataSetFromMatrix(m, data.frame(condition = cond, row.names = colnames(m)), ~condition)
  dds <- DESeq(dds, quiet = TRUE, parallel = TRUE)
  res <- results(dds, contrast = c("condition", "B", "A"), parallel = TRUE)
  out <- data.frame(feature = rownames(res), baseMean = res$baseMean,
                    log2FoldChange = res$log2FoldChange, pvalue = res$pvalue, padj = res$padj)
  con <- gzfile(sprintf("%s.%s.deseq2.tsv.gz", prefix, lab), "w")
  write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
  message(sprintf("%s: %d features, %d with padj < 0.05", lab, nrow(out), sum(out$padj < 0.05, na.rm = TRUE)))
}
writeLines(c(paste("DESeq2", as.character(packageVersion("DESeq2"))), R.version.string),
           sprintf("%s.deseq2_version.txt", prefix))
