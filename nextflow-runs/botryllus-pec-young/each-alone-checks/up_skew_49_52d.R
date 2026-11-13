# Why do 86% of genes go up at 49-52 d, fitted alone? Refit with each library left out in
# turn, same DESeq2 calls as fits.R (poscounts, ~ treatment), and report the share of genes
# with a positive fold change. Colony 6941c has a treated library (JA151) and no control.
suppressMessages(library(DESeq2))
cts <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(cts) <- cts$gene_id; cts$gene_id <- NULL; cts$gene_name <- NULL
cts <- round(as.matrix(cts))
s <- read.csv("samples.csv"); rownames(s) <- s$sample
s <- s[s$window == "young_49_52d", ]
fit <- function(d, label) {
  d$treatment <- factor(d$treatment, levels=c("control","treated"))
  m <- cts[, d$sample]; m <- m[rowSums(m >= 1) >= 1, ]
  dds <- DESeqDataSetFromMatrix(m, d, ~ treatment)
  dds <- DESeq(dds, test="Wald", fitType="parametric", minReplicatesForReplace=7, sfType="poscounts", quiet=TRUE)
  r <- results(dds, contrast=c("treatment","treated","control"))
  ok <- !is.na(r$pvalue) & r$log2FoldChange != 0
  sf <- paste(sprintf("%s=%.2f", d$vial, sizeFactors(dds)), collapse=" ")
  cat(sprintf("%-14s libs %d tested %d share_up %.3f p<.05 %d | size factors %s\n",
    label, ncol(m), sum(ok), mean(r$log2FoldChange[ok] > 0), sum(r$pvalue < 0.05, na.rm=TRUE), sf))
}
fit(s, "all 5")
for (v in s$vial) fit(s[s$vial != v, ], paste("without", v))
