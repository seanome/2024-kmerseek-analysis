# Which design reproduces the paper's own p-values, from the paper's own counts?
suppressMessages(library(DESeq2))
cts <- read.delim("paper_counts.tsv", row.names=1, check.names=FALSE)
cts <- round(as.matrix(cts)); cat("rows with NA:", head(rownames(cts)[rowSums(is.na(cts))>0], 8), "\n"); cts <- cts[complete.cases(cts),]
g <- read.delim("paper_groups.tsv")
res <- list()
for (k in c(5,4,1,2)) {
  s <- g[g$group==k,]; rownames(s) <- s$vial
  s$treat <- factor(s$treat, levels=c("Con","Tre")); s$colony <- factor(s$colony)
  m <- cts[, s$vial]; 
  pap <- read.delim(sprintf("paper_group%d.tsv",k))
  pp <- pap$pvalue[match(rownames(m), pap$Gene_ID)]
  for (des in c("~ treat","~ colony + treat")) {
    r <- tryCatch({
      dds <- DESeqDataSetFromMatrix(m, s, as.formula(des))
      dds <- suppressMessages(DESeq(dds, quiet=TRUE))
      as.data.frame(results(dds, contrast=c("treat","Tre","Con")))
    }, error=function(e) {cat("ERR",k,des,conditionMessage(e),"\n"); NULL})
    if (is.null(r)) next
    ok <- !is.na(r$pvalue) & !is.na(pp)
    cat(sprintf("group %d %-18s genes %d spearman_p %.5f pearson_logp %.5f medabsdiff_logp %.4f n_p05 %d paper_n_p05 %d\n",
      k, des, sum(ok), cor(r$pvalue[ok],pp[ok],method="spearman"),
      cor(-log10(r$pvalue[ok]+1e-300),-log10(pp[ok]+1e-300)),
      median(abs(log10(r$pvalue[ok]+1e-300)-log10(pp[ok]+1e-300))),
      sum(r$pvalue[ok]<0.05), sum(pp[ok]<0.05)))
  }
}
