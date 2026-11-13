# Each comparison fitted on its own libraries, same DESeq2 calls as the pipeline's template.
suppressMessages(library(DESeq2))
cts <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(cts) <- cts$gene_id; cts$gene_id <- NULL; cts$gene_name <- NULL
cts <- round(as.matrix(cts))
s <- read.csv("samples.csv"); rownames(s) <- s$sample
cts <- cts[rowSums(cts>=1)>=1, s$sample]
cat("genes", nrow(cts), "\n")
sets <- split(s, s$window)
paper <- c(JA120=1,JA121=1,JA131=1,JA150=1,JA151=1,JA152=1,JA153=1,JA154=1)
sets[["paper_13_49d"]] <- s[s$vial %in% names(paper),]
for (n in names(sets)) for (des in c("~ treatment","~ colony + treatment")) {
  d <- sets[[n]]; d$treatment <- factor(d$treatment,levels=c("control","treated")); d$colony <- factor(d$colony)
  m <- cts[, d$sample]
  step <- "DESeq"
  out <- tryCatch({
    dds <- DESeqDataSetFromMatrix(m, d, as.formula(des))
    dds <- DESeq(dds, test="Wald", fitType="parametric", minReplicatesForReplace=7, sfType="poscounts", quiet=TRUE)
    r <- results(dds, contrast=c("treatment","treated","control"))
    step <- "vst"
    v <- tryCatch({vst(dds, blind=TRUE, nsub=1000); "vst ok"}, error=function(e) paste("VST FAIL:",conditionMessage(e)))
    sprintf("libs %d residual_df %d tested %d p<.05 %d padj<.05 %d baseMean>5 %d | %s", ncol(m), ncol(m)-qr(model.matrix(as.formula(des),d))$rank,
      sum(!is.na(r$pvalue)), sum(r$pvalue<0.05,na.rm=TRUE), sum(r$padj<0.05,na.rm=TRUE), sum(rowMeans(counts(dds,normalized=TRUE))>5), v)
  }, error=function(e) paste("FAIL at",step,":",conditionMessage(e)))
  cat(sprintf("%-16s %-22s %s\n", n, des, out))
}
