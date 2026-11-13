# Smallest window: does vst work with a lower nsub? Also the all-52 fit the report uses.
suppressMessages(library(DESeq2))
cts <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(cts) <- cts$gene_id; cts$gene_id <- NULL; cts$gene_name <- NULL
cts <- round(as.matrix(cts))
s <- read.csv("samples.csv"); rownames(s) <- s$sample
cts <- cts[rowSums(cts>=1)>=1, s$sample]
d <- s[s$window=="young_21_28d",]; d$treatment <- factor(d$treatment); d$colony <- factor(d$colony)
dds <- DESeq(DESeqDataSetFromMatrix(cts[,d$sample], d, ~ colony + treatment), sfType="poscounts", quiet=TRUE)
for (n in c(10,12,14)) { r <- tryCatch({vst(dds, blind=TRUE, nsub=n); "ok"}, error=function(e) conditionMessage(e)); cat("21-28d nsub",n,r,"\n") }
r <- tryCatch({x <- varianceStabilizingTransformation(dds, blind=TRUE); "direct ok"}, error=function(e) conditionMessage(e)); cat("21-28d direct",r,"\n")
r <- tryCatch({x <- rlog(dds, blind=TRUE, fitType="parametric"); "rlog ok"}, error=function(e) conditionMessage(e)); cat("21-28d rlog",r,"\n")
