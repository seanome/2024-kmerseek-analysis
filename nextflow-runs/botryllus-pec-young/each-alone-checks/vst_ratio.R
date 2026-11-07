suppressMessages(library(DESeq2))
cts <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(cts) <- cts$gene_id; cts$gene_id <- NULL; cts$gene_name <- NULL
cts <- round(as.matrix(cts)); s <- read.csv("samples_paper.csv"); rownames(s) <- s$sample
cts <- cts[rowSums(cts[,s$sample]>=1)>=1, s$sample]
s$win <- sub("_(control|treated)$","",s$group)
for (w in c(unique(s$win),"all")) { d <- if (w=="all") s else s[s$win==w,]; d$group <- factor(d$group)
 dds <- DESeq(DESeqDataSetFromMatrix(cts[,d$sample],d,~0+group),sfType="ratio",quiet=TRUE)
 cat(w, ncol(d), "baseMean>5:", sum(rowMeans(counts(dds,normalized=TRUE))>5), tryCatch({vst(dds,blind=TRUE,nsub=1000);"vst ok"},error=function(e) paste("VST FAIL",conditionMessage(e))),"\n") }
