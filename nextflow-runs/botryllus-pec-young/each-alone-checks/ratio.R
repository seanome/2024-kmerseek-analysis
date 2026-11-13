suppressMessages(library(DESeq2))
cts <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(cts) <- cts$gene_id; cts$gene_id <- NULL; cts$gene_name <- NULL
cts <- round(as.matrix(cts)); s <- read.csv("samples.csv"); g <- read.delim("paper_groups.tsv")
s <- merge(s, g[,c("vial","group")], by="vial", suffixes=c("","_p")); rownames(s) <- s$sample
cts <- cts[rowSums(cts[,s$sample]>=1)>=1, s$sample]
cat("genes in 38 libs",nrow(cts),"counted in every lib",sum(rowSums(cts>0)==ncol(cts)),"\n")
for (k in list(1,2,4,5,c(1,2,4,5))) for (sf in c("ratio","poscounts")) {
 d <- s[s$group_p %in% k,]; d$treatment <- factor(d$treatment); m <- cts[,d$sample]
 x <- tryCatch({dds <- DESeq(DESeqDataSetFromMatrix(m,d,~treatment),sfType=sf,quiet=TRUE); r <- results(dds,contrast=c("treatment","treated","control"))
   sprintf("tested %d p<.05 %d padj<.05 %d", sum(!is.na(r$pvalue)),sum(r$pvalue<.05,na.rm=TRUE),sum(r$padj<.05,na.rm=TRUE))},error=function(e) paste("FAIL",conditionMessage(e)))
 cat(paste(k,collapse=","),sf,x,"\n") }
