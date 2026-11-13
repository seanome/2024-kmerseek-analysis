# Compare quantifiers with one gene universe, design and DESeq2 implementation.
suppressPackageStartupMessages(library(DESeq2))
a <- commandArgs(trailingOnly=TRUE)
stopifnot(length(a)==4)
read_counts <- function(p) {
  d <- read.delim(p, row.names=1, check.names=FALSE)
  m <- round(as.matrix(d))
  if (anyDuplicated(rownames(m)) || any(!is.finite(m)) || any(m<0)) stop('Invalid counts')
  storage.mode(m) <- 'integer'
  m
}
k <- read_counts(a[1]); s <- read_counts(a[2])
meta <- read.delim(a[3], row.names=1)
stopifnot(setequal(colnames(k),rownames(meta)),setequal(colnames(s),rownames(meta)))
k <- k[,rownames(meta),drop=FALSE]; s <- s[,rownames(meta),drop=FALSE]
genes <- intersect(rownames(k),rownames(s))
keep <- function(x) rowSums(x>=5)>=3 & rowSums(x)>=30
genes <- genes[keep(k[genes,,drop=FALSE]) & keep(s[genes,,drop=FALSE])]
if (!length(genes)) stop('No jointly testable genes')
meta$condition <- factor(meta$condition,levels=c('wild','breed'))
stopifnot(all(table(meta$condition)==3))
dir.create(a[4],recursive=TRUE,showWarnings=FALSE)
fit <- function(counts,name) {
  d <- DESeqDataSetFromMatrix(counts[genes,,drop=FALSE],meta,~condition)
  d <- DESeq(d,quiet=TRUE)
  r <- as.data.frame(results(d,contrast=c('condition','breed','wild'),independentFiltering=FALSE))
  r$padj <- p.adjust(replace(r$pvalue,is.na(r$pvalue),1),'BH')
  r$gene_id <- rownames(r)
  write.table(r,file.path(a[4],paste0(name,'-DESeq2.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
  r
}
rk <- fit(k,'kallisto'); rs <- fit(s,'salmon')
joined <- merge(rk,rs,by='gene_id',suffixes=c('_kallisto','_salmon'))
joined$DE_kallisto <- joined$padj_kallisto<.05
joined$DE_salmon <- joined$padj_salmon<.05
joined$direction_agrees <- sign(joined$log2FoldChange_kallisto)==sign(joined$log2FoldChange_salmon)
write.table(joined,file.path(a[4],'gene-comparison.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
summary <- data.frame(common_tested=length(genes),kallisto_DE=sum(joined$DE_kallisto),
 salmon_DE=sum(joined$DE_salmon),both_DE=sum(joined$DE_kallisto & joined$DE_salmon),
 opposite_DE=sum(joined$DE_kallisto & joined$DE_salmon & !joined$direction_agrees,na.rm=TRUE),
 lfc_spearman=cor(joined$log2FoldChange_kallisto,joined$log2FoldChange_salmon,method='spearman',use='complete.obs'))
write.table(summary,file.path(a[4],'summary.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
capture.output(sessionInfo(),file=file.path(a[4],'R-session.txt'))
