# Refit the supplied HTSeq counts; retain the published tables as a separate source.
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(TRUE); input <- args[1]; out <- args[2]; design_file <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
counts <- read.delim(gzfile(file.path(input,'published-counts.tsv.gz')),row.names=1,check.names=FALSE)
stopifnot(!anyDuplicated(rownames(counts)),ncol(counts)==64,all(as.matrix(counts)>=0))
groups <- read.delim(file.path(input,'published-groups.tsv'),check.names=FALSE)
sets <- read.delim(file.path(input,'immune-gene-sets.tsv'))
summary <- list()
fit_model <- function(mat, meta, formula, name, scope) {
 stopifnot(identical(colnames(mat),rownames(meta)))
 mm <- model.matrix(formula,meta)
 if(qr(mm)$rank!=ncol(mm) || nrow(mm)<=ncol(mm)) stop('Unestimable model')
 keep <- rowSums(mat)>=1
 result_file <- file.path(out,paste0(name,'.tsv.gz'))
 if(file.exists(result_file)) {
  r <- read.delim(gzfile(result_file),check.names=FALSE)
  if(!setequal(r$gene_id,rownames(mat)[keep]) || !all(r$scope==scope)) stop('Existing result scope differs')
 } else {
 dds <- DESeqDataSetFromMatrix(round(mat[keep,,drop=FALSE]),meta,formula)
 dds <- DESeq(dds,quiet=TRUE,parallel=FALSE)
 r <- as.data.frame(results(dds,contrast=c('condition','Treatment','Control'),alpha=.05))
 r$gene_id <- rownames(r);r$scope <- scope
 # The adjusted primary comparisons use the predeclared filter and global BH with no independent filtering.
 if(scope=='genotype_collection_adjusted') {
  r <- as.data.frame(results(dds,contrast=c('condition','Treatment','Control'),independentFiltering=FALSE,pAdjustMethod='none'))
  r$padj <- p.adjust(r$pvalue,method='BH',n=nrow(r))
  r$gene_id <- rownames(r);r$scope <- scope
 }
 }
 r$immune_role <- sets$role[match(r$gene_id,sets$gene_id)]
 write.table(r,gzfile(file.path(out,paste0(name,'.tsv.gz'))),sep='\t',quote=FALSE,row.names=FALSE)
 summary[[name]] <<- data.frame(contrast=name,scope=scope,libraries_or_units=ncol(mat),design=paste(deparse(formula),collapse=''),eligible_genes=nrow(r),raw_p_le005_abs_lfc_gt1=sum(r$pvalue<=.05 & abs(r$log2FoldChange)>1,na.rm=TRUE),padj_le005_abs_lfc_gt1=sum(r$padj<=.05 & abs(r$log2FoldChange)>1,na.rm=TRUE),padj_lt005=sum(r$padj<.05,na.rm=TRUE),missing_p=sum(is.na(r$pvalue)))
 return(invisible(NULL))
}
for(g in unique(groups$group)) {
 sub <- groups[groups$group==g,,drop=FALSE];runs <- sub$run
 if(any(colSums(counts[,runs,drop=FALSE])<10000)) stop('Published comparison includes a below-10000-count library')
 meta <- data.frame(condition=ifelse(grepl('PaceMakerCon_',sub$sample),'Control','Treatment'),row.names=runs)
 meta$condition <- factor(meta$condition,levels=c('Control','Treatment'))
 name <- paste0('paper_group',sub('Group ([0-9]+).*','\\1',g))
 fit_model(as.matrix(counts[,runs,drop=FALSE]),meta,~condition,name,'published_groups_condition_only_refit')
}
design <- read.delim(design_file,check.names=FALSE)
for(contrast in unique(design$contrast)) {
 libs <- design[design$contrast==contrast,,drop=FALSE]
 gs <- unique(libs$genotype)
 unit <- expand.grid(condition=c('Control','Treatment'),genotype=gs,stringsAsFactors=FALSE)
 unit$unit <- paste(unit$genotype,unit$condition,sep='_');rownames(unit)<-unit$unit
 pooled <- matrix(0,nrow=nrow(counts),ncol=nrow(unit),dimnames=list(rownames(counts),unit$unit))
 for(i in seq_len(nrow(libs))) {
  u<-paste(libs$genotype[i],libs$condition[i],sep='_');pooled[,u]<-pooled[,u]+counts[,libs$run[i]]
 }
 keep <- rowSums(pooled>=5)>=2
 unit$genotype<-factor(unit$genotype);unit$condition<-factor(unit$condition,levels=c('Control','Treatment'))
 fit_model(pooled[keep,,drop=FALSE],unit,~genotype+condition,contrast,'genotype_collection_adjusted')
 write.table(unit,file.path(out,paste0(contrast,'-units.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
}
write.table(do.call(rbind,summary),file.path(out,'model-summary.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'R-session.txt'))
writeLines('Completed supplied-count refits. Not raw-read reproduction; Figure 4E original expression transform/code unresolved.',file.path(out,'complete.txt'))
