# External genotype-blocked DESeq2 model of native DE-kupl counts.
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(TRUE)
folder <- args[1]; contrast_name <- args[2]; assay <- args[3]
meta <- read.delim(file.path(folder, 'units.tsv'), row.names=1, check.names=FALSE)
meta$genotype <- factor(meta$genotype)
meta$condition <- factor(meta$condition, levels=c('Control','Treatment'))
mm <- model.matrix(~ genotype + condition, meta)
if(qr(mm)$rank != ncol(mm) || nrow(mm) <= ncol(mm)) stop('Unestimable blocked design')
nf <- read.delim(file.path(folder,'normalization.tsv'), row.names=1)
if(!'normalization_factor' %in% names(nf)) stop('Native normalization header differs')
factors <- nf[rownames(meta),'normalization_factor']
if(any(!is.finite(factors)) || any(factors<=0)) stop('Invalid normalization factors')
factors <- factors/exp(mean(log(factors)))
parts <- sort(list.files(folder, pattern='^part-[0-9]+[.]tsv[.]gz$', full.names=TRUE))
outputs <- character(); all_p <- list(); notes <- list()
for(i in seq_along(parts)) {
  dat <- read.delim(gzfile(parts[i]), row.names=1, check.names=FALSE)
  if(!nrow(dat)) next
  if(!identical(colnames(dat),rownames(meta))) stop('Unit order differs')
  dds <- DESeqDataSetFromMatrix(as.matrix(dat), meta, ~ genotype + condition)
  sizeFactors(dds) <- factors
  dds <- estimateDispersionsGeneEst(dds, quiet=TRUE)
  fit_note <- 'parametric dispersion fit'
  dds <- tryCatch(estimateDispersionsMAP(estimateDispersionsFit(dds, quiet=TRUE), quiet=TRUE), error=function(e) {
    if(!grepl('2 orders of magnitude',conditionMessage(e),fixed=TRUE)) stop(e)
    dispersions(dds) <- mcols(dds)$dispGeneEst
    fit_note <<- 'gene-wise dispersion fallback after documented DESeq2 fit refusal'
    dds
  })
  dds <- nbinomWaldTest(dds, quiet=TRUE)
  res <- as.data.frame(results(dds, contrast=c('condition','Treatment','Control'), independentFiltering=FALSE, pAdjustMethod='none'))
  res$tag <- rownames(res); res$dispersion_fit <- fit_note
  path <- file.path(folder,paste0('raw-result-',sprintf('%03d',i),'.rds'))
  saveRDS(res,path); outputs <- c(outputs,path); all_p[[length(all_p)+1]] <- res$pvalue
  notes[[length(notes)+1]] <- data.frame(part=i, features=nrow(res), missing_p=sum(is.na(res$pvalue)), dispersion_fit=fit_note)
}
p <- unlist(all_p, use.names=FALSE)
if(!length(p)) stop('No eligible features; no negative result inferred')
adjusted <- p.adjust(p, method='BH', n=length(p))
out <- gzfile(file.path(folder,'all-features.tsv.gz'),'wt')
position <- 1L
for(i in seq_along(outputs)) {
  res <- readRDS(outputs[i]); n <- nrow(res)
  res$padj <- adjusted[seq.int(position,length.out=n)];position <- position+n
  res$contrast <- contrast_name;res$assay <- assay
  res$status <- ifelse(is.na(res$padj),'not testable',ifelse(res$padj<0.05,'DE detected','no DE detected'))
  write.table(res,out,sep='\t',quote=FALSE,row.names=FALSE,col.names=(i==1))
}
close(out)
write.table(do.call(rbind,notes),file.path(folder,'fit-audit.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
writeLines(c(paste('eligible features',length(p)),paste('missing p-values',sum(is.na(p))),paste('genotype units',nlevels(meta$genotype)),paste('design rank',qr(mm)$rank),paste('residual degrees of freedom',nrow(mm)-ncol(mm)),paste('DESeq2',packageVersion('DESeq2'))),file.path(folder,'model.complete.txt'))
capture.output(sessionInfo(),file=file.path(folder,'R-session.txt'))
