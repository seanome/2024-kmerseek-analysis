# Do size factors cause the gap between the paper-style refit and the paper's gene lists?
# Each of the paper's four comparisons is fitted with ~ treatment on its own libraries, on
# this run's counts and on the paper's counts, and only the size factors change:
#   default    DESeq2's default (median of ratios over genes counted in every library)
#   poscounts  DESeq2's poscounts
#   total      each library's summed gene counts, divided by their geometric mean
#   swapped    the default size factors of the OTHER data set, same libraries
# The pipeline's DESeq2 step also corrects this run's counts for each gene's length in each
# library (salmon.merged.gene_lengths.tsv, as tximport does), so the fits on this run's
# counts do the same, and default_no_gene_lengths shows what that correction changes. The
# paper's counts have no such table.
# Inputs in the current folder: salmon.merged.gene_counts.tsv, salmon.merged.gene_lengths.tsv, samples_paper.csv,
# paper_counts.tsv, paper_groups.tsv, paper_group<k>.tsv (export_paper_counts.py),
# paper_gene_overlap_genes.tsv (from ../each-alone-comparison/paper/), and pipeline/ holding
# the refit's *.deseq2.results.tsv (differentialabundance-paper/tables/differential/rnaseq/).
# Writes size_factors_fits.tsv, size_factors_values.tsv and size_factors_genes.tsv.gz.
suppressMessages(library(DESeq2))
GROUPS <- c("1"="young_2h", "2"="young_24h", "4"="young_13_49d", "5"="young_132d")
LABEL <- c("1"="2 h", "2"="24 h", "4"="13-49 d", "5"="132 d")

ours <- read.delim("salmon.merged.gene_counts.tsv", check.names=FALSE)
rownames(ours) <- ours$gene_id; ours$gene_id <- NULL; ours$gene_name <- NULL
ours <- round(as.matrix(ours))
s <- read.csv("samples_paper.csv")
ours <- ours[, s$sample]; colnames(ours) <- s$vial
# The pipeline drops genes with no count of 1 or more in any of the 38 libraries.
ours <- ours[rowSums(ours >= 1) >= 1, ]
len <- read.delim("salmon.merged.gene_lengths.tsv", check.names=FALSE)
rownames(len) <- len$gene_id; len <- as.matrix(len[rownames(ours), s$sample]); colnames(len) <- s$vial
stopifnot(all(is.finite(len)), all(len > 0))
s$comparison <- sub("_(control|treated)$", "", s$group)

paper <- read.delim("paper_counts.tsv", row.names=1, check.names=FALSE)
paper <- round(as.matrix(paper)); paper <- paper[complete.cases(paper), ]
# The paper's table ends with six rows that are not genes: HTSeq's tallies of reads it did not
# count (__no_feature, __ambiguous, __too_low_aQual, __not_aligned, __alignment_not_unique)
# and a Sum row. They are dropped here. The first two jobs (51909796, 51911916) kept them.
not_gene <- !grepl("^g[0-9]+$", rownames(paper))
cat("rows of the paper's table that are not genes, dropped:", rownames(paper)[not_gene], "\n")
paper <- paper[!not_gene, ]
g <- read.delim("paper_groups.tsv")
stopifnot(setequal(g$vial, s$vial), setequal(colnames(paper), s$vial))
cat("genes: this run", nrow(ours), "paper", nrow(paper), "libraries", nrow(s), "\n")

listed <- read.delim("paper_gene_overlap_genes.tsv")
listed$fun_gene[tolower(as.character(listed$matched)) != "true"] <- NA
cat("listed entries", nrow(listed), "with a matched gene in this run", sum(!is.na(listed$fun_gene)), "\n")

total_factors <- function(m) { t <- colSums(m); t / exp(mean(log(t))) }
# Per-gene, per-library length correction, centred per gene, as DESeq2 does with avgTxLength.
centred <- function(x) x / exp(rowMeans(log(x)))
library_factors <- function(m, L=NULL, type="ratio") estimateSizeFactorsForMatrix(if (is.null(L)) m else m / centred(L), type=type)

fit <- function(m, d, sf_type, sf=NULL, L=NULL) {
  dds <- DESeqDataSetFromMatrix(m, d, ~ treatment)
  type <- if (sf_type == "poscounts") "poscounts" else "ratio"
  if (is.null(sf)) {
    if (!is.null(L)) assays(dds)[["avgTxLength"]] <- L
    dds <- estimateSizeFactors(dds, type=type)
    sf <- library_factors(m, L, type)
  } else if (is.null(L)) {
    sizeFactors(dds) <- sf[colnames(m)]
  } else {
    # The given library factors times the gene-length correction, centred as DESeq2 does.
    normalizationFactors(dds) <- centred(t(t(centred(L)) * sf[colnames(m)]))
  }
  # Same calls as the pipeline's DESeq2 step; DESeq() keeps factors already set.
  dds <- DESeq(dds, test="Wald", fitType="parametric", minReplicatesForReplace=7, quiet=TRUE)
  r <- as.data.frame(results(dds, contrast=c("treatment", "treated", "control")))
  list(r=r, sf=sf[colnames(m)])
}

fits <- list(); values <- list(); genes <- list()
for (k in names(GROUPS)) {
  d <- s[s$comparison == GROUPS[k], ]; rownames(d) <- d$vial
  stopifnot(setequal(d$vial, g$vial[g$group == as.integer(k)]))
  d$treatment <- factor(d$treatment, levels=c("control", "treated"))
  data <- list(this_run=ours[, d$vial], paper=paper[, d$vial])
  lens <- list(this_run=len[, d$vial], paper=NULL)
  base <- list(this_run=library_factors(data$this_run, lens$this_run), paper=library_factors(data$paper))
  lk <- listed[listed$comparison == LABEL[k], ]
  for (counts in names(data)) {
    m <- data[[counts]]
    other <- setdiff(names(data), counts)
    kinds <- c("default", "poscounts", "total", "swapped", if (counts == "this_run") "default_no_gene_lengths")
    for (sf_type in kinds) {
      sf <- switch(sf_type, total=total_factors(m), swapped=base[[other]], NULL)
      L <- if (sf_type == "default_no_gene_lengths") NULL else lens[[counts]]
      f <- suppressMessages(fit(m, d, sf_type, sf, L)); r <- f$r
      id <- if (counts == "paper") lk$paper_gene else lk$fun_gene
      lr <- r[match(id, rownames(r)), ]
      tested <- !is.na(id) & !is.na(lr$pvalue)
      same <- ifelse(lk$direction == "up", lr$log2FoldChange > 0, lr$log2FoldChange < 0)
      hit <- !is.na(r$pvalue) & r$pvalue <= 0.05 & abs(r$log2FoldChange) > 1
      row <- data.frame(counts=counts, comparison=LABEL[k], size_factors=sf_type, libraries=ncol(m),
        genes_counted_in_every_library=sum(rowSums(m > 0) == ncol(m)),
        largest_over_smallest_size_factor=round(max(f$sf) / min(f$sf), 1),
        tested=sum(!is.na(r$pvalue)), p_lt_0_05=sum(r$pvalue < 0.05, na.rm=TRUE),
        padj_lt_0_05=sum(r$padj < 0.05, na.rm=TRUE),
        p_le_0_05_2fold_up=sum(hit & r$log2FoldChange > 0), p_le_0_05_2fold_down=sum(hit & r$log2FoldChange < 0),
        share_up=round(mean(r$log2FoldChange[!is.na(r$pvalue) & r$log2FoldChange != 0] > 0), 3),
        paper_listed=nrow(lk), listed_tested=sum(tested),
        listed_p_lt_0_05_same_direction=sum(tested & lr$pvalue < 0.05 & same),
        listed_same_sign=sum(tested & same))
      # The two fits that must give back known results.
      row$check <- NA
      if (sf_type == "default" && counts == "paper") {
        pap <- read.delim(sprintf("paper_group%s.tsv", k))
        pp <- pap$pvalue[match(rownames(r), pap$Gene_ID)]; ok <- !is.na(pp) & !is.na(r$pvalue)
        row$check <- sprintf("spearman with paper p %.5f over %d genes", cor(r$pvalue[ok], pp[ok], method="spearman"), sum(ok))
      }
      if (sf_type == "default" && counts == "this_run") {
        f1 <- Sys.glob(sprintf("pipeline/%s_treated_vs_control*.deseq2.results.tsv", GROUPS[k]))
        pl <- read.delim(f1[1]); pp <- pl$pvalue[match(rownames(r), pl$gene_id)]
        ok <- !is.na(pp) & !is.na(r$pvalue)
        row$check <- sprintf("largest p difference from pipeline %.2g, spearman %.5f over %d genes, tested there %d",
          max(abs(pp[ok] - r$pvalue[ok])), cor(pp[ok], r$pvalue[ok], method="spearman"), sum(ok), sum(!is.na(pl$pvalue)))
      }
      fits[[length(fits) + 1]] <- row
      values[[length(values) + 1]] <- data.frame(counts=counts, comparison=LABEL[k], size_factors=sf_type,
        vial=names(f$sf), treatment=d[names(f$sf), "treatment"], library_total=colSums(m)[names(f$sf)], size_factor=signif(f$sf, 5))
      genes[[length(genes) + 1]] <- data.frame(counts=counts, comparison=LABEL[k], size_factors=sf_type, gene=rownames(r),
        baseMean=signif(r$baseMean, 5), log2FoldChange=signif(r$log2FoldChange, 5), pvalue=signif(r$pvalue, 5), padj=signif(r$padj, 5))
      cat(sprintf("%-8s %-7s %-23s tested %5d p<.05 %4d 2-fold up %4d down %4d | listed %3d tested %3d found %3d | %s\n",
        counts, LABEL[k], sf_type, row$tested, row$p_lt_0_05, row$p_le_0_05_2fold_up, row$p_le_0_05_2fold_down,
        row$paper_listed, row$listed_tested, row$listed_p_lt_0_05_same_direction, ifelse(is.na(row$check), "", row$check)))
    }
  }
}
fits <- do.call(rbind, fits)
write.table(fits, "size_factors_fits.tsv", sep="\t", quote=FALSE, row.names=FALSE)
write.table(do.call(rbind, values), "size_factors_values.tsv", sep="\t", quote=FALSE, row.names=FALSE)
gz <- gzfile("size_factors_genes.tsv.gz", "w")
write.table(do.call(rbind, genes), gz, sep="\t", quote=FALSE, row.names=FALSE); close(gz)
cat("\nlisted genes found, all four comparisons (of", nrow(listed), "entries)\n")
print(aggregate(listed_p_lt_0_05_same_direction ~ counts + size_factors, fits, sum))
cat("DESeq2", as.character(packageVersion("DESeq2")), R.version.string, "\n")
