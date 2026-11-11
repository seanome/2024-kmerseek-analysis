# Which part of the counts causes the gap with the paper's gene lists?
# Every count table in the current folder is fitted the same way: each of the paper's four
# comparisons alone, ~ treatment, DESeq2's default size factors, the paper's 38 libraries.
# Tables (written by count_split.nf, named <genome>__<counter>.tsv):
#   paper_genome__*  this run's reads aligned to the paper's genome, counted on its g genes
#   this_genome__*   this run's reads aligned to the April 2024 genome, counted on FUN genes
#   *__salmon        Salmon's counts from nf-core/rnaseq, with <genome>__salmon_lengths.tsv
#   *__featurecounts_s0, _s1, _s2   read pairs that overlap one gene only and align to one
#                    place, as HTSeq counts them; strand ignored (s0), read 1 on the gene's
#                    strand (s1), read 1 on the opposite strand (s2)
# plus the paper's own table (paper_counts.tsv from export_paper_counts.py), fitted first.
# Other inputs: samples_paper.csv, paper_groups.tsv, paper_gene_overlap_genes.tsv (from
# ../each-alone-comparison/paper/). A table that is not in the folder is left out.
# Writes count_split_fits.tsv, count_split_listed.tsv and count_split_libraries.tsv.
suppressMessages(library(DESeq2))
GROUPS <- c("1"="young_2h", "2"="young_24h", "4"="young_13_49d", "5"="young_132d")
LABEL <- c("1"="2 h", "2"="24 h", "4"="13-49 d", "5"="132 d")

s <- read.csv("samples_paper.csv")
s$comparison <- sub("_(control|treated)$", "", s$group)
g <- read.delim("paper_groups.tsv")
stopifnot(setequal(g$vial, s$vial))
listed <- read.delim("paper_gene_overlap_genes.tsv")
listed$fun_gene[tolower(as.character(listed$matched)) != "true"] <- NA

# One matrix per table, genes x 38 vials, plus Salmon's gene lengths where there are any.
by_vial <- function(m) {
  colnames(m) <- sub("\\..*$", "", basename(colnames(m)))
  stopifnot(all(s$sample %in% colnames(m)))
  m <- m[, s$sample, drop=FALSE]; colnames(m) <- s$vial
  m
}
read_salmon <- function(f) {
  x <- read.delim(f, check.names=FALSE); rownames(x) <- x$gene_id
  by_vial(as.matrix(x[, setdiff(colnames(x), c("gene_id", "gene_name"))]))
}
read_featurecounts <- function(f) {
  x <- read.delim(f, comment.char="#", check.names=FALSE); rownames(x) <- x$Geneid
  by_vial(as.matrix(x[, setdiff(colnames(x), c("Geneid", "Chr", "Start", "End", "Strand", "Length"))]))
}
paper <- read.delim("paper_counts.tsv", row.names=1, check.names=FALSE)
paper <- round(as.matrix(paper)); paper <- paper[complete.cases(paper), ]
# The paper's table ends with six rows that are not genes (HTSeq's tallies and a Sum row).
paper <- paper[grepl("^g[0-9]+$", rownames(paper)), s$vial]
tables <- list(paper_table=list(counts=paper, lengths=NULL, ids="paper_gene"))
for (f in sort(Sys.glob("*__*.tsv"))) {
  name <- sub("\\.tsv$", "", f)
  if (grepl("_lengths$", name)) next
  salmon <- grepl("__salmon$", name)
  m <- if (salmon) read_salmon(f) else read_featurecounts(f)
  L <- if (salmon) read_salmon(sub("\\.tsv$", "_lengths.tsv", f)) else NULL
  tables[[name]] <- list(counts=round(m), lengths=L,
    ids=if (grepl("^paper_genome", name)) "paper_gene" else "fun_gene")
}
cat("tables:", names(tables), "\n")

fits <- list(); entries <- list(); libs <- list()
for (name in names(tables)) {
  tab <- tables[[name]]
  # The pipeline drops genes with no count of 1 or more in any of the 38 libraries.
  keep <- rowSums(tab$counts >= 1) >= 1
  m_all <- tab$counts[keep, ]
  on_paper_genes <- tab$ids == "paper_gene" && name != "paper_table"
  if (on_paper_genes) stopifnot(mean(rownames(tab$counts) %in% rownames(paper)) > 0.99)
  for (v in s$vial) {
    # How close is this table to the paper's, library by library, over the paper's genes?
    rho <- NA
    if (on_paper_genes) {
      both <- intersect(rownames(tab$counts), rownames(paper))
      rho <- cor(tab$counts[both, v], paper[both, v], method="spearman")
    }
    libs[[length(libs) + 1]] <- data.frame(table=name, vial=v, counted=sum(tab$counts[, v]),
      paper_counted=sum(paper[, v]), spearman_with_paper_table=round(rho, 4))
  }
  for (k in names(GROUPS)) {
    d <- s[s$comparison == GROUPS[k], ]; rownames(d) <- d$vial
    d$treatment <- factor(d$treatment, levels=c("control", "treated"))
    m <- m_all[, d$vial]
    dds <- DESeqDataSetFromMatrix(m, d, ~ treatment)
    if (!is.null(tab$lengths)) assays(dds)[["avgTxLength"]] <- tab$lengths[rownames(m), d$vial]
    # Same calls as the pipeline's DESeq2 step.
    dds <- suppressMessages(DESeq(dds, test="Wald", fitType="parametric", minReplicatesForReplace=7, quiet=TRUE))
    r <- as.data.frame(results(dds, contrast=c("treatment", "treated", "control")))
    lk <- listed[listed$comparison == LABEL[k], ]
    id <- lk[[tab$ids]]
    lr <- r[match(id, rownames(r)), ]
    tested <- !is.na(id) & !is.na(lr$pvalue)
    same <- ifelse(lk$direction == "up", lr$log2FoldChange > 0, lr$log2FoldChange < 0)
    found <- tested & lr$pvalue < 0.05 & same
    hit <- !is.na(r$pvalue) & r$pvalue <= 0.05 & abs(r$log2FoldChange) > 1
    row <- data.frame(table=name, comparison=LABEL[k], libraries=ncol(m), genes=nrow(m),
      genes_counted_in_every_library=sum(rowSums(m > 0) == ncol(m)), counted=sum(m),
      tested=sum(!is.na(r$pvalue)), p_lt_0_05=sum(r$pvalue < 0.05, na.rm=TRUE),
      p_le_0_05_2fold_up=sum(hit & r$log2FoldChange > 0), p_le_0_05_2fold_down=sum(hit & r$log2FoldChange < 0),
      paper_listed=nrow(lk), listed_with_a_gene=sum(!is.na(id)), listed_tested=sum(tested),
      listed_p_lt_0_05_same_direction=sum(found), listed_same_sign=sum(tested & same), row.names=NULL)
    fits[[length(fits) + 1]] <- row
    entries[[length(entries) + 1]] <- data.frame(table=name, comparison=LABEL[k], paper_gene=lk$paper_gene,
      direction=lk$direction, gene=id, tested=tested, log2FoldChange=signif(lr$log2FoldChange, 4),
      pvalue=signif(lr$pvalue, 4), found=found, row.names=NULL)
    cat(sprintf("%-32s %-7s genes %5d counted %10.0f tested %5d p<.05 %4d 2-fold up %4d down %4d | listed %3d with a gene %3d tested %3d found %3d\n",
      name, LABEL[k], row$genes, row$counted, row$tested, row$p_lt_0_05, row$p_le_0_05_2fold_up, row$p_le_0_05_2fold_down,
      row$paper_listed, row$listed_with_a_gene, row$listed_tested, row$listed_p_lt_0_05_same_direction))
  }
}
fits <- do.call(rbind, fits); libs <- do.call(rbind, libs)
write.table(fits, "count_split_fits.tsv", sep="\t", quote=FALSE, row.names=FALSE)
write.table(do.call(rbind, entries), "count_split_listed.tsv", sep="\t", quote=FALSE, row.names=FALSE)
write.table(libs, "count_split_libraries.tsv", sep="\t", quote=FALSE, row.names=FALSE)
cat("\nlisted genes found, all four comparisons (of", nrow(listed), "entries)\n")
print(aggregate(listed_p_lt_0_05_same_direction ~ table, fits, sum))
cat("\nread pairs counted, as a share of the paper's table, and rank correlation with it per library (median over 38)\n")
libs$share <- libs$counted / libs$paper_counted
print(aggregate(cbind(share, spearman_with_paper_table) ~ table, libs, median, na.action=na.pass))
cat("DESeq2", as.character(packageVersion("DESeq2")), R.version.string, "\n")
