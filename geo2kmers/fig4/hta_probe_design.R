# Write what each HTA 2.0 probe measures: its probe set, transcript cluster, exon or junction, and position.
# Usage: Rscript hta_probe_design.R <out.tsv.gz>
# One probe can belong to several probe sets, so it can have several rows. Sequences are as in pmSequence.
suppressMessages({
  library(DBI)
  library(Biostrings)
  library(pd.hta.2.0)
})
out <- commandArgs(trailingOnly = TRUE)[1]
connection <- db(pd.hta.2.0)
sets <- dbGetQuery(
  connection,
  paste(
    "SELECT pmfeature.fid AS fid, featureSet.man_fsetid AS probe_set, core_mps.meta_fsetid AS transcript_cluster,",
    "featureSet.exon_id AS exon_id, featureSet.chrom AS chrom, featureSet.strand AS strand,",
    "featureSet.junction_start_edge AS junction_start, featureSet.junction_stop_edge AS junction_stop",
    "FROM pmfeature JOIN featureSet USING (fsetid) LEFT JOIN core_mps USING (fsetid)"
  )
)
data(pmSequence, package = "pd.hta.2.0", envir = environment())
sequences <- as.character(pmSequence$sequence)
names(sequences) <- pmSequence$fid
sets$sequence <- sequences[as.character(sets$fid)]
path <- gzfile(out, "w")
write.table(sets, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
close(path)
