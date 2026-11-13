# Write each Affymetrix array's 25-letter probes as probe_set, sequence, from Bioconductor's packages.
# Usage: Rscript export_probe_tables.R <out folder>
# GEO's platform tables for these arrays hold no probe sequences, so these packages are the source.
suppressMessages({
  library(DBI)
  library(Biostrings)
  library(hgu133plus2probe)
  library(hgu219probe)
  library(pd.hta.2.0)
})
out <- commandArgs(trailingOnly = TRUE)[1]

write_table <- function(probe_set, sequence, platform) {
  path <- gzfile(file.path(out, paste0(platform, ".probes.tsv.gz")), "w")
  write.table(data.frame(probe_set = probe_set, sequence = sequence), path, sep = "\t", quote = FALSE, row.names = FALSE)
  close(path)
}

plus2 <- as.data.frame(hgu133plus2probe)
write_table(plus2$Probe.Set.Name, plus2$sequence, "GPL570")
u219 <- as.data.frame(hgu219probe)
write_table(u219$Probe.Set.Name, u219$sequence, "GPL13667")

connection <- db(pd.hta.2.0)
# core_mps links each probe's feature set to the transcript cluster the series reports a value for
sets <- dbGetQuery(
  connection,
  "SELECT pmfeature.fid AS fid, core_mps.meta_fsetid AS man_fsetid FROM pmfeature JOIN core_mps USING (fsetid)"
)
data(pmSequence, package = "pd.hta.2.0", envir = environment())
sequences <- as.character(pmSequence$sequence)
names(sequences) <- pmSequence$fid
path <- gzfile(file.path(out, "GPL17586.raw.tsv.gz"), "w")
write.table(data.frame(sets, sequence = sequences[as.character(sets$fid)]), path, sep = "\t", quote = FALSE, row.names = FALSE)
close(path)
