# Write one intensity per probe and sample from Affymetrix CEL files, so probes of one gene keep their own values.
# Usage: Rscript affymetrix_probe_intensities.R <folder of CEL files> <out.tsv.gz>
# Each array loses its 5th percentile as background, as geo2kmers does, and arrays are then quantile normalised,
# on the linear scale. This is done here because oligo's routines stop with a thread error in the conda build.
suppressMessages(library(oligo))
arguments <- commandArgs(trailingOnly = TRUE)
files <- list.files(arguments[1], pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)
raw <- read.celfiles(files)
values <- pm(raw)
values <- apply(values, 2, function(x) pmax(x - quantile(x, 0.05), 0))
reference <- rowMeans(apply(values, 2, sort))
values <- apply(values, 2, function(x) reference[rank(x, ties.method = "first")])
colnames(values) <- sub("^(GSM[0-9]+).*", "\\1", basename(files))
fids <- pmindex(raw)
data(pmSequence, package = annotation(raw), envir = environment())
sequences <- as.character(pmSequence$sequence)
names(sequences) <- pmSequence$fid
table <- data.frame(sequence = sequences[as.character(fids)], values, check.names = FALSE)
table <- table[!is.na(table$sequence), ]
path <- gzfile(arguments[2], "w")
write.table(table, path, sep = "\t", quote = FALSE, row.names = FALSE)
close(path)
