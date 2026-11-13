#!/usr/bin/env Rscript
# Simulate 3 vs 3 paired-end RNA-seq samples from the answer key's transcripts with polyester.
# Settings are notebooks/360_data/preregistered.md, section "Reads".
#
#   simulate_reads.R transcripts.fa answer_key.tsv outdir
#
# Writes outdir/sample_0[1-6]_[12].fasta.gz (samples 1-3 condition A, 4-6 condition B, which
# gets the fold change), true_counts.tsv (read pairs per transcript and sample, polyester's own
# count matrix) and polyester_version.txt.
suppressPackageStartupMessages({
  library(polyester)
  library(Biostrings)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
fa <- args[1]
key <- read.delim(args[2], stringsAsFactors = FALSE)
outdir <- args[3]

tx <- readDNAStringSet(fa)
ids <- sub(" .*", "", names(tx))
fc <- key$fold_change[match(ids, key$transcript_id)]
if (anyNA(fc)) stop("transcripts missing from the answer key: ", paste(ids[is.na(fc)], collapse = ", "))

reads_per_transcript <- round(width(tx) / 5)
fold_changes <- cbind(rep(1, length(ids)), fc)

simulate_experiment(
  fasta = fa,
  reads_per_transcript = reads_per_transcript,
  num_reps = c(3, 3),
  fold_changes = fold_changes,
  outdir = outdir,
  paired = TRUE,
  readlen = 100,
  fraglen = 250,
  fragsd = 25,
  error_model = "uniform",
  error_rate = 0.005,
  seed = 360,
  gzip = TRUE
)

load(file.path(outdir, "sim_counts_matrix.rda")) # counts_matrix
colnames(counts_matrix) <- sprintf("sample_%02d", seq_len(ncol(counts_matrix)))
write.table(data.frame(transcript_id = ids, counts_matrix, check.names = FALSE),
            file.path(outdir, "true_counts.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(c(paste("polyester", as.character(packageVersion("polyester"))),
             paste("Biostrings", as.character(packageVersion("Biostrings"))),
             R.version.string),
           file.path(outdir, "polyester_version.txt"))
