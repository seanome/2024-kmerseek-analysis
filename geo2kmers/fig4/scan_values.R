# Write SCAN values for each PM probe and each probe set from Affymetrix CEL files, one array at a time.
# Usage: Rscript scan_values.R <folder of CEL files> <probes.tsv.gz> <probe_set_values.tsv.gz> <cores> [check|-]
#        [<probes without unfitted.tsv.gz> <probe_set_values without unfitted.tsv.gz>]
# The rules are in scan_plan.md. SCAN.UPC's SCANfast returns only probe-set values (its writer for single probes is
# commented out in the package), so this runs SCANfast's steps from processCelFile with the package's own functions:
# fit the sequence model on every n-th probe (10000 probes, convThreshold 0.5), standardise each probe within bins
# of 5000 probes with similar predicted brightness, and take the 10 % trimmed mean over each probe set's probes.
# With "check", the first array's probe-set values are also computed by SCANfast, the largest difference printed, and
# the script stops if it is above 1e-6.
# SCAN.UPC's bin loop starts at the 5000th probe, so the 4999 probes predicted dimmest in each array are never
# standardised and stay exactly 0. The last two files leave them out: probes that are 0 in any array are dropped from
# the probe table, and a probe's 0 is left out of its probe set's trimmed mean in that array (scan_plan.md).
suppressMessages({
  library(SCAN.UPC)
  library(oligo)
})
arguments <- commandArgs(trailingOnly = TRUE)
files <- list.files(arguments[1], pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)
cores <- as.integer(arguments[4])
scan <- getNamespace("SCAN.UPC")

# One array: each PM probe's position, sequence, probe set and SCAN value, as in SCAN.UPC's processCelFile.
# With reference, only the values are kept, after checking the probes come in the reference's order.
probe_values <- function(path, reference = NULL) {
  array <- read.celfiles(path)
  data <- scan$getDataForMostArrayTypes(path, array, NA, FALSE)
  if (is.null(data)) stop(path, " has too many zero intensities for SCAN")
  design <- scan$buildDesignMatrix(data[, 2], verbose = FALSE)
  raw <- as.numeric(data[, 1])
  if (min(raw) == 0) raw <- raw + 1
  y <- log2(raw)
  binsize <- 5000
  sampled <- scan$getSampleIndices(total = nrow(data), intervalN = 10000, verbose = FALSE)
  fit <- scan$EM_vMix(y = y[sampled], X = design[sampled, ], nbins = 25, convThreshold = 0.5, verbose = FALSE)
  m1 <- design %*% fit$b1
  index <- order(m1)
  normalised <- rep(0, length(y))
  for (i in 1:(nrow(data) / binsize)) {
    part <- index[(binsize * i):min(binsize * i + binsize, length(y))]
    normalised[part] <- (y[part] - m1[part]) / as.vector(scan$sig(y = y[part], m = m1[part], verbose = FALSE))
  }
  if (!is.null(reference)) {
    stopifnot(identical(data[, 3], reference$xy))
    return(round(normalised, 8))
  }
  data.frame(xy = data[, 3], sequence = data[, 2], probe_set = data[, 4], value = round(normalised, 8))
}

name_of <- function(path) sub("^(GSM[0-9]+).*", "\\1", basename(path))
first <- probe_values(files[1])
rest <- parallel::mclapply(files[-1], probe_values, reference = first, mc.cores = cores)
failed <- !vapply(rest, is.numeric, TRUE)
if (any(failed)) stop("failed: ", paste(basename(files[-1][failed]), collapse = " "), "\n", rest[failed][[1]])
values <- cbind(first$value, do.call(cbind, rest))
colnames(values) <- vapply(files, name_of, "")

probes <- data.frame(sequence = first$sequence, values, check.names = FALSE)
path <- gzfile(arguments[2], "w")
write.table(probes, path, sep = "\t", quote = FALSE, row.names = FALSE)
close(path)

sets <- apply(values, 2, function(v) tapply(v, first$probe_set, mean, trim = 0.1))
path <- gzfile(arguments[3], "w")
write.table(data.frame(probe_set = rownames(sets), round(sets, 4), check.names = FALSE), path, sep = "\t",
            quote = FALSE, row.names = FALSE)
close(path)
cat(length(files), "arrays,", nrow(probes), "PM probes,", nrow(sets), "probe sets\n")
cat("probe values: median", median(probes[[2]]), "range", range(probes[[2]]), "\n")

if (length(arguments) == 7) {
  fitted <- rowSums(values == 0) == 0
  path <- gzfile(arguments[6], "w")
  write.table(probes[fitted, ], path, sep = "\t", quote = FALSE, row.names = FALSE)
  close(path)
  sets <- apply(values, 2, function(v) tapply(ifelse(v == 0, NA, v), first$probe_set, mean, trim = 0.1, na.rm = TRUE))
  path <- gzfile(arguments[7], "w")
  # A probe set whose probes are all 0 in an array has no value there; hp_value_table.py drops its k-mers
  write.table(data.frame(probe_set = rownames(sets), round(sets, 4), check.names = FALSE), path, sep = "\t",
              quote = FALSE, row.names = FALSE, na = "nan")
  close(path)
  cat(sum(!fitted), "probes are 0 in some array and left out;", sum(!is.finite(sets)), "probe set values missing\n")
}

if (length(arguments) >= 5 && arguments[5] == "check") {
  reference <- exprs(SCANfast(files[1], verbose = FALSE))[, 1]
  ours <- sets[names(reference), 1]
  cat("check against SCANfast on", basename(files[1]), ": largest difference", max(abs(ours - reference)), "over",
      length(reference), "probe sets\n")
  if (max(abs(ours - reference)) > 1e-6) stop("these values differ from SCANfast's")
}
