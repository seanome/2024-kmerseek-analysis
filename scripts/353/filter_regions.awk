#!/usr/bin/awk -f
# Keep the kmerseek region rows notebook 353 counts, as the CSV streams past. Same rule as
# filter_regions.py (kept for its docstring and the test), in awk because kmerseek writes
# 74 million rows for 42,000 short queries at dayhoff6 k6 and Python's csv module made the
# filter, not kmerseek, the slow step (14.7 min, 116% CPU, development pilot 48678981).
#
#   kmerseek search ... | awk -v max_e=10 -v summary=s.tsv -f filter_regions.awk | gzip > regions.tsv.gz
#
# A row is kept when region_evalue is a number at most max_e. kmerseek's CSV has no quoted
# fields in these columns; a line holding a double quote stops the filter with exit 3 rather
# than being split wrongly.
BEGIN {
    FS = ","; OFS = "\t"
    n_keep = split("query_name target_name region_start region_end target_start target_end region_subseq target_subseq region_length region_n_mismatches region_n_shared_kmers region_evalue region_evalue_source region_run_evalue region_ka_evalue", keep, " ")
}
NR == 1 {
    for (i = 1; i <= NF; i++) col[$i] = i
    line = ""
    for (j = 1; j <= n_keep; j++) {
        if (!(keep[j] in col)) { print "kmerseek output has no column " keep[j] > "/dev/stderr"; exit 2 }
        line = line (j > 1 ? OFS : "") keep[j]
    }
    print line
    e = col["region_evalue"]; q = col["query_name"]
    next
}
index($0, "\"") { print "quoted field on line " NR > "/dev/stderr"; bad = 1; exit 3 }
{
    rows_in++
    if (!($q in seen)) { seen[$q] = 1; n_queries++ }
    v = $e
    if (v == "" || v !~ /^[0-9.eE+-]+$/) { no_e++; next }
    if (v + 0 <= max_e + 0) {
        rows_kept++
        line = ""
        for (j = 1; j <= n_keep; j++) line = line (j > 1 ? OFS : "") $(col[keep[j]])
        print line
    }
}
END {
    if (bad) exit 3
    if (NR == 0) print "query_name\ttarget_name\tregion_start\tregion_end\ttarget_start\ttarget_end\tregion_subseq\ttarget_subseq\tregion_length\tregion_n_mismatches\tregion_n_shared_kmers\tregion_evalue\tregion_evalue_source\tregion_run_evalue\tregion_ka_evalue"
    printf "rows_in\trows_kept\trows_no_evalue\tqueries_with_rows\n%d\t%d\t%d\t%d\n", rows_in, rows_kept, no_e, n_queries > summary
}
