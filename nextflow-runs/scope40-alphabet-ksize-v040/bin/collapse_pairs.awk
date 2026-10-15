#!/usr/bin/awk -f
# One row per protein pair from kmerseek search's one row per matched region.
#
#   kmerseek search ... | collapse_pairs.awk directions.csv - > pairs.csv
#
# directions.csv (from score_scope40_setting.py --print-directions) lists each score column
# and whether its best value is the smallest (min) or the largest (max). Every other column
# is dropped. A pair's rows are merged while they are adjacent, keeping each score's best
# value; the scorer merges again with polars, so a pair whose rows are not adjacent is still
# counted once. Writes the number of region rows read to raw_rows.txt.
#
# kmerseek writes an infinite E-value as "inf". awk implementations disagree on "inf" + 0
# (gawk reads 0, mawk reads infinity), so num() turns it into infinity explicitly; an empty
# field is a missing value and never replaces a present one.

BEGIN { FS = ","; OFS = "," }

FNR == NR {
    if ($1 != "column") dir[$1] = $2
    next
}

FNR == 1 {
    for (i = 1; i <= NF; i++) {
        if ($i == "query_name") qi = i
        else if ($i == "target_name") ti = i
        else if ($i in dir) { n++; col[n] = i; best_is[n] = dir[$i]; name[n] = $i }
    }
    if (!qi || !ti) { print "collapse_pairs.awk: no query_name or target_name column" > "/dev/stderr"; exit 3 }
    if (n == 0) { print "collapse_pairs.awk: none of the score columns is in the header" > "/dev/stderr"; exit 3 }
    line = "query_name" OFS "target_name"
    for (j = 1; j <= n; j++) line = line OFS name[j]
    print line
    next
}

{
    rows++
    key = $qi SUBSEP $ti
    if (key != cur) {
        flush()
        cur = key; q = $qi; t = $ti
        for (j = 1; j <= n; j++) best[j] = $col[j]
        next
    }
    for (j = 1; j <= n; j++) {
        v = $col[j]
        if (v == "") continue
        if (best[j] == "" || (best_is[j] == "min" ? num(v) < num(best[j]) : num(v) > num(best[j]))) best[j] = v
    }
}

function num(x) {
    if (x ~ /^\+?[iI]nf/) return 1e308 * 10
    if (x ~ /^-[iI]nf/) return -1e308 * 10
    return x + 0
}

function flush(   line, j) {
    if (cur == "") return
    line = q OFS t
    for (j = 1; j <= n; j++) line = line OFS best[j]
    print line
}

END {
    flush()
    print rows + 0 > "raw_rows.txt"
}
