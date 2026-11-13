# Notebook 299: reduce `sourmash scripts manysearch` output as it streams, because some
# settings report almost every query against almost every gene (hp k = 10 at scaled 100 gave
# 52.7 million rows for 39_000 genes in the laptop test; the catalog has about 1_200 times
# as many). Keeps what the kill test needs and nothing else:
#
#   counts: query_id <tab> number of catalog genes it hit (at least one shared hash)
#   pairs:  query_id <tab> gene <tab> DIAMOND E <tab> containment <tab> shared hashes,
#           for every hit that DIAMOND also reports (E <= 1e-3) for that query
#   bad:    lines whose column count is not 16 (a comma inside a name would shift columns)
#
# Usage: awk -v diamond=pairs.tsv -v counts=c.tsv -v pairs=p.tsv -v bad=b.txt -f this.awk < manysearch.csv
# where pairs.tsv is "query <tab> target <tab> evalue" from DIAMOND's hits.
BEGIN {
    FS = ","; OFS = "\t"
    while ((getline line < diamond) > 0) { split(line, f, "\t"); ev[f[1] SUBSEP f[2]] = f[3] }
    nbad = 0
}
$1 == "query_name" { next }  # a header line; chunked searches send one per chunk
NF != 16 { nbad++; next }
{
    q = $1; t = $3; sub(/ .*/, "", t)
    n[q]++
    if ((q SUBSEP t) in ev) print q, t, ev[q SUBSEP t], $4, $5 > pairs
}
END {
    for (q in n) print q, n[q] > counts
    print nbad > bad
    close(pairs)
    close(counts)
}
