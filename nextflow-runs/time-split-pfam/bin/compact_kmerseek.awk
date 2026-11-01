#!/usr/bin/awk -f
# Reduce a kmerseek search CSV (stdin) to the family transfers the time-split test scores
# (notebook 280). A hit transfers family F when its interval on the target overlaps a
# Pfam 33.1 domain of F on that Swiss-Prot 2019_08 protein; a hit that overlaps no domain
# can transfer nothing and is dropped. One output row per (hit, overlapped domain).
#
#   kmerseek search ... | awk -F, -v domains=sprot_domains.tsv -f compact_kmerseek.awk
#
# Coordinates: kmerseek reports 0-based starts with exclusive ends; this writes 1-based
# starts with ends included, the convention of Pfam, MMseqs2 and HMMER, so start+1, end.
# Columns are found by header name, never by position. Query and target names are bare
# accessions (make_search_inputs.py), so a bare comma split is safe.
BEGIN {
    OFS = "\t"
    if (domains == "") { print "set -v domains=<sprot_domains.tsv>" > "/dev/stderr"; exit 2 }
    while ((getline line < domains) > 0) {
        split(line, d, "\t")
        if (d[1] == "sp_acc") continue
        i = ++n[d[1]]; fam[d[1], i] = d[2]; ds[d[1], i] = d[3]; de[d[1], i] = d[4]
    }
    close(domains)
    split("query_name target_name region_start region_end target_start target_end " \
          "region_length region_n_shared_kmers region_poisson_score region_tfidf " \
          "region_mean_idf region_n_mismatches region_ka_bits region_evalue " \
          "query_poisson_pvalue", want, " ")
}
NR == 1 {
    for (i = 1; i <= NF; i++) col[$i] = i
    for (j in want) if (!(want[j] in col)) {
        print "kmerseek output has no column " want[j] > "/dev/stderr"; bad = 1
    }
    if (bad) exit 3
    print "query", "qstart", "qend", "target", "tstart", "tend", "pfam_acc", "dstart", "dend",
          "region_length", "region_n_shared_kmers", "region_poisson_score", "region_tfidf",
          "region_mean_idf", "region_n_mismatches", "region_ka_bits", "region_evalue",
          "query_poisson_pvalue"
    next
}
{
    t = $col["target_name"]
    if (!(t in n)) next
    ts = $col["target_start"] + 1; te = $col["target_end"] + 0
    for (i = 1; i <= n[t]; i++) {
        if (ts <= de[t, i] && ds[t, i] <= te) {
            print $col["query_name"], $col["region_start"] + 1, $col["region_end"] + 0, t, ts, te,
                  fam[t, i], ds[t, i], de[t, i], $col["region_length"],
                  $col["region_n_shared_kmers"], $col["region_poisson_score"],
                  $col["region_tfidf"], $col["region_mean_idf"], $col["region_n_mismatches"],
                  $col["region_ka_bits"], $col["region_evalue"], $col["query_poisson_pvalue"]
        }
    }
}
