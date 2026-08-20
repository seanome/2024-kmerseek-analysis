# Reduce Reseek output to this pipeline's normalized columns.
#
# Lives in its own file rather than inline in main.nf: awk's regexes and $-variables
# collide with Groovy's GString lexer, which failed to tokenise the process script at the
# exact column where the awk began. A separate file has no such interaction.
#
# In:  query target qlo qhi tlo thi pctid pvalue   (Reseek -columns order)
# Out: same eight, with query/target reduced to bare UniProt accessions.
BEGIN { FS = "\t"; OFS = "\t" }
$1 ~ /^query/ { next }                      # drop a header row if -columns emits one
{
    for (i = 1; i <= 2; i++) {
        n = split($i, parts, "/")           # Reseek names rows by structure path
        leaf = parts[n]
        if (leaf ~ /^AF-/) {                # AF-<acc>-F1.cif -> <acc>
            split(leaf, q, "-")
            $i = q[2]
        } else {
            sub(/\.(pdb|cif|mmcif|ent)(\.gz)?$/, "", leaf)
            $i = leaf
        }
    }
    print
}
