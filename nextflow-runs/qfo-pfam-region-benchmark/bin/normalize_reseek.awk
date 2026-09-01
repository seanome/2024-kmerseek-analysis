# Reduce Reseek output to this pipeline's normalized columns.
#
# Lives in its own file rather than inline in main.nf: awk's regexes and $-variables
# collide with Groovy's GString lexer, which failed to tokenise the process script at the
# exact column where the awk began. A separate file has no such interaction.
#
# In:  query target qlo qhi tlo thi pctid pvalue   (Reseek -columns order)
# Out: same eight, with query/target reduced to bare UniProt accessions.
# AlphaFold models proteins over 2700 aa as overlapping 1400-residue fragments with a
# 200-residue stride, numbering each fragment's residues from 1. A hit on F<n> is therefore
# offset by (n-1)*200 from the full-sequence coordinates Pfam uses. Verified directly on
# AF-A0A087WUL8-F2: auth_seq_id 1..1400, SIFTS xref UniProt 201..1600.
function af_offset(leaf,   n) {
    if (match(leaf, /-F[0-9]+/)) {
        n = substr(leaf, RSTART + 2, RLENGTH - 2) + 0
        return (n - 1) * 200
    }
    return 0
}

BEGIN { FS = "\t"; OFS = "\t" }
$1 ~ /^query/ { next }                      # drop a header row if -columns emits one
{
    for (i = 1; i <= 2; i++) {
        n = split($i, parts, "/")           # Reseek names rows by structure path
        leaf = parts[n]
        if (leaf ~ /^AF-/) {                # AF-<acc>-F<n>.cif -> <acc>, plus the offset
            off = af_offset(leaf)
            if (off) {                      # cols 3,4 are the query span; 5,6 the target
                $(2 * i + 1) += off
                $(2 * i + 2) += off
            }
            split(leaf, q, "-")
            $i = q[2]
        } else {
            sub(/\.(pdb|cif|mmcif|ent)(\.gz)?$/, "", leaf)
            $i = leaf
        }
    }
    print
}
