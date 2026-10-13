#!/usr/bin/env bash
# Undo the false fragment offset in stored Reseek and Folddisco region tables
# (regions/reseek/*.tsv.gz, regions/folddisco/*.tsv.gz).
#
# bin/normalize_reseek.awk and bin/folddisco_to_regions.py read the fragment number from
# the first "-F<digits>" in a name, so an accession that starts with F and a digit (F6SXM4)
# was read as that fragment and (d-1)*200 was added to its positions, d being the
# accession's first digit. Fixed in both on 2026-10-01; tables written before that are
# corrected here. Both tools write query, target, qstart, qend, tstart, tend as columns 1-6.
# Foldseek parses names from "AF-" onward and was never affected.
#
# Exact only while no such accession has a real fragment F2 or higher. Checked 2026-10-01
# on the Sherlock structure cache: 0 such files in all 28 species. The script refuses to
# write a start below 1, which is what a real F2+ fragment would produce.
#
# Each table is rewritten in place and the shifted original moved to <backup_dir>. Keep
# that outside results/: loaders glob regions/<tool>/*.tsv.gz, and a backup left there would
# be read as a second table for the tool. A table already in <backup_dir> is skipped, so
# a second run changes nothing.
#
# Usage: fix_af_f_accession_offsets.sh <regions/reseek or regions/folddisco dir> <backup_dir>
set -euo pipefail

dir="$1"
backup_dir="$2"
mkdir -p "$backup_dir"
for f in "$dir"/*.tsv.gz; do
    [[ -e "$f" ]] || continue
    backup="$backup_dir/$(basename "$f")"
    if [[ -e "$backup" ]]; then
        echo "skip (already corrected): $f"
        continue
    fi
    tmp="$f.fixing"
    gzip -dc "$f" | awk -F'\t' 'BEGIN { OFS = "\t" }
        {
            # cols 1,2 = query, target; 3,4 = query span; 5,6 = target span
            for (i = 1; i <= 2; i++) {
                if ($i ~ /^F[2-9]/) {
                    off = (substr($i, 2, 1) - 1) * 200
                    $(2 * i + 1) -= off; $(2 * i + 2) -= off
                    if ($(2 * i + 1) < 1) {
                        print "start below 1 after correction, line " NR ": " $0 > "/dev/stderr"
                        exit 1
                    }
                    n[i]++
                }
            }
            print
        }
        END { printf "  rows corrected: query %d, target %d of %d\n", n[1], n[2], NR > "/dev/stderr" }' \
        | gzip -c > "$tmp"
    echo "$f"
    /bin/mv -f "$f" "$backup"
    /bin/mv -f "$tmp" "$f"
done
