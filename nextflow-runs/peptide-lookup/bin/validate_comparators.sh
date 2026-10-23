#!/bin/bash
# Check the two database counts and the source/hidden examples in notebook section 11.
set -euo pipefail
module load biology diamond/2.1.23 ncbi-blast+/2.16.0
s4=$SCRATCH/peptide-lookup-300/step4
{
 diamond dbinfo --db "$s4/comparators/diamond.dmnd"
 blastdbcmd -db "$s4/comparators/blast" -info
 blastdbcmd -db "$s4/comparators/blast" -entry MGYG000276714_01980 -outfmt '%a %l'
 if blastdbcmd -db "$s4/comparators/blast" -entry MGYG000029077_00369 -outfmt '%a %l'; then
   echo 'ERROR: hidden source found' >&2
   exit 1
 else
   echo 'Checked hidden accession MGYG000029077_00369: absent'
 fi
} > "$s4/comparator_validation.txt" 2>&1
