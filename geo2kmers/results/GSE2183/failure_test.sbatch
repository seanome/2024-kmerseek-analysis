#!/bin/bash
#SBATCH -J geo2kmers-verify-fail
#SBATCH -p normal,hns
#SBATCH -A ayeletv
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/users/olgabot/geo2kmers-verify/fail.%j.log
set -uo pipefail
V=/scratch/users/olgabot/geo2kmers-verify
REAL=/scratch/users/olgabot/geo2kmers-test/env/bin/python
F=$V/fail
mkdir -p $F/bin
cat > $F/bin/kill2 <<EOF
#!/bin/bash
c=$F/count.kill2; n=\$(cat \$c 2>/dev/null || echo 0); echo \$((n+1)) > \$c
if [ \$n -lt 2 ]; then kill -9 \$\$; fi
exec $REAL "\$@"
EOF
cat > $F/bin/killall <<EOF
#!/bin/bash
kill -9 \$\$
EOF
cat > $F/bin/exit1 <<EOF
#!/bin/bash
echo failing on purpose >&2
exit 1
EOF
chmod +x $F/bin/*
ml load biology nextflow
export NXF_OPTS="-Xms512m -Xmx2g"
SMALL="--sketches dna:21 --modes present"
run() { # name, then nextflow args
  name=$1; shift
  d=$F/$name; mkdir -p $d; cd $d
  start=$(date +%s)
  nextflow run $V/code/main.nf -profile small --gse GSE2183 --outdir $d/out --python_sequences $REAL --python_sketch $REAL $SMALL "$@" > nf.log 2>&1
  st=$?
  echo "=== $name exit $st wall $(( $(date +%s) - start ))"
  cut -f2,3,4,7 geo2kmers.trace.txt | awk 'NR>1' | sort | uniq -c | sort -rn | head -8
  grep -E 'ERROR|Error executing' nf.log | head -3 | cut -c1-160
  ls out/sketches 2>/dev/null | wc -l
}
run kill2 --python_fetch $F/bin/kill2
run killall --python_fetch $F/bin/killall
run exit1 --python_fetch $F/bin/exit1
run skfail --python_fetch $REAL --python_sketch $F/bin/exit1
run options --python_fetch $REAL --gpl GPL1799 --sample_tables
run badgpl --python_fetch $REAL --gpl GPL0
(unset NCBI_EMAIL; run noemail --python_fetch $REAL)
run noprot --python_fetch $REAL --proteome /no/such/file.fa
echo finished
