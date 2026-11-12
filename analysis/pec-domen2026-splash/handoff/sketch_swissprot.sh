#!/usr/bin/env bash
set -euo pipefail
cd /home/claude/work/swissprot
while ! grep -q FINISHED /home/claude/work/tools/pip_b.log 2>/dev/null; do sleep 30; done
# manysketch wants a CSV: name,genome_filename,protein_filename ; --singleton sketches each record separately
echo "name,genome_filename,protein_filename" > sp.csv; echo "swissprot,,swissprot.gz" >> sp.csv
/home/claude/work/tools/env/bin/time -f "%e s %M KB" -o sketch.time sourmash scripts manysketch sp.csv --singleton \
  -p protein,k=7,scaled=5,abund -p dayhoff,k=12,scaled=5,abund -p hp,k=30,scaled=5,abund -c 2 -o swissprot.scaled5.zip > sketch.log 2>&1
echo DONE > sketch.done
