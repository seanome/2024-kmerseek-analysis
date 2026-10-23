#!/bin/bash
# Recover every phmmer and jackhmmer chunk's hits from the task work folders.
#
# Before 2026-10-20, main.nf named a chunk's output after the chunk's simpleName, which is
# "queries" for every chunk (queries.1.fasta, queries.2.fasta, ...). Every chunk published
# queries.<tool>.tsv.gz, and each publish overwrote the one before, so the published folder
# held one chunk. Each task's own output is still in its work folder. This copies each one
# to hits/<tool>/queries.<n>.<tool>.tsv.gz, the name the fixed main.nf gives it, and moves
# the overwritten file to hits/<tool>/overwritten/, where score.py does not read it.
#
# Usage (on Sherlock, from nextflow-runs/convergence-afp):
#   scripts/convergence_recover_hmmer_chunks.sh runs/<run> results-<run>/hits
set -euo pipefail
launch=$1
hits=$2

for tool in phmmer jackhmmer; do
    mkdir -p "$hits/$tool/overwritten"
    if [ -f "$hits/$tool/queries.$tool.tsv.gz" ]; then
        mv "$hits/$tool/queries.$tool.tsv.gz" "$hits/$tool/overwritten/"
    fi
    n=0
    for h in $(awk -F'\t' -v p="${tool}Search" \
        '$3 ~ "^"p && ($5 == "COMPLETED" || $5 == "CACHED") {print $2}' \
        "$launch"/trace*.txt | sort -u); do
        d=$(ls -d "$launch"/work/"$h"* | head -1)
        chunk=$(cd "$d" && ls queries.*.fasta)
        out="$hits/$tool/${chunk%.fasta}.$tool.tsv.gz"
        cp "$d/queries.$tool.tsv.gz" "$out"
        n=$((n + 1))
        echo "$tool: $d/$chunk -> $out ($(zcat "$out" | tail -n +2 | cut -f1 | sort -u | wc -l) queries with hits)"
    done
    echo "$tool: $n chunk(s) recovered"
done
