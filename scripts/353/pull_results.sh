#!/bin/bash
# Notebook 353: copy the small result tables from Sherlock into the repo (laptop).
# The query FASTA files and the full region tables stay on Sherlock in
# /scratch/users/olgabot/nb353 (rebuildable from the committed scripts and data).
#
#   bash scripts/353/pull_results.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dest=$(cd "$here/../.." && pwd)/data/kill_tests/short_stretch/results
mkdir -p "$dest"
# Tables are gzipped on the way (-n: no time stamp, so an unchanged table gives the same file).
ssh sherlock 'cd /scratch/users/olgabot/nb353 && \
  for f in $(ls -d dev/*/ held_out/*/ 2>/dev/null); do \
    for t in real_scores.tsv shuffle_scores.tsv seg_flags.tsv; do [ -s $f$t ] && gzip -n -c $f$t > $f$t.gz; done; \
  done; \
  tar cf - $(ls dev/*/{real_scores,shuffle_scores,seg_flags}.tsv.gz held_out/*/{real_scores,shuffle_scores,seg_flags}.tsv.gz \
               dev/*/{filter_summary,evalue_sources,queries.counts}.tsv held_out/*/{filter_summary,evalue_sources,queries.counts}.tsv \
               dev/*/search.time held_out/*/chunks/*.search.time held_out/*/examples.tsv.gz \
               index/*/index.log index/*/index.time index/*/ka_fit_stored 2>/dev/null)' \
  | tar xf - -C "$dest"
ssh sherlock "sacct -u olgabot -S 2026-10-20 --name=nb353-pilot,nb353-dev,nb353-heldout-prep,nb353-heldout-search,nb353-heldout-score -P --format=JobID,JobName,State,Elapsed,TotalCPU,CPUTimeRAW,AllocCPUS,Start,End" > "$dest/sacct.tsv"
find "$dest" -type f | sort
