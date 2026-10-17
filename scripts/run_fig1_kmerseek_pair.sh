#!/usr/bin/env bash
# Figure 1a input: kmerseek pair on CED-9 (P41958) against BCL2 (P10415), hp_lehninger2, k = 17.
#
# Usage: scripts/run_fig1_kmerseek_pair.sh /path/to/kmerseek/checkout /path/to/kmerseek/binary
#
# Writes data/fig1_ced9_bcl2/ced9_bcl2_hp_lehninger2_k17.pair.json and a .provenance.txt next to
# it with the kmerseek commit and version. Notebook 275 reads both and stops if the region is not
# CED-9 163-181 against BCL2 139-157 (1-based, inclusive) from 3 shared 17-mers.
#
# The FASTA and the UniProt motif tables in data/fig1_ced9_bcl2/ are from UniProt release
# 2026_03 (rest.uniprot.org, fields accession,id,length,ft_motif,ft_transmem).
set -euo pipefail

KMERSEEK_REPO=$1
KMERSEEK_BIN=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
D=$HERE/data/fig1_ced9_bcl2
OUT=$D/ced9_bcl2_hp_lehninger2_k17.pair.json

"$KMERSEEK_BIN" pair \
    --query "$D/ced9_P41958_bcl2_P10415.fasta" --query-name "sp|P41958|CED9_CAEEL" \
    --target "$D/ced9_P41958_bcl2_P10415.fasta" --target-name "sp|P10415|BCL2_HUMAN" \
    --alphabet hp_lehninger2 --ksize 17 --output "$OUT"

{
    echo "kmerseek_commit: $(git -C "$KMERSEEK_REPO" rev-parse HEAD)"
    echo "kmerseek_commit_is_origin_main: $(git -C "$KMERSEEK_REPO" rev-parse origin/main)"
    echo "kmerseek_version: $("$KMERSEEK_BIN" --version)"
    echo "command: kmerseek pair --query-name sp|P41958|CED9_CAEEL --target-name sp|P10415|BCL2_HUMAN --alphabet hp_lehninger2 --ksize 17"
    echo "run_date: $(date +%F)"
} > "${OUT%.json}.provenance.txt"
cat "${OUT%.json}.provenance.txt"
