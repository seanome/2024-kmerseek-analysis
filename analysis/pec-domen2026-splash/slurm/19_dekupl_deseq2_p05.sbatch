#!/bin/bash
#SBATCH --job-name=nb361-dekupl-deseq2-p05
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=%x.%A_%a.out
# DE-kupl's DESeq2 test on the six real contrasts, then its k-mers at uncorrected p < 0.05 with no
# fold-change cutoff, merged into contigs per direction and searched (added 2026-11-02, Olga: "use
# nominal p < 0.05 for kamrat and de-kupl"). Only the simulated control had a DE-kupl DESeq2 run.
#
# The k-mer counts are not redone: the test step of dekupl-run 1.3.5 (/dekupl/bin/DESeq2_diff_method.R,
# the command its Snakefile runs, copied from the control's dekupl_DESeq2.log) is run on the
# limma run's raw-counts.tsv.gz and sample_conditions_full.tsv (same k-mers, same normalisation
# factors). Its filters are opened up (corrected p <= 1, |log2FC| >= 0), so its diff-counts.tsv.gz
# keeps every tested k-mer with DESeq2's own log2 fold change; raw_pvals.txt.gz has the
# uncorrected p. Output folder: $D/dekupl_DESeq2_allkmers (kmer_counts and metadata are links to
# the limma run's). Every step is timed with /usr/bin/time -v.
# Resources: the control's whole DE-kupl DESeq2 run (2.76 M k-mers, 3 chunks of 1 M) peaked at
# 3.1 GB and 16 min; the test runs chunks of 1 M k-mers in parallel, one per CPU, so about 1 GB per
# CPU, plus the selection, which holds up to 53 M k-mers (young_24h) in memory three times.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
CONTRASTS=(young_2h young_24h old_24h young_13_49d young_132d old_41_93d)
C=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}
D=$PEC_ROOT/contrasts/$C
DSG=$PEC_CODE/design/contrast_$C.design.tsv
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
T="/usr/bin/time -v -o"
IMG=$PEC_ROOT/images/dekupl-run_1.3.5.sif
LIM=$D/dekupl_limma/DEkupl_result
TMP=${L_SCRATCH:-$PEC_ROOT}/dekupl_deseq2_$SLURM_JOB_ID
mkdir -p "$TMP"
W=$(mktemp -d "$PEC_ROOT/tmp_dkd_${C}_XXXX")
cd "$W"

# --- DE-kupl DESeq2 test, every k-mer kept
R=$D/dekupl_DESeq2_allkmers/DEkupl_result
if [ ! -s "$R/control_vs_PEC_kmer_counts/raw_pvals.txt.gz" ]; then
  mkdir -p dk/DEkupl_result/control_vs_PEC_kmer_counts dk/DEkupl_result/Logs
  ln -s "$LIM/kmer_counts" dk/DEkupl_result/kmer_counts
  ln -s "$LIM/metadata" dk/DEkupl_result/metadata
  cd dk
  export APPTAINERENV_R_LIBS=$PEC_ROOT/Rlib_dekupl APPTAINERENV_TMPDIR=$TMP
  $T test.time apptainer exec --bind /scratch --bind "$TMP" "$IMG" Rscript /dekupl/bin/DESeq2_diff_method.R \
    /dekupl/bin/TtestFilter DEkupl_result/kmer_counts/raw-counts.tsv.gz DEkupl_result/metadata/sample_conditions_full.tsv \
    1 0 control PEC "$N" 1000000 "$TMP/test_diff" \
    DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz DEkupl_result/control_vs_PEC_kmer_counts/raw_pvals.txt.gz \
    DEkupl_result/Logs/test_diff_counts.logs fixed > test.log 2>&1
  rm -f tmp_shuff
  test -s DEkupl_result/control_vs_PEC_kmer_counts/raw_pvals.txt.gz
  cd "$W"
  mv dk "$D/dekupl_DESeq2_allkmers"
fi
rm -rf "$TMP"

# --- k-mers with p < 0.05, merged per direction
if [ ! -s "$D/dekupl_deseq2_p05/contigs.tsv" ]; then
  mkdir -p dekupl_deseq2_p05
  $T dekupl_deseq2_p05/select.time python "$PEC_CODE/scripts/select_dekupl_nominal.py" \
    "$R" "$DSG" dekupl_deseq2_p05 "$R/control_vs_PEC_kmer_counts/diff-counts.tsv.gz" > dekupl_deseq2_p05/select.log
  cd dekupl_deseq2_p05
  for dir in up down; do
    gzip "$dir.diff-counts.tsv"
    $T "merge_$dir.time" apptainer exec --bind /scratch "$IMG" \
      /dekupl/bin/mergeTags -k 31 -n "$dir.diff-counts.tsv.gz" 2> "merge_$dir.log" | gzip -c > "$dir.merged.tsv.gz"
  done
  python - <<'EOF'
import polars as pl
parts = [pl.read_csv(f"{d}.merged.tsv.gz", separator="\t").select("contig", "nb_merged_kmers", "pvalue", "log2FC")
         .with_columns(pl.lit(d).alias("direction")) for d in ("up", "down")]
t = pl.concat(parts)
bad = t.filter(((pl.col("direction") == "up") & (pl.col("log2FC") <= 0)) | ((pl.col("direction") == "down") & (pl.col("log2FC") >= 0)))
assert bad.height == 0, bad.head()
t.write_csv("contigs.tsv", separator="\t")
print("contigs", t.group_by("direction").len().sort("direction").rows())
EOF
  cd "$W"
  test -s dekupl_deseq2_p05/contigs.tsv
  mv dekupl_deseq2_p05 "$D/"
fi
cat "$D/dekupl_deseq2_p05/summary.json"

# --- protein search, as the search process of main.nf (same commands as 18_kamrat_dekupl_p05.sbatch)
src=dekupl_deseq2_p05
if [ ! -e "$D/search_$src/search.done" ]; then
  mkdir "search_$src" && cd "search_$src"
  ln -s "$D/$src/contigs.tsv" seqs.tsv
  START=$(date +%s)
  "$PT" translate seqs.tsv pep 20 contig 2> translate.log
  cat translate.log
  if [ -s pep.peptides.faa ]; then
    seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
    for fa in chunks/*.faa; do
      c=$(basename "$fa" .faa); c=${c##*.}
      $T "sketch.$c.time" sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$c" "$PWD/$fa") \
        --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o "$c.scaled5.zip"
      for db in proteome swissprot botznik; do
        for spec in protein:7 dayhoff:12 hp:30; do
          m=${spec%:*}; k=${spec#*:}
          for s in 10 100 5; do
            $T "ms.$c.$db.$m.k$k.s$s.time" sourmash scripts manysearch "$c.scaled5.zip" "$PEC_ROOT/db/$db.$m.k$k.s$s.rocksdb" \
              -m $m -k $k -s $s -t 0.000001 -N -c "$N" -o "ms.$c.$db.$m.k$k.s$s.csv" 2>> manysearch.log
          done
        done
      done
    done
    $T pairs.time python "$PEC_CODE/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
    for db in proteome swissprot botznik; do
      $T "align.$db.time" python "$PEC_CODE/scripts/pair_align.py" pairs.$db.tsv pep.peptides.faa "$PEC_ROOT/db/$db.faa" align.$db.parquet "$N"
    done
  fi
  echo "wall seconds $(( $(date +%s) - START ))" > search.wall
  rm -f seqs.tsv
  touch search.done
  cd "$W"
  mv "search_$src" "$D/"
fi
echo "$src peptides: $(grep -c '>' "$D/search_$src/pep.peptides.faa" || true)"
cd "$PEC_ROOT" && rm -rf "$W"
