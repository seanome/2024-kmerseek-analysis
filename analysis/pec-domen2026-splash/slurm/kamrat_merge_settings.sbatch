#!/bin/bash
# KaMRaT merge settings against contig length, on the three small contrasts (notebook 361).
# One index and score per contrast, as in the kamrat process of nextflow-runs/pec-domen2026/main.nf,
# then one merge per setting. Writes $W/<contrast>/<setting>/{contigs_all.tsv,kept/contigs.tsv} and
# $W/lengths.<contrast>.tsv. The index holds DE-kupl's k-mer table, so k stays 31.
#SBATCH --job-name=nb361-kamrat-merge
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/users/olgabot/nb361-pec/kamrat_merge_settings/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=$S/kamrat_merge_settings
export PATH=$S/env/bin:$PATH
CONTRASTS=(young_13_49d young_132d old_41_93d)
C=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}
K="apptainer exec --bind /scratch $S/images/kamrat_v1.2.sif kamrat"
D=$R/design/contrast_$C.design.tsv
mkdir -p $W/$C && cd $W/$C
awk -F'\t' 'NR > 1 { print $1 "\t" $2 }' $D > design.txt
rm -rf idx && mkdir idx
$K index -intab $S/contrasts/$C/dekupl_limma/DEkupl_result/kmer_counts/raw-counts.tsv.gz -outdir idx -klen 31 -unstrand -nfbase 1000000000 > kamrat.log 2>&1
$K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt bin -outpath score.bin >> kamrat.log 2>&1
for s in "pearson:0.20 30-15" "pearson:0.35 30-15" "pearson:0.50 30-15" "none 30-15" "pearson:0.20 30-25" "none 30-25"; do
  set -- $s
  t=${1//:/_}.ov${2}
  mkdir -p $t
  $K merge -idxdir idx -overlap $2 -with score.bin:min -interv $1 -outfmt tab -counts mean:float -outpath $t/contigs_all.tsv >> kamrat.log 2>&1
  python $R/scripts/select_kamrat_nominal.py $t/contigs_all.tsv $D $t/kept 0 > $t/kept.log
done
python $R/scripts/kamrat_merge_lengths.py $W/$C $W/lengths.$C.tsv
rm -rf idx score.bin
