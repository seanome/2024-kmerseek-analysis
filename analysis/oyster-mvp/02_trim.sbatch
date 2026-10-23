#!/bin/bash
#SBATCH --job-name=oyster-mvp-trim
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=%x.%A.%a.out
# One library at one depth per task: task = depth_index * 6 + sample_index,
# depths 1m, 5m, full. The 1m and 5m inputs are the first 4,000,000 and
# 20,000,000 lines (1M and 5M pairs) of each ENA file: deterministic prefixes,
# not random subsamples. Trimming: Trim Galore! 0.6.10 + cutadapt 5.2 with the
# nf-core/rnaseq 3.27.0 default settings (-q 20 --length 20, paired).
# Output appears in $ROOT/trimmed/<depth>/ only after validation.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
depths=(1m 5m full)
i=$SLURM_ARRAY_TASK_ID
sample=${SAMPLES[$((i % 6))]}
depth=${depths[$((i / 6))]}
lines=${DEPTH_LINES[$depth]}
out="$ROOT/trimmed/$depth"
done_marker="$out/$sample.validated.json"
[ -f "$done_marker" ] && { echo "already done $sample $depth"; exit 0; }
export PATH="$SHAZAM_ENV/bin:$PATH"
work="$out/.work-$sample"
rm -rf "$work"; mkdir -p "$work"

if [ "$depth" = 1m ] && [ -f "$ROOT/prefix/1m/complete" ]; then
  # Checked 1M prefixes (01b_pilot_prefix.sbatch); compared with the full
  # files by check_prefix.sbatch once those are verified.
  in1="$ROOT/prefix/1m/${sample}_1.fastq.gz"; in2="$ROOT/prefix/1m/${sample}_2.fastq.gz"
elif [ ! -f "$ROOT/raw/$sample.verified.tsv" ]; then
  echo "raw $sample not verified" >&2; exit 1
elif [ "$lines" -gt 0 ]; then
  for m in 1 2; do
    # head closes the pipe early; that SIGPIPE in zcat is expected.
    set +o pipefail
    zcat "$ROOT/raw/${sample}_$m.fastq.gz" | head -n "$lines" | pigz -p 4 -1 > "$work/${sample}_$m.fastq.gz"
    set -o pipefail
    got=$(zcat "$work/${sample}_$m.fastq.gz" | wc -l)
    [ "$got" -eq "$lines" ] || { echo "prefix $sample $m has $got lines, want $lines" >&2; exit 1; }
  done
  in1="$work/${sample}_1.fastq.gz"; in2="$work/${sample}_2.fastq.gz"
else
  in1="$ROOT/raw/${sample}_1.fastq.gz"; in2="$ROOT/raw/${sample}_2.fastq.gz"
fi

timed "trim-$depth-$sample" "$TRIMGALORE" --paired --gzip --cores 4 -q 20 --length 20 \
  --basename "$sample" -o "$work" "$in1" "$in2"

python3 - "$work" "$sample" "$depth" "$lines" <<'EOF'
import json, re, subprocess, sys
from pathlib import Path
work, sample, depth, lines = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
def count(p):
    n = int(subprocess.run(f"zcat {p} | wc -l", shell=True, check=True, capture_output=True, text=True).stdout)
    assert n % 4 == 0, (p, n)
    return n // 4
def grab(text, label):
    m = re.search(re.escape(label) + r"\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None
reports = sorted(work.glob("*_trimming_report.txt"))
assert len(reports) == 2, reports
r1, r2 = (p.read_text() for p in reports)
kept1, kept2 = count(work / f"{sample}_val_1.fq.gz"), count(work / f"{sample}_val_2.fq.gz")
assert kept1 == kept2, (kept1, kept2)
pairs_in = grab(r1, "Total reads processed:")
removed = grab(r2, "Number of sequence pairs removed because at least one read was shorter than the length cutoff (20 bp):")
assert pairs_in - removed == kept1, (pairs_in, removed, kept1)
if lines:
    assert pairs_in == lines // 4, (pairs_in, lines)
rec = dict(sample=sample, depth=depth, pairs_in=pairs_in, pairs_kept=kept1, pairs_removed=removed,
           reads_with_adapters=[grab(r1, "Reads with adapters:"), grab(r2, "Reads with adapters:")],
           quality_trimmed_bp=[grab(r1, "Quality-trimmed:"), grab(r2, "Quality-trimmed:")],
           prefix="first %d pairs of the ENA file (deterministic, not random)" % (lines // 4) if lines else "full file")
(work / "stats.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
EOF

# Publish: move outputs into place, then name them the way DE-kupl expects.
mv "$work/${sample}_val_1.fq.gz" "$out/${sample}_val_1.fq.gz"
mv "$work/${sample}_val_2.fq.gz" "$out/${sample}_val_2.fq.gz"
ln -sf "${sample}_val_1.fq.gz" "$out/${sample}_1.fq.gz"
ln -sf "${sample}_val_2.fq.gz" "$out/${sample}_2.fq.gz"
mkdir -p "$out/reports"; mv "$work"/*_trimming_report.txt "$out/reports/"
mv "$work/stats.json" "$done_marker"
rm -rf "$work"
