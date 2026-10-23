#!/bin/bash
#SBATCH --job-name=botryllus-PEC-nfcore-trim
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code"
ml devel java/21.0.4 python/3.12.1 rust/1.90.0
export NXF_HOME="$HOME/.nextflow"
export NXF_TEMP="$L_SCRATCH_JOB"
export APPTAINER_TMPDIR="$L_SCRATCH_JOB"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
export NXF_SINGULARITY_CACHEDIR="$ROOT/images"
mkdir -p "$ROOT/nfcore" "$ROOT/images" "$ROOT/software"
cd "$ROOT"
# Reuse a read-only pinned executable; capture its digest before use.
NEXTFLOW="$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4"
sha256sum "$NEXTFLOW" > nfcore/nextflow.sha256
"$NEXTFLOW" -version > nfcore/nextflow-version.txt
REV=a1fcdddd3b826fe46eb46f0479f2ff8a7815af05
if [ ! -d "software/rnaseq-${REV}" ]; then
    curl --fail --location --retry 5 -o software/rnaseq-source.tar.gz.partial \
        "https://codeload.github.com/nf-core/rnaseq/tar.gz/${REV}"
    tar -tzf software/rnaseq-source.tar.gz.partial > software/rnaseq-source-files.txt
    mv software/rnaseq-source.tar.gz.partial software/rnaseq-source.tar.gz
    tar -xzf software/rnaseq-source.tar.gz -C software
fi
sha256sum software/rnaseq-source.tar.gz > nfcore/rnaseq-source.sha256
for ref in botznik-transcripts.fa nc_nsi_nf.gtf; do
    if [ ! -e "references/$ref" ]; then
        /bin/cp "$GROUP_HOME/Botryllus_Genome/$ref" "references/$ref.partial"
        mv "references/$ref.partial" "references/$ref"
    fi
done
sha256sum references/botznik-transcripts.fa references/nc_nsi_nf.gtf > nfcore/reference-SHA256SUMS
python3 - "$ROOT" "$CODE/ena-runs.tsv" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=list(csv.DictReader(open(sys.argv[2]),delimiter='\t'))
with (root/'nfcore/samplesheet.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['sample','fastq_1','fastq_2','strandedness'])
    for row in rows:
        run=row['run_accession'];marker=root/'raw'/run/'validated.json'
        record=json.loads(marker.read_text())
        assert record['archive']['md5']==row['fastq_md5']
        assert record['pairs']==int(row['read_count'])
        writer.writerow([run,*[f['path'] for f in record['files']],'auto'])
PY
"$NEXTFLOW" -c "$CODE/nfcore.config" config "software/rnaseq-${REV}" -profile singularity -o flat > nfcore/resolved.config
/usr/bin/time -v "$NEXTFLOW" run "software/rnaseq-${REV}" \
    -profile singularity -c "$CODE/nfcore.config" -work-dir "$ROOT/nfcore/work" -resume \
    --input "$ROOT/nfcore/samplesheet.csv" --outdir "$ROOT/nfcore/results" \
    --skip_alignment --skip_pseudo_alignment --save_trimmed --igenomes_ignore \
    --transcript_fasta "$ROOT/references/botznik-transcripts.fa" \
    --gtf "$ROOT/references/nc_nsi_nf.gtf" --trimmer trimgalore --min_trimmed_reads 1 \
    --extra_trimgalore_args '--quality 20 --length 62 --stringency 3 --adapter AGATCGGAAGAGC --adapter2 AGATCGGAAGAGC --no_poly_g' \
    -with-report "$ROOT/nfcore/execution-report.html" \
    -with-timeline "$ROOT/nfcore/execution-timeline.html" 2> nfcore/head.time.log
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/validate_pairs"
/usr/bin/time -v python3 "$CODE/stage_trimmed.py" "$ROOT" "$CODE/ena-runs.tsv" "$L_SCRATCH_JOB/validate_pairs" \
    2> nfcore/validation.time.log
