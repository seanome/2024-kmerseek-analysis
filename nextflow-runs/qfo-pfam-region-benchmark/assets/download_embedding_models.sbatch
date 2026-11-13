#!/bin/bash
# Model weights for the embedding search arms, into $SCRATCH/data/embedding-models.
# Submitted by `make embedding-models`.
#
#   ProtT5-XL-UniRef50 (TM-Vec's encoder)  zenodo.org/record/4644188, prot_t5_xl_uniref50.zip,
#                                          5_279_684_584 bytes (Content-Length, 2026-10-16)
#   PLMSearch data (model/plmsearch.sav and model/esm/ ESM-1b weights)
#                                          zenodo.org/records/11480660, plmsearch_data.tar.gz,
#                                          12_995_086_677 bytes (Zenodo API, 2026-10-16);
#                                          only model/ is kept after unpacking
#
# NOT downloaded here, by hand instead: the TM-Vec checkpoint and its params json, and
# the DeepBLAST checkpoint. The URLs in the DeepBLAST wiki
# (users.flatironinstitute.org/.../tm_vec_cath_model.ckpt, .../deepblast-v3.ckpt) returned
# 404 on 2026-10-16; the TM-Vec README points to a figshare share link
# (figshare.com/s/e414d6a52fd471d86d69) whose files load only in a browser. Put them in
# $SCRATCH/data/embedding-models/tmvec/ as tm_vec_cath_model.ckpt,
# tm_vec_cath_model_params.json and deepblast.ckpt.
#SBATCH --job-name=embedding-models
#SBATCH --account=ayeletv
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=embedding-models-%j.out
set -euo pipefail

OUT=$SCRATCH/data/embedding-models
mkdir -p "$OUT/prott5" "$OUT/plmsearch" "$OUT/tmvec"

fetch() {
    local url=$1 dest=$2
    for attempt in 1 2 3 4 5 6 7 8; do
        if curl -fL -C - --retry 3 --speed-limit 10000 --speed-time 120 -o "$dest" "$url"; then
            return
        fi
        echo "attempt $attempt for $dest stopped; retrying"
        sleep 30
    done
    echo "could not download $url" >&2
    exit 1
}

cd "$OUT/prott5"
[ -f .done ] || { fetch https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip prot_t5_xl_uniref50.zip \
    && unzip -q -o prot_t5_xl_uniref50.zip && touch .done; }

cd "$OUT/plmsearch"
[ -f .done ] || { fetch https://zenodo.org/records/11480660/files/plmsearch_data.tar.gz/content plmsearch_data.tar.gz \
    && tar -xzf plmsearch_data.tar.gz plmsearch_data/model && touch .done; }

ls -la "$OUT"/*
du -sh "$OUT"/*
