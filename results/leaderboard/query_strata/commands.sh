# Compute the leaderboard's query-side strata for your own proteome.
# These are the commands scripts/query_strata/0*.sbatch ran on the 998 human queries, with
# the Sherlock paths replaced by the three names below. Bins: RULES.md next to this file.
QUERY=query.faa          # the proteins you want to annotate
TARGET=target.faa        # the annotated proteome you copy annotations from (one per relative)
PROTEOME=proteome.faa    # your whole proteome, for BUSCO

# 1. Identity to the nearest protein in the target: best hit per query by E-value, column 3.
#    MMseqs2 from quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef
mmseqs easy-search "$QUERY" "$TARGET" hits.m8 tmp -s 7.0 -e 1e-3 --threads 8
sort -t $'\t' -k1,1 -k11,11g -k12,12gr -k2,2 hits.m8 | awk -F'\t' '$1 != prev {print; prev = $1}' > best_hits.m8

# 2. Membrane class: DeepTMHMM's predicted type (TM, SP+TM, BETA, SP, GLOB) per protein.
#    Image docker.io/deeptmhmm/deeptmhmm@sha256:e527883fd2114007c6208c3d764fece40016cc95e209eab93016644c3e7ccb16
for f in deeptmhmm_cv_0.model deeptmhmm_cv_1.model deeptmhmm_cv_2.model deeptmhmm_cv_3.model deeptmhmm_cv_4.model esm_model_alphabet.pt esm_model_args.pt esm_model_state_dict.pt; do ln -sfn "/openprotein/$f" "$f"; done
apptainer exec --writable-tmpfs --bind "$PWD" --pwd "$PWD" docker://deeptmhmm/deeptmhmm@sha256:e527883fd2114007c6208c3d764fece40016cc95e209eab93016644c3e7ccb16 python3 /openprotein/predict.py --fasta "$QUERY"

# 3. Low complexity: residues SEG masks (NCBI segmasker, SEG's defaults), divided by length.
#    segmasker from quay.io/biocontainers/blast@sha256:9dfc69f990c0aeb936276ee591ed32919a79f46dfa34060055e05a050a17959c (BLAST+ 2.16.0)
segmasker -in "$QUERY" -infmt fasta -outfmt interval -out query.seg.intervals

# 4. Length of each protein. (Region length is end - start + 1 of each region you score.)
awk '/^>/{if(id)print id"\t"n; id=substr($1,2); n=0; next}{n+=length($0)}END{print id"\t"n}' "$QUERY" > query.length.tsv

# 5. Pfam family breadth: find the Pfam families on your proteins, then look each one up in
#    pfam_family_breadth.tsv (superkingdoms and phyla, from Pfam 38's reference proteomes).
#    The leaderboard's own regions come from Pfam's region table, so this hmmscan step was
#    not run for it; the lookup table was made by scripts/query_strata/05_pfam_breadth.sbatch.
hmmscan --cut_ga --domtblout query.domtbl Pfam-A.hmm "$QUERY"

# 6. Gene-model completeness: BUSCO on the whole proteome; read each protein's rows of
#    full_table.tsv (Complete or Duplicated = complete; Fragmented only = fragmented).
#    BUSCO 6.1.0 from quay.io/biocontainers/busco@sha256:acf4e2f5bf6eb5c195b4165b4261e233974df5b23795319e2e4185569c9c903c
busco --download primates_odb12 --download_path busco_downloads
busco -i "$PROTEOME" -m proteins -l primates_odb12 -c 8 -o busco_out --offline --download_path busco_downloads
