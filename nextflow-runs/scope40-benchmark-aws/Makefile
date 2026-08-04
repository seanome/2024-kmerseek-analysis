# scope40-benchmark-aws/Makefile
#
# Manage containers, data staging, and pipeline launches.
#
# Required one-time setup:
#   1. Set REGISTRY to your ECR registry or Docker Hub username
#   2. Build + push scope40-eval and kmerseek containers (see below)
#   3. Upload SCOPe40 data to S3 (make upload-data)
#   4. Configure Seqera Platform compute environments (CPU + GPU)
#
# Then run: make test-local  OR  make run-aws

# ─── Configuration ────────────────────────────────────────────────────────────

REGISTRY     ?= YOUR-ECR-REGISTRY.dkr.ecr.us-east-1.amazonaws.com
S3_BUCKET    ?= s3://YOUR-BUCKET-NAME
AWS_REGION   ?= us-east-1
GPU_QUEUE    ?= gpu          # AWS Batch GPU job queue name (Seqera compute env)
NF           := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

# Local data paths (for upload-data and local runs)
SCOPE_FASTA  := $(HOME)/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
PDBSTYLE_DIR := $(HOME)/data/scope/pdbstyle-2.08

# Container image names
EVAL_IMAGE      := $(REGISTRY)/scope40-eval:1.0
KMERSEEK_IMAGE  := $(REGISTRY)/kmerseek:0.3.1
KMERSEEK_SRC    := $(HOME)/code/kmerseek   # path to kmerseek repo
# Public BioContainers build — has foldseek on PATH + procps; no custom build needed
FOLDSEEK_IMAGE  := quay.io/biocontainers/foldseek:10.941cd33--h5021889_1

# S3 data paths
S3_FASTA     := $(S3_BUCKET)/scope40/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
S3_PDBSTYLE  := $(S3_BUCKET)/scope40/pdbstyle-2.08
S3_OUTDIR    := $(S3_BUCKET)/scope40/results

# ─── Container management ─────────────────────────────────────────────────────

.PHONY: build-eval push-eval build-kmerseek push-kmerseek build-all push-all

build-eval:
	docker build -f docker/Dockerfile.eval -t scope40-eval:1.0 -t $(EVAL_IMAGE) .

push-eval: build-eval
	docker push $(EVAL_IMAGE)

build-kmerseek:
	docker build -f $(KMERSEEK_SRC)/Dockerfile \
	    -t kmerseek:0.3.1 -t $(KMERSEEK_IMAGE) $(KMERSEEK_SRC)

push-kmerseek: build-kmerseek
	docker push $(KMERSEEK_IMAGE)

build-all: build-eval build-kmerseek

push-all: push-eval push-kmerseek
	@echo "All custom containers pushed to $(REGISTRY)"
	@echo "Public containers (FoldSeek, MMseqs2, BLAST+, DIAMOND) are pulled at runtime."
	@echo "PLMSearch + TEA containers: build from GitHub repos (see TODO notes in main.nf)."

# Authenticate Docker to ECR
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) \
	| docker login --username AWS --password-stdin $(REGISTRY)

# ─── Data staging ─────────────────────────────────────────────────────────────

.PHONY: upload-data upload-fasta upload-pdbstyle

upload-fasta:
	aws s3 cp $(SCOPE_FASTA) $(S3_FASTA)

upload-pdbstyle:
	@echo "Uploading pdbstyle-2.08 (~300 MB, ~15,000 files)..."
	aws s3 sync $(PDBSTYLE_DIR) $(S3_PDBSTYLE)/ \
	    --exclude ".DS_Store"

upload-data: upload-fasta upload-pdbstyle
	@echo "SCOPe40 data uploaded to $(S3_BUCKET)/scope40/"

# ─── Pipeline runs — LOCAL ────────────────────────────────────────────────────

.PHONY: test-local run-local

# 10×10 test (CPU tools only, local Docker)
test-local:
	$(NF) run main.nf \
	    --scope_fasta  $(SCOPE_FASTA) \
	    --pdbstyle_dir $(PDBSTYLE_DIR) \
	    --n_test       10 \
	    --outdir       results-test \
	    --eval_container    scope40-eval:1.0 \
	    --kmerseek_container kmerseek:0.3.1 \
	    --foldseek_container $(FOLDSEEK_IMAGE) \
	    -profile local \
	    -resume

# Full benchmark (CPU tools only, local)
run-local:
	$(NF) run main.nf \
	    --scope_fasta  $(SCOPE_FASTA) \
	    --pdbstyle_dir $(PDBSTYLE_DIR) \
	    --outdir       results-full \
	    --eval_container    $(EVAL_IMAGE) \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    --foldseek_container $(FOLDSEEK_IMAGE) \
	    -profile local \
	    -resume

# Kmerseek 0.4.0 sweep only (local binary, no container) — hp x k15-19 (6 variants),
# dayhoff x k10-20, protein x k5-15, all-vs-all self-search on SCOPe40. Other tools'
# AUCs come from a prior run-local/run-aws; this just adds/refreshes the kmerseek rows.
run-local-kmerseek-sweep:
	$(NF) run main.nf \
	    --scope_fasta  $(SCOPE_FASTA) \
	    --pdbstyle_dir $(PDBSTYLE_DIR) \
	    --outdir       results-kmerseek-v040-sweep \
	    --eval_container   $(EVAL_IMAGE) \
	    --run_foldseek false \
	    --run_mmseqs2  false \
	    --run_blast    false \
	    --run_diamond  false \
	    --run_kmerseek true \
	    -profile local \
	    -resume

# ─── Pipeline runs — AWS via Seqera Platform ─────────────────────────────────
#
# Prerequisites:
#   1. `pip install seqera` and `tw login`
#   2. Create compute environments in Tower:
#       - CPU env  (e.g. c5.4xlarge / r5.4xlarge)  → set TOWER_CPU_ENV
#       - GPU env  (e.g. p3.2xlarge / g4dn.xlarge) → set TOWER_GPU_ENV
#   3. make push-all  (push custom containers to ECR)
#   4. make upload-data  (upload SCOPe40 data to S3)

TOWER_WORKSPACE  ?= YOUR-WORKSPACE
TOWER_CPU_ENV    ?= scope40-cpu
TOWER_GPU_ENV    ?= scope40-gpu

.PHONY: test-aws run-aws run-aws-gpu

# 10×10 test on AWS (CPU tools only)
test-aws:
	$(NF) run main.nf \
	    --scope_fasta  $(S3_FASTA) \
	    --pdbstyle_dir $(S3_PDBSTYLE) \
	    --n_test       10 \
	    --outdir       $(S3_OUTDIR)-test \
	    --eval_container    $(EVAL_IMAGE) \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    --foldseek_container $(FOLDSEEK_IMAGE) \
	    --gpu_queue    $(GPU_QUEUE) \
	    -profile aws \
	    -resume

# Full benchmark — CPU tools only (FoldSeek, MMseqs2, BLAST+, DIAMOND, Kmerseek)
run-aws:
	$(NF) run main.nf \
	    --scope_fasta  $(S3_FASTA) \
	    --pdbstyle_dir $(S3_PDBSTYLE) \
	    --outdir       $(S3_OUTDIR) \
	    --eval_container    $(EVAL_IMAGE) \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    --foldseek_container $(FOLDSEEK_IMAGE) \
	    --gpu_queue    $(GPU_QUEUE) \
	    -profile aws \
	    -resume

# Full benchmark — all tools including GPU (PLMSearch + TEA + FoldSeek-EF)
# Requires verified container images for PLMSearch and TEA (see TODO in main.nf)
run-aws-gpu:
	$(NF) run main.nf \
	    --scope_fasta  $(S3_FASTA) \
	    --pdbstyle_dir $(S3_PDBSTYLE) \
	    --outdir       $(S3_OUTDIR) \
	    --eval_container    $(EVAL_IMAGE) \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    --foldseek_container $(FOLDSEEK_IMAGE) \
	    --run_plmsearch  true \
	    --run_foldseek_ef true \
	    --run_tea        true \
	    --gpu_queue    $(GPU_QUEUE) \
	    -profile aws \
	    -resume

# ─── Helpers ──────────────────────────────────────────────────────────────────

.PHONY: clean-work clean-results

clean-work:
	rm -rf work .nextflow .nextflow.log*

clean-results:
	rm -rf results-test results-full

# Print current polars version in eval container
check-eval-image:
	docker run --rm --entrypoint "" scope40-eval:1.0 python3 -c \
	    "import polars as pl; print('polars', pl.__version__)"
