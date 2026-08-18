# qfo-hp-alphabet-sweep/Makefile
#
# 6 HP alphabets x 7 ksizes x 9 QfO reference species vs human.
#
# Local (bare binary, fast iteration):
#   make test-local   # 1 species, 1 encoding, 2 ksizes — sanity check
#   make run-local    # full sweep
#
# AWS (Seqera Platform, Fusion + Wave):
#   source ~/.config/seqera/seanome-kmerseek.env
#   make push-kmerseek   # one-time / after kmerseek changes
#   make test-aws
#   make run-aws

NF := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

# Account-specific values — override on the command line or in a gitignored
# .env; never hardcode the real registry/bucket/queue in this public repo.
REGISTRY       ?= YOUR-ECR-REGISTRY.dkr.ecr.us-west-2.amazonaws.com
AWS_REGION     ?= us-west-2
# olgabot/bump-version-0.4.0 branch
KMERSEEK_SRC   := $(HOME)/code/kmerseek
KMERSEEK_IMAGE := $(REGISTRY)/kmerseek:0.4.0

S3_BUCKET  ?= s3://YOUR-BUCKET-NAME/qfo-hp-alphabet-sweep
S3_OUTDIR  := $(S3_BUCKET)
S3_WORKDIR := $(S3_BUCKET)/work

.PHONY: test-local run-local test-aws run-aws build-kmerseek push-kmerseek ecr-login clean-work

# ─── Container (only needed for -profile aws) ────────────────────────────────

build-kmerseek:
	docker build -f $(KMERSEEK_SRC)/Dockerfile \
	    -t kmerseek:0.4.0 -t $(KMERSEEK_IMAGE) $(KMERSEEK_SRC)

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) \
	| docker login --username AWS --password-stdin $(REGISTRY)

push-kmerseek: build-kmerseek ecr-login
	docker push $(KMERSEEK_IMAGE)

# ─── Local runs (bare binary, no container) ──────────────────────────────────

test-local:
	$(NF) run main.nf -profile local,test -resume

run-local:
	$(NF) run main.nf -profile local -resume

# ─── AWS runs (Seqera Platform, Fusion + Wave) ───────────────────────────────
# Prereq: source ~/.config/seqera/seanome-kmerseek.env

test-aws:
	$(NF) run main.nf \
	    --outdir $(S3_OUTDIR)-test \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    -w $(S3_WORKDIR)-test \
	    -profile aws,test \
	    -resume

run-aws:
	$(NF) run main.nf \
	    --outdir $(S3_OUTDIR) \
	    --kmerseek_container $(KMERSEEK_IMAGE) \
	    -w $(S3_WORKDIR) \
	    -profile aws \
	    -resume

# ─── Helpers ──────────────────────────────────────────────────────────────────

clean-work:
	/bin/rm -rf work .nextflow .nextflow.log*
