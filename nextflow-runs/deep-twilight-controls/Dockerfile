# kmerseek (branch olgabot/extend-without-fit-ka-fixes, on olgabot/run-evalue) + python/polars.
#
# Copied from nextflow-runs/qfo-pfam-region-benchmark/Dockerfile; only the kmerseek commit
# differs. Build context is the kmerseek repo root, at the commit named by GIT_SHA:
#   make push-image    (see the Makefile for the tag)
#
# --platform linux/amd64 is mandatory: Sherlock compute nodes are amd64, and Apptainer
# refuses to run an arm64 image built on Apple Silicon.

# ---- builder ---------------------------------------------------------------
FROM rust:slim AS builder

ARG GIT_SHA=90c581a011d42002eeeb204ec6efea79ece0359d

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang cmake git libclang-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src/ src/
COPY benches/ benches/
COPY examples/ examples/

RUN echo "${GIT_SHA}" > /build/GIT_SHA && \
    cargo build --release --no-default-features --bin kmerseek

# ---- runtime ---------------------------------------------------------------
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates procps zstd python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Debian trixie marks its python as externally managed. This image has no system python
# tooling to protect, so installing straight into it is simpler than carrying a venv and
# keeping every process script aware of its path.
# numpy is required, not incidental: bin/cafa_metrics.py builds the Fmax/Smin threshold
# grid with it. polars does not guarantee pulling it in.
# Pinned exactly, not floored. `polars>=1.0` meant the image's polars floated with
# whatever was newest at build time, so the same IMAGE_TAG rebuilt later was a different
# image -- the same class of problem as IMAGE_TAG drifting from the config literal, one
# layer down.
#
# It was already biting: polars 1.43 warns that str.split's `empty_as_null` default flips
# in 2.0, while polars 1.34 on the analysis Mac does not even accept that argument and
# already splits the 2.0 way. So the seven .str.split() calls in bin/ -- on HGNC
# uniprot_ids, on sp|ACC|NAME accessions, and on versioned Pfam ids -- had different
# empty-field semantics in the container than outside it, silently.
#
# These are the versions the 2026-08-24 image actually contains, read out of the running
# image rather than chosen, so pinning them changes nothing about what runs today and
# makes a rebuild reproduce it instead of drifting.
# polars[rtcompat]: the build that does not need AVX2, so the same image runs under the
# amd64 emulation of an Apple Silicon laptop (`make smoke`) as well as on Sherlock. The
# plain build segfaults there on its first query.
RUN pip3 install --no-cache-dir --break-system-packages \
        'polars[rtcompat]==1.43.2' 'pyarrow==25.0.1' 'numpy==2.5.2'

COPY --from=builder /build/target/release/kmerseek /usr/local/bin/kmerseek
COPY --from=builder /build/GIT_SHA /etc/kmerseek_git_sha

WORKDIR /data
