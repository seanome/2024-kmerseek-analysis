# kmerseek (cysteine-framework build) + python/polars in one image.
#
# Build context is the kmerseek repo root, not this repo:
#   docker build --platform linux/amd64 -t kmerseek-cys:latest \
#     --build-arg GIT_SHA=05b4131300b64b7b2c34d6dd1ec03743613cf2bf \
#     -f nextflow-runs/cys-frameworks/Dockerfile \
#     /Users/olga/code/kmerseek-cys-runs
#
# --platform linux/amd64 is mandatory: Sherlock compute nodes are amd64, and Apptainer
# refuses to run an arm64 image built on Apple Silicon.
#
# phmmer is not in here. It comes from its own pinned biocontainer, declared on
# phmmerSearch in main.nf, the same way the deep-twilight controls do it.
#
# The image holds whatever kmerseek commit is checked out in the build context. GIT_SHA
# only labels it, in /etc/kmerseek_git_sha. `make build-image` checks that the two agree.
#
# The kmerseek commit this image must hold is 05b4131, the same as KMERSEEK_SHA in the
# Makefile. That is c39370c plus PR 154, the Karlin-Altschul fit that counts regions per
# score bin. c39370c is v0.4.0, plus PR 141 (an index entry is keyed by its sequence
# rather than by its k-mer set), plus cys2 (PR 147) and cys_hp3 (PR 148). The Python
# steps still run on the older c39370c image (pipeline_image in nextflow.config), which
# is fine because they run no kmerseek. The key fix is not optional here. Without it two sequences with equal
# k-mer sets collapse into one index entry and only the first is stored, and under cys2
# that is the common case rather than the corner case, because cys2 is built so that
# peptides sharing a cysteine framework encode identically. Measured on the 125 test
# ribonucleases: a cys2 index held 94 of them before the fix and 121 after, where 121 is
# right because 4 of the 125 are duplicate sequences.

# ---- builder ---------------------------------------------------------------
FROM rust:slim AS builder

ARG GIT_SHA=05b4131300b64b7b2c34d6dd1ec03743613cf2bf

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

# Pinned exactly rather than floored, so the same tag rebuilt later is the same image.
# These are the versions the qfo-pfam-region-benchmark image carries, kept the same here
# so a script behaves the same in both.
RUN pip3 install --no-cache-dir --break-system-packages \
        'polars==1.43.2' 'pyarrow==25.0.1' 'numpy==2.5.2'

COPY --from=builder /build/target/release/kmerseek /usr/local/bin/kmerseek
COPY --from=builder /build/GIT_SHA /etc/kmerseek_git_sha

# No ENTRYPOINT: Nextflow runs /bin/bash inside the container and would not reach past one.
WORKDIR /data
