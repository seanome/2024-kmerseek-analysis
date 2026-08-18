# kmerseek (region-scoring branch) + python/polars in one image.
#
# The search arm and the scoring arm share this image so a Sherlock run pulls and
# converts one container instead of two. Baseline tools (HMMER, MMseqs2, HH-suite,
# Foldseek) come from their own pinned biocontainers, declared per process in main.nf.
#
# Build context is the kmerseek repo root:
#   docker build --platform linux/amd64 -t kmerseek-region:latest \
#     --build-arg GIT_SHA=0302511391c5f8eef499179430764d3d0881e529 \
#     -f nextflow-runs/qfo-pfam-region-benchmark/Dockerfile \
#     /Users/olga/code/kmerseek-region-scoring
#
# --platform linux/amd64 is mandatory: Sherlock compute nodes are amd64, and Apptainer
# refuses to run an arm64 image built on Apple Silicon.

# ---- builder ---------------------------------------------------------------
FROM rust:slim AS builder

ARG GIT_SHA=0302511391c5f8eef499179430764d3d0881e529

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
RUN pip3 install --no-cache-dir --break-system-packages \
        'polars>=1.0' 'pyarrow>=15' 'numpy>=1.26'

COPY --from=builder /build/target/release/kmerseek /usr/local/bin/kmerseek
COPY --from=builder /build/GIT_SHA /etc/kmerseek_git_sha

WORKDIR /data
