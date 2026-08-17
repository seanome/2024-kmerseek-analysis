# Build kmerseek-spectra:latest from the olgabot/kmer-frequency-histogram branch.
#
# Build context must be the kmerseek repo root:
#   docker build -t kmerseek-spectra:latest \
#     --build-arg GIT_SHA=b2dfde27f368a4e99a73b429d0c772ce932fd9e3 \
#     -f nextflow-runs/kmer-spectra/Dockerfile \
#     /path/to/kmerseek

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:slim AS builder

ARG GIT_SHA=b2dfde27f368a4e99a73b429d0c772ce932fd9e3

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    git \
    libclang-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src/ src/
COPY benches/ benches/
COPY examples/ examples/

# Embed the source SHA for traceability.
RUN echo "${GIT_SHA}" > /build/GIT_SHA && \
    cargo build --release --no-default-features --bin kmerseek

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/kmerseek /usr/local/bin/kmerseek
COPY --from=builder /build/GIT_SHA /etc/kmerseek_git_sha

WORKDIR /data
