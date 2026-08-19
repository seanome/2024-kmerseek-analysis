# Build kmerseek-spectra:latest from the olgabot/kmer-stats-review-fixes branch
# (olgabot/kmer-frequency-histogram was merged into this branch's history via PR #36).
#
# Build context must be the kmerseek repo root:
#   docker build -t kmerseek-spectra:latest \
#     --build-arg GIT_SHA=f006f821829e7354579267cc5336b341427105b3 \
#     -f nextflow-runs/kmer-spectra/Dockerfile \
#     /path/to/kmerseek
#
# GIT_SHA is a traceability label only (embedded via `echo "${GIT_SHA}" >
# /build/GIT_SHA` below) -- it does NOT check out that commit. The actual code
# built is whatever's in the build context directory at build time, so keep
# this value in sync with that directory's real HEAD (`git rev-parse HEAD`) by
# hand; it silently goes stale otherwise, which is exactly what happened to the
# previous default (pinned to a commit several fixes behind what was actually
# being built and deployed).

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:slim AS builder

ARG GIT_SHA=f006f821829e7354579267cc5336b341427105b3

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
