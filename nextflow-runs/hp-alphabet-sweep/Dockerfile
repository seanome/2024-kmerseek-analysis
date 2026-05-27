# Build kmerseek:0.3.0 from the olgabot/more-hp-alphabets branch.
#
# Build context must be the kmerseek repo root:
#   docker build -t kmerseek:0.3.0 \
#     --build-arg GIT_SHA=745dc5919762a698c0af9991361e1fa2ec992739 \
#     -f nextflow-runs/hp-alphabet-sweep/Dockerfile \
#     /path/to/kmerseek

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:slim AS builder

ARG GIT_SHA=745dc5919762a698c0af9991361e1fa2ec992739

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
