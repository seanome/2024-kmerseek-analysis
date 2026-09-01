# kmerseek:bench — kmerseek:0.3.0 runtime + python3 (for in-process timing scripts)
#
# Build (from kmerseek repo root, after building kmerseek:0.3.0):
#   docker build -t kmerseek:bench \
#     -f /path/to/nextflow-runs/throughput-benchmark/Dockerfile .
#
# Requires kmerseek:0.3.0 to be present locally or in a registry.

FROM kmerseek:0.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
    && rm -rf /var/lib/apt/lists/*
