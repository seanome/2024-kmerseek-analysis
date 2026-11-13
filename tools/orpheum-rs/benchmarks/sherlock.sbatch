#!/bin/bash
#SBATCH --job-name=orpheum-rust-benchmark
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ml devel rust/1.90.0
export CARGO_HOME="$GROUP_HOME/opt/orpheum-rs/cargo"
export CARGO_TARGET_DIR="$L_SCRATCH_JOB/target"
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
CODE="$HOME/orpheum-rs-code"
cargo build --release --locked --jobs 4 --manifest-path "$CODE/Cargo.toml"
cargo test --locked --jobs 4 --manifest-path "$CODE/Cargo.toml"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/tests/check_reference.py" "$CARGO_TARGET_DIR/release/orpheum-rs"
/usr/bin/time -v "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/benchmarks/sherlock.py"
