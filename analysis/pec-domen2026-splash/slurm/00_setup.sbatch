#!/bin/bash
#SBATCH --job-name=nb361-setup
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
# Builds the pinned tool environment, the SPLASH 2.11.9 release binary and pec-tools,
# then records every version. Restartable: each part has a done-marker.
set -euo pipefail
: "${PEC_ROOT:?set PEC_ROOT}" "${PEC_CODE:?set PEC_CODE (the analysis/pec-domen2026-splash folder)}"
cd "$PEC_ROOT"
export MAMBA_ROOT_PREFIX=$PEC_ROOT/mamba
MM=$PEC_ROOT/bin/micromamba
ENV=$PEC_ROOT/env
mkdir -p versions
if [ ! -e env.done ]; then
  rm -rf "$ENV"
  $MM create -y -p "$ENV" -f "$PEC_CODE/environment-pec361.yml"
  $MM list -p "$ENV" --explicit --md5 > versions/env_explicit.txt
  $MM list -p "$ENV" > versions/env_list.txt
  touch env.done
fi
if [ ! -e splash.done ]; then
  V=2.11.9
  mkdir -p splash-$V && cd splash-$V
  curl -fsSL -o splash.tar.gz https://github.com/refresh-bio/SPLASH/releases/download/v$V/splash-$V.linux.x64.tar.gz
  sha256sum splash.tar.gz > ../versions/splash-$V.tar.gz.sha256
  tar xzf splash.tar.gz
  cd ..
  ln -sfn splash-$V/$(ls splash-$V | grep -v tar.gz | head -1) splash
  ls -la splash/ > versions/splash_files.txt
  ./splash/splash --version > versions/splash_version.txt 2>&1 || true
  touch splash.done
fi
# pec-tools is rebuilt whenever its source changes (stamp = sha256 of the source files)
SRC_HASH=$(cat "$PEC_CODE/pec-tools/Cargo.toml" "$PEC_CODE/pec-tools/src/main.rs" | sha256sum | cut -d' ' -f1)
if [ "$(cat pectools.done 2>/dev/null)" != "$SRC_HASH" ]; then
  rm -rf pec-tools && cp -r "$PEC_CODE/pec-tools" pec-tools
  cd pec-tools
  export CARGO_HOME=$PEC_ROOT/cargo
  # conda compilers set CC/CFLAGS that break cargo builds; this crate has no C deps, but unset them anyway
  unset CC CXX CFLAGS CXXFLAGS LDFLAGS
  "$ENV/bin/cargo" test --release 2>&1 | tee ../versions/pec-tools_cargo_test.txt
  grep -q "test result: ok" ../versions/pec-tools_cargo_test.txt
  "$ENV/bin/cargo" build --release
  cd ..
  "$ENV/bin/rustc" --version > versions/rustc.txt
  sha256sum pec-tools/target/release/pec-tools > versions/pec-tools.sha256
  echo "$SRC_HASH" > pectools.done
fi
{ for t in fastp sourmash diamond hmmscan seqkit python; do echo "== $t"; "$ENV/bin/$t" --version 2>&1 | head -2; done
  "$ENV/bin/python" -c 'import importlib.metadata as m; print({p: m.version(p) for p in ["pydeseq2","sourmash","sourmash_plugin_branchwater","polars","pandas","biopython","matplotlib","numpy","scipy"]})'
} > versions/tool_versions.txt 2>&1
cat versions/tool_versions.txt versions/splash_version.txt
