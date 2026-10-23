#!/bin/bash
#SBATCH --job-name=oyster-mvp-setup
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out
# Install the pinned tools once: two conda environments, Trim Galore! 0.6.10,
# SPLASH 2.11.9, dekupl-run at commit 172720c and its four C helpers.
# Restartable: each piece is skipped when its marker file exists.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
mkdir -p "$OPT/sw" "$OPT/envs" "$ROOT/software" "$(dirname "$MAMBA")"
export MAMBA_ROOT_PREFIX="$OPT/mamba-root"
export PIP_CACHE_DIR="$L_SCRATCH_JOB/pip-cache"

[ -x "$MAMBA" ] || install -m 755 "$SCRATCH/299-omrgc-pilot/bin/micromamba" "$MAMBA"
"$MAMBA" --version

if [ ! -f "$SHAZAM_ENV/.complete" ]; then
  rm -rf "$SHAZAM_ENV"
  "$MAMBA" create -y -p "$SHAZAM_ENV" --strict-channel-priority -c conda-forge -c bioconda \
    python=3.12 sourmash=4.9.4 kallisto=0.51.1 pigz pandas scipy numpy pytest \
    scikit-learn anndata matplotlib formulaic
  "$SHAZAM_ENV/bin/python" -m pip install --no-cache-dir --only-binary=:all: biopython==1.88 cutadapt==5.2 pydeseq2==0.5.4
  "$SHAZAM_ENV/bin/python" -m pip install --no-cache-dir -e "$CODE/clam-context"
  touch "$SHAZAM_ENV/.complete"
fi

if [ ! -f "$DEKUPL_ENV/.complete" ]; then
  rm -rf "$DEKUPL_ENV"
  "$MAMBA" create -y -p "$DEKUPL_ENV" --strict-channel-priority -c conda-forge -c bioconda \
    python=3.11 r-base=4.3 bioconductor-deseq2=1.42 r-data.table r-pheatmap r-foreach \
    r-doparallel r-rcolorbrewer kmer-jellyfish=2.3 pigz cmake make zlib wget coreutils
  "$DEKUPL_ENV/bin/python" -m pip install --no-cache-dir --prefer-binary snakemake==7.32.4 "pulp<2.8"
  touch "$DEKUPL_ENV/.complete"
fi

cd "$OPT/sw"
if [ ! -x "$TRIMGALORE" ]; then
  rm -rf TrimGalore-0.6.10
  git clone --depth 1 --branch 0.6.10 https://github.com/FelixKrueger/TrimGalore.git TrimGalore-0.6.10
fi

if [ ! -x "$SPLASH_DIR/splash" ]; then
  rm -rf "$SPLASH_DIR" splash.tar.gz
  wget -q -O splash.tar.gz https://github.com/refresh-bio/SPLASH/releases/download/v2.11.9/splash-2.11.9.linux.x64.tar.gz
  sha256sum splash.tar.gz > "$ROOT/software/splash-2.11.9.tar.gz.sha256"
  mkdir -p "$SPLASH_DIR" && tar -xzf splash.tar.gz -C "$SPLASH_DIR" --strip-components=1
  rm splash.tar.gz
fi

if [ ! -f "$DEKUPL_SRC/.complete" ]; then
  rm -rf "$DEKUPL_SRC"
  git clone https://github.com/Transipedia/dekupl-run.git "$DEKUPL_SRC"
  cd "$DEKUPL_SRC"
  git checkout "$DEKUPL_COMMIT"
  git submodule update --init
  for d in joinCounts mergeTags computeNF TtestFilter; do make -C "share/$d"; done
  touch .complete
  cd "$OPT/sw"
fi

# Record what was installed, then check each tool runs.
{
  echo "micromamba $("$MAMBA" --version)"
  "$SHAZAM_ENV/bin/python" - <<'EOF'
import importlib.metadata as m
for p in ["sourmash", "biopython", "cutadapt", "pydeseq2", "clam-context", "pandas", "scipy"]:
    print(p, m.version(p))
EOF
  "$SHAZAM_ENV/bin/kallisto" version
  PATH="$SHAZAM_ENV/bin:$PATH" "$TRIMGALORE" --version | grep -i version
  "$DEKUPL_ENV/bin/jellyfish" --version
  "$DEKUPL_ENV/bin/snakemake" --version
  "$DEKUPL_ENV/bin/Rscript" -e 'cat("DESeq2", as.character(packageVersion("DESeq2")), "\n")'
  echo "dekupl-run $(git -C "$DEKUPL_SRC" rev-parse HEAD)"
  git -C "$DEKUPL_SRC" submodule status
  "$SPLASH_DIR/splash" --version 2>&1 | head -3
} | tee "$ROOT/software/versions.txt"
"$MAMBA" list -p "$SHAZAM_ENV" --explicit > "$ROOT/software/shazam-explicit.txt"
"$MAMBA" list -p "$DEKUPL_ENV" --explicit > "$ROOT/software/dekupl-explicit.txt"
"$SHAZAM_ENV/bin/python" -m pytest -q "$CODE/clam-context/tests" | tail -3
touch "$ROOT/setup.complete"
