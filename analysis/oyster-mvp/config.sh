# Shared paths and pinned versions for the oyster reference-free run.
# Sourced by every sbatch script. Code lives in $HOME, environments and tools
# in $GROUP_HOME, data in $SCRATCH (Sherlock storage rules).
ROOT="$SCRATCH/oyster-mvp-20261023"
CODE="$HOME/oyster-mvp-code"
OPT="$GROUP_HOME/opt/oyster-mvp"
MAMBA="$OPT/bin/micromamba"
SHAZAM_ENV="$OPT/envs/shazam"   # sourmash, biopython, pydeseq2, kallisto, cutadapt
DEKUPL_ENV="$OPT/envs/dekupl"   # python 3.11, snakemake 7.32.4, R + DESeq2, jellyfish
TRIMGALORE="$OPT/sw/TrimGalore-0.6.10/trim_galore"
SPLASH_DIR="$OPT/sw/splash-2.11.9"
DEKUPL_SRC="$OPT/sw/dekupl-run"
DEKUPL_COMMIT=172720c1d9b798f162bf9eba6808c20b65d27fc7
SAMPLES=(DW1 DW2 DW3 ZW1 ZW2 ZW3)
declare -A CONDITION=([DW1]=wild [DW2]=wild [DW3]=wild [ZW1]=breed [ZW2]=breed [ZW3]=breed)
# Deterministic prefixes: first N pairs of each ENA file, not random subsamples.
declare -A DEPTH_LINES=([1m]=4000000 [5m]=20000000 [full]=0)

# Run a stage under GNU time and keep wall, CPU and peak memory in one file.
# Usage: timed <stage-name> <cmd...>; writes $ROOT/timing/<stage-name>.time
timed() {
  local name=$1; shift
  mkdir -p "$ROOT/timing"
  /usr/bin/time -v -o "$ROOT/timing/$name.time" "$@"
}

# Same, for a shell function (GNU time can only run programs).
# Usage: timed_fn <stage-name> <function> <args...>
timed_fn() {
  local name=$1; shift
  export -f "$1"
  timed "$name" bash -c '"$@"' _ "$@"
}
