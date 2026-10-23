---
name: kmerseek-sherlock-source-build
description: Use when building kmerseek from source on Sherlock, particularly when bindgen cannot load libclang or PyO3 selects system Python 3.6.
---

Use `nextflow-runs/peptide-lookup/sbatch/build.sbatch` as the tested build recipe.
Keep its source SHA pin, compiler modules, Python interpreter, and runtime dependencies
when adapting it. Discover current module versions before a new build.

- `libclang.so` can fail to load because `libtinfo.so.6` is missing. Loading the
  `system ncurses/6.4` module resolved it; check with `ldd` before resubmission.
- PyO3 selected system Python 3.6 despite the Rust module being loaded. Load
  `python/3.12.1` and set `PYO3_PYTHON` to its absolute interpreter path.
- LLVM puts libc++ headers on CPATH. The recipe removes those entries while keeping
  LIBCLANG_PATH set. GCC must compile RocksDB against its own C++ headers.
- Build inside Slurm. Disposable Cargo artifacts can use scratch; the resulting
  binary is installed under GROUP_HOME/opt with the full source SHA and binary hash.

Observed 2026-10-22 on notebook 300. Jobs 49021979 and 49022302 failed on the missing
library and old Python; job 49022533 succeeded in 4m19s, with Slurm peak RSS 6.25 GiB.
This is a staging skill because this recipe has one completed source build.
