#!/usr/bin/env nextflow

/*
 * throughput_benchmark.nf
 *
 * Measures wall time, CPU usage, and peak RAM for Kmerseek, BLAST, and MMseqs2
 * on the same task: search N randomly-sampled SCOPe-40 query sequences against
 * the full SCOPe-40 target database (~15k proteins).
 *
 * All processes run in Docker containers (see nextflow.config for image tags).
 * Resource tracking is done inside each script via Python's `resource` module
 * (stdlib) because Nextflow's trace is accurate for container runs but the
 * in-process timing also provides sub-second precision and is captured even if
 * the Nextflow trace is not available.
 *
 * Script format: every timed process uses
 *     ${params.python3} << 'PYEOF'
 *     ${TIMING_HELPER}          ← Groovy-interpolated before bash sees the script
 *     ... process-specific code
 *     PYEOF
 * The single-quoted PYEOF prevents shell from expanding $vars; Groovy
 * interpolation (${...}) already ran at pipeline-definition time.
 *
 * Container images: defined in nextflow.config params block.
 * Build kmerseek:bench with the Dockerfile in this directory.
 *
 * Usage:
 *   cd nextflow-runs/throughput-benchmark
 *   /path/to/nextflow run main.nf
 *   /path/to/nextflow run main.nf --n_queries 1000 --resume
 */

nextflow.enable.dsl=2

// ── Parameters ────────────────────────────────────────────────────────────────

def home = System.getProperty('user.home')

params.fasta        = "${home}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir       = "${home}/data/scope/results-throughput-benchmark"

params.n_queries    = 1000
params.seed         = 42

// Kmerseek best-performing config from notebooks 075/076
params.km_encoding  = "hp-pbotc-1st-ed"
params.km_ksize     = 26
params.km_threshold = 0.01

// BLAST
params.blast_evalue = 1.0

// MMseqs2
params.mmseqs_sens  = 7.5

// Local binary paths (no Docker)
params.python3     = "${home}/anaconda3/envs/2025-kmerseek-analysis/bin/python3"
params.kmerseek    = "${home}/code/kmerseek/target/release/kmerseek"
params.blastp      = "${home}/anaconda3/envs/orthofinder/bin/blastp"
params.makeblastdb = "${home}/anaconda3/envs/orthofinder/bin/makeblastdb"
params.mmseqs      = "${home}/anaconda3/envs/orthofinder/bin/mmseqs"

// Known SCOPe40 target count (used in timing rows where n_targets is fixed)
def N_TARGETS = 15153


// ── Python timing helper (injected into every timed script via Groovy) ────────
//
// Provides run_timed() and write_timing_row(), which capture:
//   wall_seconds, cpu_user_seconds, cpu_sys_seconds, cpu_pct, peak_rss_mb
//
// resource.RUSAGE_CHILDREN is zero at process start (no prior children), so
// ru_maxrss after subprocess equals that tool's exact peak RSS.

def TIMING_HELPER = '''\
import platform, resource, subprocess, sys, time

TIMING_HEADER = (
    "method\\tphase\\tn_queries\\tn_targets\\t"
    "wall_seconds\\tqueries_per_second\\t"
    "cpu_user_seconds\\tcpu_sys_seconds\\tcpu_pct\\tpeak_rss_mb\\n"
)

def run_timed(cmd, **kwargs):
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0  = time.monotonic()
    r   = subprocess.run(cmd, **kwargs)
    t1  = time.monotonic()
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r, t1 - t0, ru0, ru1

def resource_stats(wall_s, ru0, ru1):
    cpu_user = ru1.ru_utime - ru0.ru_utime
    cpu_sys  = ru1.ru_stime - ru0.ru_stime
    cpu_pct  = (cpu_user + cpu_sys) / wall_s * 100 if wall_s > 0 else 0.0
    rss_raw  = ru1.ru_maxrss
    # macOS: bytes; Linux: kilobytes
    rss_mb   = rss_raw / 1024**2 if platform.system() == "Darwin" else rss_raw / 1024
    return cpu_user, cpu_sys, cpu_pct, rss_mb

def write_timing_row(path, method, phase, n_queries, n_targets, wall_s, ru0, ru1):
    cpu_user, cpu_sys, cpu_pct, rss_mb = resource_stats(wall_s, ru0, ru1)
    is_numeric = isinstance(n_queries, (int, float))
    qps_str    = f"{n_queries / wall_s:.2f}" if is_numeric and wall_s > 0 else "NA"
    n_q_str    = str(n_queries) if is_numeric else "NA"
    with open(path, "w") as f:
        f.write(TIMING_HEADER)
        f.write(
            f"{method}\\t{phase}\\t{n_q_str}\\t{n_targets}\\t"
            f"{wall_s:.3f}\\t{qps_str}\\t"
            f"{cpu_user:.3f}\\t{cpu_sys:.3f}\\t{cpu_pct:.1f}\\t{rss_mb:.1f}\\n"
        )
    print(
        f"{method} {phase}: wall={wall_s:.3f}s  "
        f"cpu={cpu_user + cpu_sys:.3f}s ({cpu_pct:.0f}%)  "
        f"rss={rss_mb:.0f} MB",
        flush=True,
    )
'''


// ── Process: subsample queries ────────────────────────────────────────────────

process SUBSAMPLE_QUERIES {
    tag "n=${params.n_queries}"
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path "scope40_query_subset_n${params.n_queries}.fa"

    script:
    """
    ${params.python3} << 'PYEOF'
    import random

    random.seed(${params.seed})

    records = []
    header, seq_parts = None, []
    with open("${fasta}") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header, seq_parts = line, []
            else:
                seq_parts.append(line)
        if header is not None:
            records.append((header, "".join(seq_parts)))

    n_total = len(records)
    subset  = random.sample(records, min(${params.n_queries}, n_total))

    out = "scope40_query_subset_n${params.n_queries}.fa"
    with open(out, "w") as fh:
        for hdr, seq in subset:
            fh.write(hdr + "\\n" + seq + "\\n")
    print(f"Sampled {len(subset)} / {n_total} sequences -> {out}", flush=True)
    PYEOF
    """
}


// ── Process: build kmerseek index (timed) ────────────────────────────────────

process KMERSEEK_INDEX {
    tag "k=${params.km_ksize}"
    storeDir "${params.outdir}/indices"
    publishDir params.outdir, mode: 'copy', pattern: 'timing_kmerseek_index.tsv'

    input:
    path fasta

    output:
    path "scope40.${params.km_encoding.replace('-','_')}.k${params.km_ksize}.rocksdb", type: 'dir', emit: index
    path "timing_kmerseek_index.tsv",                                                               emit: timing

    script:
    def db = "scope40.${params.km_encoding.replace('-','_')}.k${params.km_ksize}.rocksdb"
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}

    r, wall, ru0, ru1 = run_timed(
        ["${params.kmerseek}", "index",
         "--encoding", "${params.km_encoding}",
         "--ksize",    "${params.km_ksize}",
         "--scaled",   "1",
         "--input",    "${fasta}",
         "--output",   "${db}"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_kmerseek_index.tsv",
                     "kmerseek", "index", "NA", ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: time kmerseek search ─────────────────────────────────────────────

process KMERSEEK_SEARCH_TIMING {
    tag "k=${params.km_ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    path query_fa
    path target_db

    output:
    path "timing_kmerseek_search.tsv"

    script:
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}

    with open("${query_fa}") as fh:
        n_queries = sum(1 for line in fh if line.startswith(">"))

    r, wall, ru0, ru1 = run_timed(
        ["${params.kmerseek}", "search",
         "--encoding",  "${params.km_encoding}",
         "--ksize",     "${params.km_ksize}",
         "--threshold", "${params.km_threshold}",
         "--query",     "${query_fa}",
         "--target",    "${target_db}"],
        stdout=subprocess.DEVNULL, stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_kmerseek_search.tsv",
                     "kmerseek", "search", n_queries, ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: build BLAST database (timed) ─────────────────────────────────────

process BLAST_MAKEDB {
    tag "blast-db"
    storeDir "${params.outdir}/blast_db"
    publishDir params.outdir, mode: 'copy', pattern: 'timing_blast_makedb.tsv'

    input:
    path fasta

    output:
    path "scope40_blast_db.*",      emit: db
    path "timing_blast_makedb.tsv", emit: timing

    script:
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}

    r, wall, ru0, ru1 = run_timed(
        ["${params.makeblastdb}",
         "-in",       "${fasta}",
         "-dbtype",   "prot",
         "-out",      "scope40_blast_db",
         "-parse_seqids"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_blast_makedb.tsv",
                     "blast", "db_build", "NA", ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: time BLAST search ────────────────────────────────────────────────

process BLAST_SEARCH_TIMING {
    tag "blastp"
    publishDir params.outdir, mode: 'copy'

    input:
    path query_fa
    path db_files

    output:
    path "timing_blast_search.tsv"

    script:
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}
    from pathlib import Path

    with open("${query_fa}") as fh:
        n_queries = sum(1 for line in fh if line.startswith(">"))

    phr_files = list(Path(".").glob("scope40_blast_db*.phr"))
    if not phr_files:
        sys.stderr.write("ERROR: no scope40_blast_db*.phr found in work dir\\n")
        sys.exit(1)
    db_prefix = str(phr_files[0]).removesuffix(".phr")

    r, wall, ru0, ru1 = run_timed(
        ["${params.blastp}",
         "-query",       "${query_fa}",
         "-db",          db_prefix,
         "-evalue",      "${params.blast_evalue}",
         "-outfmt",      "6",
         "-num_threads", "1",
         "-out",         "/dev/null"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_blast_search.tsv",
                     "blast", "search", n_queries, ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: build MMseqs2 database (timed) ───────────────────────────────────

process MMSEQS_MAKEDB {
    tag "mmseqs-db"
    storeDir "${params.outdir}/mmseqs_db"
    publishDir params.outdir, mode: 'copy', pattern: 'timing_mmseqs_makedb.tsv'

    input:
    path fasta

    output:
    path "scope40_mmseqs_db*",       emit: db
    path "timing_mmseqs_makedb.tsv", emit: timing

    script:
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}

    r, wall, ru0, ru1 = run_timed(
        ["${params.mmseqs}", "createdb", "${fasta}", "scope40_mmseqs_db"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_mmseqs_makedb.tsv",
                     "mmseqs2", "db_build", "NA", ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: time MMseqs2 search ──────────────────────────────────────────────

process MMSEQS_SEARCH_TIMING {
    tag "mmseqs-search"
    publishDir params.outdir, mode: 'copy'

    input:
    path query_fa
    path db_files

    output:
    path "timing_mmseqs_search.tsv"

    script:
    """
    ${params.python3} << 'PYEOF'
    ${TIMING_HELPER}

    with open("${query_fa}") as fh:
        n_queries = sum(1 for line in fh if line.startswith(">"))

    # Build query db — not timed (preprocessing, analogous to kmerseek index)
    r = subprocess.run(
        ["${params.mmseqs}", "createdb", "${query_fa}", "query_mmseqs_db"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    r, wall, ru0, ru1 = run_timed(
        ["${params.mmseqs}", "search",
         "query_mmseqs_db", "scope40_mmseqs_db",
         "result_mmseqs",   "tmp_mmseqs",
         "--sensitivity",   "${params.mmseqs_sens}",
         "--threads",       "1"],
        stderr=sys.stderr,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    write_timing_row("timing_mmseqs_search.tsv",
                     "mmseqs2", "search", n_queries, ${N_TARGETS}, wall, ru0, ru1)
    PYEOF
    """
}


// ── Process: combine all timing results ───────────────────────────────────────

process COLLECT_TIMING {
    publishDir params.outdir, mode: 'copy'

    input:
    path timing_files

    output:
    path "throughput_benchmark.tsv"

    script:
    """
    ${params.python3} << 'PYEOF'
    import glob

    files = sorted(glob.glob("timing_*.tsv"))
    rows  = []
    for path in files:
        with open(path) as f:
            lines = f.readlines()
        rows.extend(lines[1:])   # skip per-file header

    header = (
        "method\\tphase\\tn_queries\\tn_targets\\t"
        "wall_seconds\\tqueries_per_second\\t"
        "cpu_user_seconds\\tcpu_sys_seconds\\tcpu_pct\\tpeak_rss_mb\\n"
    )
    with open("throughput_benchmark.tsv", "w") as out:
        out.write(header)
        out.writelines(rows)

    print("=== Throughput benchmark results ===")
    print(header.strip())
    for r in rows:
        print(r.strip())
    PYEOF
    """
}


// ── Workflow ──────────────────────────────────────────────────────────────────

workflow {
    fasta_ch = Channel.fromPath(params.fasta, checkIfExists: true)

    query_ch = SUBSAMPLE_QUERIES(fasta_ch)

    // Kmerseek
    KMERSEEK_INDEX(fasta_ch)
    km_search_timing_ch = KMERSEEK_SEARCH_TIMING(query_ch, KMERSEEK_INDEX.out.index)

    // BLAST
    BLAST_MAKEDB(fasta_ch)
    blast_search_timing_ch = BLAST_SEARCH_TIMING(query_ch, BLAST_MAKEDB.out.db.collect())

    // MMseqs2
    MMSEQS_MAKEDB(fasta_ch)
    mmseqs_search_timing_ch = MMSEQS_SEARCH_TIMING(query_ch, MMSEQS_MAKEDB.out.db.collect())

    // Collect all timing TSVs (build + search for every method)
    all_timing_ch = KMERSEEK_INDEX.out.timing
        .mix(km_search_timing_ch,
             BLAST_MAKEDB.out.timing,  blast_search_timing_ch,
             MMSEQS_MAKEDB.out.timing, mmseqs_search_timing_ch)
        .collect()

    COLLECT_TIMING(all_timing_ch)
}
