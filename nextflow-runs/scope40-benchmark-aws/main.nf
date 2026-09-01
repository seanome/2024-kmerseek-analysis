#!/usr/bin/env nextflow

/*
 * scope40-benchmark-aws/main.nf
 *
 * SCOPe40 2.08 all-vs-all benchmark — 7 homology detection methods compared
 * on a consistent dataset with a shared sensitivity-to-first-FP evaluation.
 *
 * Tools:
 *   FoldSeek    — structure-based, real PDB (pdbstyle-2.08)
 *   FoldSeek-EF — FoldSeek on ESMFold-predicted structures (GPU)
 *   MMseqs2     — sequence-based, most sensitive preset
 *   BLAST+      — blastp, standard reference
 *   DIAMOND     — ultra-sensitive blastp
 *   Kmerseek    — k-mer / HP-alphabet, thomas_dill k=26
 *   PLMSearch   — ESM2 embeddings + FAISS search (GPU)
 *   TEA         — Template-free Evolutionary Alignment on ESMFold structures (GPU)
 *
 * Evaluation: TEA/FoldSeek-paper convention — sensitivity-to-first-cross-fold-FP
 *   at family / superfamily / fold levels → .rocx files → mean AUC summary
 *
 * AWS execution: AWS Batch via Seqera Platform (Tower)
 *   See Makefile for data-staging and launch commands.
 *
 * Test mode (10 × 10):
 *   --n_test 10   takes the first 10 sequences from the SCOPe40 FASTA
 */

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

params.scope_fasta  = null       // local path or S3 URI; null → auto-download
params.pdbstyle_dir = null       // path to pdbstyle-2.08/; null → auto-download
params.outdir       = 'results'
params.evalue_report = 10.0

// Test mode: 0 = full benchmark; >0 = use first N sequences
params.n_test = 0

// Tool switches (all on by default; GPU tools default off for local dev)
params.run_foldseek    = true
params.run_foldseek_ef = false    // FoldSeek on ESMFold structures; needs GPU
params.run_mmseqs2     = true
params.run_blast       = true
params.run_diamond     = true
params.run_kmerseek    = true
params.run_plmsearch   = false    // needs GPU + PLMSearch container
params.run_tea         = false    // needs GPU + TEA container

// Kmerseek — kmerseek 0.4.0 (native --min-shared-kmers/--max-pvalue filtering at the
// source, so no huge unfiltered result files).
//
// params.kmerseek_sweep == true (default, local dev):
//   hp variants (6) x k=15-19   — HP alphabet is the priority; matches the hp-v040
//                                  human-mouse ortholog sweep exactly
//   dayhoff     x k=10-20
//   protein     x k=5-15
//   Same ksize ranges as ../human-mouse-gencode-orthologs{,-hp-v040}/*.nf.
//
// params.kmerseek_sweep == false:
//   single (params.kmerseek_encoding, params.kmerseek_k) run — e.g. the AWS mini-test.
params.kmerseek_sweep    = true
params.kmerseek_encoding = 'hp-thomas-dill'
params.kmerseek_k        = 26
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_pvalue       = 0.05
// Absolute path to the kmerseek 0.4.0 binary (olgabot/min-shared-kmers-filter branch).
// Local profile: bare host path (no container, matches sibling human-mouse pipelines).
// AWS profile: overridden to 'kmerseek' (on $PATH inside params.kmerseek_container) —
// see the `aws` profile block in nextflow.config.
params.kmerseek_bin = "/Users/olga/code/kmerseek-min-shared-kmers-filter/target/release/kmerseek"

// ESMFold chunking — sequences per GPU job (smaller = more parallelism)
params.esmfold_chunk_size = 50

// Container images — override via --eval_container etc. or set in config
params.eval_container      = 'scope40-eval:1.0'
params.kmerseek_container  = 'kmerseek:0.4.0'
// BioContainers build — unlike ghcr.io/steineggerlab/foldseek, has foldseek
// on $PATH and includes procps, both of which Nextflow's docker executor needs.
params.foldseek_container  = 'quay.io/biocontainers/foldseek:10.941cd33--h5021889_1'
// PLMSearch: https://github.com/maovshao/PLMSearch
// TODO: verify public Docker image once available; build from repo if needed
params.plmsearch_container = 'ghcr.io/maovshao/plmsearch:latest'
// ESMFold: Meta's fair-esm package; use official or community image
params.esmfold_container   = 'ghcr.io/facebookresearch/esm:esmfold'
// TEA: https://github.com/GaoTongLab/TEA  (Guo et al. 2023)
// TODO: verify public Docker image; build from repo if needed
params.tea_container       = 'ghcr.io/gaotonglab/tea:latest'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

def isS3(p) { p?.toString().startsWith('s3://') }

// ---------------------------------------------------------------------------
// DATA STAGING
// ---------------------------------------------------------------------------

process downloadScopeFasta {
    label 'download'
    storeDir "${params.outdir}/.cache"

    output:
    path "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"

    script:
    """
    curl -sfL \
        "http://scop.berkeley.edu/downloads/scopseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa.gz" \
    | gunzip > astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
    n=\$(grep -c '^>' astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa)
    echo "Downloaded \${n} SCOPe40 domains"
    """
}

process downloadPdbstyle {
    label 'download'
    storeDir "${params.outdir}/.cache"

    output:
    path "pdbstyle-2.08"

    script:
    """
    mkdir -p pdbstyle-2.08
    curl -sfL \
        "http://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz" \
    | tar -xzf - --strip-components=1 -C pdbstyle-2.08
    n=\$(find pdbstyle-2.08 -name "*.ent" | wc -l | tr -d ' ')
    echo "Downloaded \${n} pdbstyle structures"
    """
}

process subsetFasta {
    // Test mode: take first N sequences from the FASTA
    container params.eval_container
    label 'light_cpu'

    input:
    path fasta
    val  n

    output:
    path "subset.fa"

    script:
    """
    #!/usr/bin/env python3
    count = 0
    with open("${fasta}") as fi, open("subset.fa", "w") as fo:
        for line in fi:
            if line.startswith(">"):
                count += 1
                if count > ${n}:
                    break
            if count <= ${n}:
                fo.write(line)
    print(f"Wrote {count} sequences to subset.fa")
    """
}

process parseScopeHeaders {
    // Parse FASTA headers → scope_domains.tsv (domain_id, scop hierarchy)
    container params.eval_container
    label 'light_cpu'
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path "scope_domains.tsv", emit: domains

    script:
    """
    #!/usr/bin/env python3
    import re
    pat = re.compile(r'^>(\\S+)\\s+(\\S+)')
    with open("${fasta}") as fi, open("scope_domains.tsv", "w") as fo:
        fo.write("domain_id\\tscop_id\\tscop_class\\tscop_fold\\tscop_superfamily\\tscop_family\\n")
        for line in fi:
            if not line.startswith(">"):
                continue
            m = pat.match(line.rstrip())
            if not m:
                continue
            did, sid = m.group(1), m.group(2)
            parts = sid.split(".")
            cls  = parts[0] if len(parts) >= 1 else ""
            fold = ".".join(parts[:2]) if len(parts) >= 2 else cls
            sfam = ".".join(parts[:3]) if len(parts) >= 3 else fold
            fo.write(f"{did}\\t{sid}\\t{cls}\\t{fold}\\t{sfam}\\t{sid}\\n")
    """
}

process collectPdbstyleStructures {
    // Copy pdbstyle .ent files matching the FASTA domain IDs → structures/
    // Filtered automatically: in test mode only N domains are in the FASTA.
    container params.eval_container
    label 'light_cpu'

    input:
    path pdbstyle_dir
    path fasta

    output:
    path "structures"

    script:
    """
    #!/usr/bin/env python3
    import os, re, shutil

    domain_ids = set()
    with open("${fasta}") as f:
        for line in f:
            if line.startswith(">"):
                m = re.match(r">(\\S+)", line)
                if m:
                    domain_ids.add(m.group(1))

    os.makedirs("structures", exist_ok=True)
    found = 0
    for root, _, files in os.walk("${pdbstyle_dir}"):
        for fn in files:
            if fn.endswith(".ent"):
                did = fn[:-4]
                if did in domain_ids:
                    shutil.copy2(os.path.join(root, fn), f"structures/{did}.pdb")
                    found += 1

    print(f"Collected {found}/{len(domain_ids)} structures from pdbstyle")
    if found == 0:
        raise RuntimeError("No structures found in pdbstyle_dir for the given FASTA")
    """
}

// ---------------------------------------------------------------------------
// FOLDSEEK — structure-based, real PDB (pdbstyle-2.08)
// ---------------------------------------------------------------------------

process foldseekSearch {
    container { params.foldseek_container }
    label 'high_cpu'
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path structures

    output:
    tuple val('foldseek'), path("foldseek_scope40.tsv.gz")

    script:
    """
    mkdir -p foldseek_tmp
    foldseek easy-search \\
        ${structures} \\
        ${structures} \\
        foldseek_scope40.tsv \\
        foldseek_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads       ${task.cpus} \\
        -e              ${params.evalue_report} \\
        --exhaustive-search 0
    gzip -c foldseek_scope40.tsv > foldseek_scope40.tsv.gz
    rm -f foldseek_scope40.tsv
    rm -rf foldseek_tmp
    """
}

// ---------------------------------------------------------------------------
// MMSEQS2 — sequence-based
// ---------------------------------------------------------------------------

process mmseqs2Search {
    container 'quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0'
    label 'high_cpu'
    publishDir "${params.outdir}/mmseqs2", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path fasta

    output:
    tuple val('mmseqs2'), path("mmseqs2_scope40.tsv.gz")

    script:
    """
    mkdir -p mmseqs2_tmp
    mmseqs easy-search \\
        ${fasta} \\
        ${fasta} \\
        mmseqs2_scope40.tsv \\
        mmseqs2_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus} \\
        -e        ${params.evalue_report} \\
        -s        7.5
    gzip -c mmseqs2_scope40.tsv > mmseqs2_scope40.tsv.gz
    rm -f mmseqs2_scope40.tsv
    rm -rf mmseqs2_tmp
    """
}

// ---------------------------------------------------------------------------
// BLAST+ — blastp, canonical reference
// ---------------------------------------------------------------------------

process blastMakeDB {
    container 'ncbi/blast:2.16.0'
    label 'light_cpu'

    input:
    path fasta

    output:
    path "blast_db"

    script:
    """
    mkdir -p blast_db
    makeblastdb -in ${fasta} -dbtype prot -out blast_db/scope40 -title scope40
    """
}

process blastSearch {
    container 'ncbi/blast:2.16.0'
    label 'high_cpu'
    publishDir "${params.outdir}/blast", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path fasta
    path blast_db

    output:
    tuple val('blast'), path("blast_scope40.tsv.gz")

    script:
    // outfmt 6: qseqid sseqid bitscore evalue (reordered for evaluator)
    """
    blastp \\
        -query          ${fasta} \\
        -db             ${blast_db}/scope40 \\
        -outfmt         "6 qseqid sseqid bitscore evalue" \\
        -evalue         ${params.evalue_report} \\
        -num_threads    ${task.cpus} \\
        -max_target_seqs 10000 \\
        -out            blast_scope40.tsv
    gzip -c blast_scope40.tsv > blast_scope40.tsv.gz
    rm -f blast_scope40.tsv
    """
}

// ---------------------------------------------------------------------------
// DIAMOND — ultra-sensitive blastp
// ---------------------------------------------------------------------------

process diamondMakeDB {
    container 'quay.io/biocontainers/diamond:2.1.24--hf93d47f_0'
    label 'light_cpu'

    input:
    path fasta

    output:
    path "scope40.dmnd"

    script:
    """
    diamond makedb --in ${fasta} --db scope40 --threads ${task.cpus}
    """
}

process diamondSearch {
    container 'quay.io/biocontainers/diamond:2.1.24--hf93d47f_0'
    label 'high_cpu'
    publishDir "${params.outdir}/diamond", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path fasta
    path diamond_db

    output:
    tuple val('diamond'), path("diamond_scope40.tsv.gz")

    script:
    """
    diamond blastp \\
        --query           ${fasta} \\
        --db              ${diamond_db} \\
        --ultra-sensitive \\
        --outfmt 6 qseqid sseqid bitscore evalue \\
        --evalue          ${params.evalue_report} \\
        --max-target-seqs 0 \\
        --threads         ${task.cpus} \\
        --block-size      2 \\
        --out             /dev/stdout \\
    | gzip -c > diamond_scope40.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// KMERSEEK — 0.4.0 local binary sweep: HP (6 variants) x k=15-19, dayhoff x
// k=10-20, protein x k=5-15, all-vs-all self-search on the SCOPe40 FASTA.
// Native --min-shared-kmers/--max-pvalue filtering keeps raw results small,
// so (unlike the other tools here) there's no separate DB-build container —
// just the local kmerseek 0.4.0 binary called directly.
// ---------------------------------------------------------------------------

process kmerseekIndex {
    tag "${encoding}_k${ksize}"
    storeDir "${params.outdir}/.cache/kmerseek_indices"
    label 'medium_cpu'

    input:
    tuple path(fasta), val(encoding), val(ksize)

    output:
    tuple val(encoding), val(ksize),
        path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir'),
        path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.index.log")

    script:
    // kmerseek normalises the encoding to underscores when auto-naming the database
    // (e.g. hp-thomas-dill -> hp_thomas_dill) — both declared outputs must use that
    // normalised name (see sibling hp-v040 pipeline for why storeDir needs this exactly).
    def enc_fname = encoding.replace('-', '_')
    def log_file  = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.index.log"
    def db_dir    = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.rocksdb"
    """
    echo "=== Indexing (kmerseek 0.4.0 local): scope40 ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    ${params.kmerseek_bin} index \\
        --encoding ${encoding} \\
        --ksize    ${ksize} \\
        --scaled   1 \\
        --input    ${fasta} \\
        2>&1 | tee -a ${log_file} || true

    # Ensure output directory exists even if k-mer space is saturated
    mkdir -p ${db_dir}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    """
}

process kmerseekSearch {
    tag "${encoding}_k${ksize}"
    label 'medium_cpu'

    input:
    tuple val(encoding), val(ksize), path(fasta), path(kmerseek_db), path(index_log)

    output:
    tuple val(encoding), val(ksize), path("kmerseek_scope40.${encoding}.k${ksize}.raw.csv.gz")

    script:
    def raw_csv = "kmerseek_scope40.${encoding}.k${ksize}.raw.csv.gz"
    """
    ${params.kmerseek_bin} search \\
        --query             ${fasta} \\
        --target            ${kmerseek_db} \\
        --encoding          ${encoding} \\
        --ksize             ${ksize} \\
        --scaled            1 \\
        --threshold         ${params.threshold} \\
        --min-shared-kmers  ${params.min_shared_kmers} \\
        --max-pvalue        ${params.max_pvalue} \\
    | gzip > ${raw_csv}
    """
}

process formatKmerseekResults {
    tag "${encoding}_k${ksize}"
    container params.eval_container
    label 'light_cpu'
    publishDir "${params.outdir}/kmerseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(encoding), val(ksize), path(raw_csv_gz)

    output:
    tuple val("kmerseek_${encoding.replace('-', '_')}_k${ksize}"), path("kmerseek_${encoding.replace('-', '_')}_k${ksize}.tsv.gz")

    script:
    def label = "kmerseek_${encoding.replace('-', '_')}_k${ksize}"
    """
    format_kmerseek_results.py ${raw_csv_gz} ${label}.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PLMSEARCH — ESM2 embeddings + FAISS nearest-neighbour search (GPU)
// Paper: Liu et al. 2023, https://doi.org/10.1093/bioinformatics/btad485
// GitHub: https://github.com/maovshao/PLMSearch
// TODO: update container image once a public image is confirmed
// ---------------------------------------------------------------------------

process plmsearchEmbed {
    // Pre-compute ESM2 embeddings for all sequences.
    container { params.plmsearch_container }
    label 'gpu'

    input:
    path fasta

    output:
    path "scope40_embeddings"

    script:
    """
    mkdir -p scope40_embeddings
    python PLMSearch.py embed \\
        --fasta  ${fasta} \\
        --output scope40_embeddings \\
        --device cuda
    """
}

process plmsearchSearch {
    container { params.plmsearch_container }
    label 'gpu'
    publishDir "${params.outdir}/plmsearch", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path embeddings
    path fasta

    output:
    tuple val('plmsearch'), path("plmsearch_scope40.tsv.gz")

    script:
    """
    python PLMSearch.py search \\
        --query  ${embeddings} \\
        --target ${embeddings} \\
        --output plmsearch_scope40.tsv \\
        --device cuda
    gzip -c plmsearch_scope40.tsv > plmsearch_scope40.tsv.gz
    rm -f plmsearch_scope40.tsv
    """
}

// ---------------------------------------------------------------------------
// ESMFOLD — structure prediction (shared by FoldSeek-EF and TEA)
// ---------------------------------------------------------------------------

process chunkFasta {
    // Split FASTA into chunks of params.esmfold_chunk_size for parallel GPU jobs
    container params.eval_container
    label 'light_cpu'

    input:
    path fasta

    output:
    path "chunks/chunk_*.fa"

    script:
    """
    #!/usr/bin/env python3
    import os
    os.makedirs("chunks", exist_ok=True)
    chunk_size = ${params.esmfold_chunk_size}

    seqs, buf = [], []
    with open("${fasta}") as f:
        for line in f:
            if line.startswith(">") and buf:
                seqs.append("".join(buf))
                buf = [line]
            else:
                buf.append(line)
    if buf:
        seqs.append("".join(buf))

    n_chunks = 0
    for i in range(0, len(seqs), chunk_size):
        with open(f"chunks/chunk_{i // chunk_size:04d}.fa", "w") as out:
            out.writelines(seqs[i:i + chunk_size])
        n_chunks += 1

    print(f"Split {len(seqs)} sequences into {n_chunks} chunks of {chunk_size}")
    """
}

process esmfoldPredict {
    // Run ESMFold on one chunk; outputs a .pdb per sequence.
    // Container: fair-esm package (pip install fair-esm), needs CUDA.
    // Outputs named after FASTA header (domain ID), e.g. d1dlwa_.pdb
    container { params.esmfold_container }
    label 'gpu'

    input:
    tuple val(chunk_id), path(chunk_fasta)

    output:
    path "${chunk_id}_structs/*.pdb"

    script:
    """
    mkdir -p ${chunk_id}_structs
    python -m esm.scripts.fold \\
        -i  ${chunk_fasta} \\
        -o  ${chunk_id}_structs/ \\
        --cpu-offload
    """
}

process mergeEsmfoldStructures {
    // Gather all per-chunk PDB files into one directory.
    container params.eval_container
    label 'light_cpu'

    input:
    path pdbs   // list of all .pdb files from all chunks

    output:
    path "esmfold_structures"

    script:
    """
    mkdir -p esmfold_structures
    for f in *.pdb; do
        [ -f "\$f" ] && cp "\$f" esmfold_structures/
    done
    n=\$(ls esmfold_structures/*.pdb | wc -l | tr -d ' ')
    echo "Merged \${n} ESMFold structures"
    [ "\$n" -gt 0 ] || { echo "ERROR: no ESMFold .pdb files found" >&2; exit 1; }
    """
}

// ---------------------------------------------------------------------------
// FOLDSEEK on ESMFold structures — same algo as foldseekSearch, different input
// ---------------------------------------------------------------------------

process foldseekSearchEF {
    container { params.foldseek_container }
    label 'high_cpu'
    publishDir "${params.outdir}/foldseek_esmfold", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path structures  // esmfold_structures/

    output:
    tuple val('foldseek_esmfold'), path("foldseek_esmfold_scope40.tsv.gz")

    script:
    """
    mkdir -p foldseek_ef_tmp
    foldseek easy-search \\
        ${structures} \\
        ${structures} \\
        foldseek_esmfold_scope40.tsv \\
        foldseek_ef_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads       ${task.cpus} \\
        -e              ${params.evalue_report} \\
        --exhaustive-search 0
    gzip -c foldseek_esmfold_scope40.tsv > foldseek_esmfold_scope40.tsv.gz
    rm -f foldseek_esmfold_scope40.tsv
    rm -rf foldseek_ef_tmp
    """
}

// ---------------------------------------------------------------------------
// TEA — Template-free Evolutionary Alignment on ESMFold structures (GPU)
// Paper: Guo et al. 2023, https://doi.org/10.1038/s41592-023-02063-4
// GitHub: https://github.com/GaoTongLab/TEA
// TODO: verify public Docker image once confirmed; build from repo if needed
// ---------------------------------------------------------------------------

process teaSearch {
    container { params.tea_container }
    label 'gpu'
    publishDir "${params.outdir}/tea", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path structures   // esmfold_structures/

    output:
    tuple val('tea'), path("tea_scope40.tsv.gz")

    script:
    // TODO: verify TEA CLI once public Docker image is available.
    // The expected output format is (query, target, score) which maps to
    // (query, target, bits, evalue) for the evaluator — score = bits, no evalue.
    """
    python tea_search.py \\
        --query   ${structures} \\
        --target  ${structures} \\
        --output  tea_scope40.tsv \\
        --threads ${task.cpus} \\
        --device  cuda
    gzip -c tea_scope40.tsv > tea_scope40.tsv.gz
    rm -f tea_scope40.tsv
    """
}

// ---------------------------------------------------------------------------
// EVALUATION — shared across all tools
// ---------------------------------------------------------------------------

process evaluateScope40 {
    // Sensitivity-to-first-FP at family / superfamily / fold levels.
    // Writes {label}.rocx and {label}_auc.txt.
    tag "${label}"
    container params.eval_container
    label 'light_cpu'
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(label), path(hits_gz)
    path  scope_domains

    output:
    path "${label}.rocx",    emit: rocx
    path "${label}_auc.txt", emit: auc

    script:
    """
    evaluate_scope40.py \\
        --hits    ${hits_gz} \\
        --domains ${scope_domains} \\
        --label   ${label} \\
        --outdir  .
    """
}

// ---------------------------------------------------------------------------
// AGGREGATION
// ---------------------------------------------------------------------------

process aggregateROCX {
    container params.eval_container
    label 'light_cpu'
    publishDir params.outdir, mode: 'copy'

    input:
    path 'rocx/*'   // all .rocx files staged into rocx/ subdirectory

    output:
    path "scope40_benchmark_summary.tsv"

    script:
    """
    aggregate_rocx.py rocx scope40_benchmark_summary.tsv
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    // ─── Data staging ────────────────────────────────────────────────────────

    fasta_ch = params.scope_fasta
        ? Channel.fromPath(params.scope_fasta, checkIfExists: true)
        : downloadScopeFasta()

    // Test mode: take only the first N sequences
    if (params.n_test > 0) {
        fasta_ch = subsetFasta(fasta_ch, params.n_test)
        log.info "TEST MODE: using first ${params.n_test} sequences"
    }

    // Shared: parse SCOP hierarchy
    domains_ch = parseScopeHeaders(fasta_ch).domains

    // ─── Tool branches ───────────────────────────────────────────────────────

    results_ch = Channel.empty()

    // pdbstyle (real PDB) structures are only used by plain FoldSeek — FoldSeek-EF
    // uses ESMFold-predicted structures instead. Skip staging entirely (no download,
    // no container needed) when run_foldseek is off.
    if (params.run_foldseek) {
        pdbstyle_ch = params.pdbstyle_dir
            ? Channel.fromPath(params.pdbstyle_dir, checkIfExists: true)
            : downloadPdbstyle()
        structures_ch = collectPdbstyleStructures(pdbstyle_ch, fasta_ch)
    }

    if (params.run_foldseek) {
        results_ch = results_ch.mix( foldseekSearch(structures_ch) )
    }

    if (params.run_mmseqs2) {
        results_ch = results_ch.mix( mmseqs2Search(fasta_ch) )
    }

    if (params.run_blast) {
        blast_db   = blastMakeDB(fasta_ch)
        results_ch = results_ch.mix( blastSearch(fasta_ch, blast_db) )
    }

    if (params.run_diamond) {
        dmnd_db    = diamondMakeDB(fasta_ch)
        results_ch = results_ch.mix( diamondSearch(fasta_ch, dmnd_db) )
    }

    if (params.run_kmerseek) {
        if (params.kmerseek_sweep) {
            // hp variants x k=15-19 (matches ../human-mouse-gencode-orthologs-hp-v040/)
            hp_ksizes = Channel.of(15, 16, 17, 18, 19)
            hp_enc_ksize = Channel.of(
                'hp-lehninger',
                'hp-thomas-dill',
                'hp-kyte-doolittle',
                'hp-thomas-dill-no-c',
                'hp-lehninger-plus-c',
                'hp-pbotc-1st-ed'
            ).combine(hp_ksizes)

            // dayhoff x k=10-20, protein x k=5-15 (matches ../human-mouse-gencode-orthologs/)
            dayhoff_enc_ksize = Channel.of('dayhoff').combine(Channel.of(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
            protein_enc_ksize = Channel.of('protein').combine(Channel.of(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

            kmerseek_enc_ksize = hp_enc_ksize.mix(dayhoff_enc_ksize).mix(protein_enc_ksize)
        } else {
            // Single config run — e.g. the AWS mini-test (hp-thomas-dill k=26).
            kmerseek_enc_ksize = Channel.of(params.kmerseek_encoding).combine(Channel.of(params.kmerseek_k))
            log.info "KMERSEEK SINGLE RUN: ${params.kmerseek_encoding} k=${params.kmerseek_k}"
        }

        km_index_inputs = fasta_ch
            .combine(kmerseek_enc_ksize)
            .map { fasta, encoding, ksize -> tuple(fasta, encoding, ksize) }

        km_indexed = kmerseekIndex(km_index_inputs)
        // (encoding, ksize, index_dir, index_log)

        km_search_inputs = km_indexed
            .combine(fasta_ch)
            .map { encoding, ksize, index_dir, index_log, fasta -> tuple(encoding, ksize, fasta, index_dir, index_log) }

        km_raw     = kmerseekSearch(km_search_inputs)
        results_ch = results_ch.mix( formatKmerseekResults(km_raw) )
    }

    if (params.run_plmsearch) {
        embeddings = plmsearchEmbed(fasta_ch)
        results_ch = results_ch.mix( plmsearchSearch(embeddings, fasta_ch) )
    }

    // ESMFold prediction (shared between FoldSeek-EF and TEA)
    if (params.run_foldseek_ef || params.run_tea) {
        chunks_ch  = chunkFasta(fasta_ch)
            .flatten()
            .map { f -> tuple(f.baseName, f) }
        pdb_files  = esmfoldPredict(chunks_ch).flatten().collect()
        ef_structs = mergeEsmfoldStructures(pdb_files)

        if (params.run_foldseek_ef) {
            results_ch = results_ch.mix( foldseekSearchEF(ef_structs) )
        }

        if (params.run_tea) {
            results_ch = results_ch.mix( teaSearch(ef_structs) )
        }
    }

    // ─── Evaluate all tools ──────────────────────────────────────────────────

    // .first() turns the single-emission domains_ch into a value channel so it's
    // reused for every tool in results_ch, instead of being consumed after one pairing.
    eval_out = evaluateScope40(results_ch, domains_ch.first())

    // ─── Aggregate ───────────────────────────────────────────────────────────

    aggregateROCX( eval_out.rocx.collect() )

    // Summary log
    results_ch.subscribe { label, _hits ->
        log.info "Search complete: ${label}"
    }
}
