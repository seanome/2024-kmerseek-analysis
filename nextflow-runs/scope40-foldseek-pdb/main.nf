#!/usr/bin/env nextflow

/*
 * scope40-foldseek-pdb/main.nf
 *
 * Run FoldSeek on real PDB crystal structures for SCOPe40 single-domain
 * proteins and evaluate against the SCOPe hierarchy.
 *
 * "Single-domain" = SCOPe domain IDs ending in '_' (the SCOP domain covers
 * the entire PDB chain), excluding multi-domain chains (digit suffix).
 * This gives ~6,500 domains from ~6,266 unique PDB entries.
 *
 * Pipeline:
 *   1. parseScopeHeaders  — parse FASTA headers → scope_domains.tsv + id list
 *   2. downloadScopeStructures — download PDB chains from RCSB
 *   3. foldseekSearch     — foldseek easy-search all-vs-all (same dir query + ref)
 *   4. evaluateFoldseek   — sensitivity-to-first-FP → foldseek_pdb.rocx
 *
 * Usage:
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf --resume
 */

nextflow.enable.dsl=2

def home = System.getProperty('user.home')

params.scope_fasta     = "${home}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir          = "${home}/data/scope/results-foldseek-pdb"
params.structures_dir  = "${home}/data/scope/pdbstyle-2.08"
params.foldseek        = "${home}/anaconda3/envs/foldseek-10.941cd33/bin/foldseek"
params.mmseqs          = "${home}/anaconda3/envs/orthofinder/bin/mmseqs"
params.evalue_report   = 10.0
params.max_workers     = 8


// ---------------------------------------------------------------------------
// PROCESS 1 — Parse SCOPe FASTA headers into a domain metadata table
// ---------------------------------------------------------------------------

process parseScopeHeaders {
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path "scope_domains.tsv",    emit: domains
    path "single_domain_ids.txt", emit: single_ids

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import re

    # Header: >d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Ciliate ...}
    pat = re.compile(r'^>(\\S+)\\s+(\\S+)')

    with open('${fasta}') as fi, \\
         open('scope_domains.tsv', 'w') as fo, \\
         open('single_domain_ids.txt', 'w') as fs:

        fo.write('domain_id\\tscop_id\\tscop_class\\tscop_fold\\tscop_superfamily\\tscop_family\\n')
        for line in fi:
            if not line.startswith('>'):
                continue
            m = pat.match(line.rstrip())
            if not m:
                continue
            domain_id, scop_id = m.group(1), m.group(2)
            parts = scop_id.split('.')
            scop_class = parts[0] if len(parts) >= 1 else ''
            scop_fold  = '.'.join(parts[:2]) if len(parts) >= 2 else scop_class
            scop_sfam  = '.'.join(parts[:3]) if len(parts) >= 3 else scop_fold
            scop_fam   = scop_id
            fo.write(f'{domain_id}\\t{scop_id}\\t{scop_class}\\t{scop_fold}\\t{scop_sfam}\\t{scop_fam}\\n')
            if domain_id.endswith('_'):
                fs.write(domain_id + '\\n')
    """
}


// ---------------------------------------------------------------------------
// PROCESS 2 — Download PDB chain structures from RCSB
// ---------------------------------------------------------------------------

process downloadScopeStructures {
    // No publishDir: the Python script caches raw PDB files in params.structures_dir/raw/
    // and Nextflow -resume handles re-use of the extracted structures directory.

    input:
    path domain_ids

    output:
    path "structures"

    script:
    """
    ${projectDir}/bin/download_scope_structures.py \\
        ${domain_ids} \\
        --outdir       structures \\
        --cache        ${params.structures_dir} \\
        --max-workers  ${params.max_workers}
    """
}


// ---------------------------------------------------------------------------
// PROCESS 3 — FoldSeek all-vs-all on SCOPe40 single-domain PDB structures
// ---------------------------------------------------------------------------

process foldseekSearch {
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path structures

    output:
    path "foldseek_scope40_pdb.tsv.gz"

    script:
    """
    mkdir -p foldseek_tmp

    ${params.foldseek} easy-search \\
        ${structures} \\
        ${structures} \\
        foldseek_scope40_pdb.tsv \\
        foldseek_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report} \\
        --exhaustive-search 0

    gzip -c foldseek_scope40_pdb.tsv > foldseek_scope40_pdb.tsv.gz
    rm -f foldseek_scope40_pdb.tsv
    rm -rf foldseek_tmp
    """
}


// ---------------------------------------------------------------------------
// PROCESS 4 — Compute sensitivity-to-first-FP → .rocx evaluation file
// ---------------------------------------------------------------------------

process evaluateFoldseek {
    publishDir params.outdir, mode: 'copy'

    input:
    path foldseek_hits
    path scope_domains

    output:
    path "foldseek_pdb.rocx"
    path "foldseek_pdb_auc.txt"

    script:
    """
    ${projectDir}/bin/evaluate_foldseek_scope.py \\
        --hits    ${foldseek_hits} \\
        --domains ${scope_domains} \\
        --label   foldseek_pdb \\
        --outdir  .
    """
}


// ---------------------------------------------------------------------------
// PROCESS 5 — Extract single-domain sequences from SCOPe40 FASTA for MMseqs2
// ---------------------------------------------------------------------------

process extractSingleDomainFasta {
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path "scope40_single_domain.fa"

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    writing = False
    with open('${fasta}') as fi, open('scope40_single_domain.fa', 'w') as fo:
        for line in fi:
            if line.startswith('>'):
                domain_id = line.split()[0][1:]
                writing = domain_id.endswith('_')
            if writing:
                fo.write(line)
    """
}


// ---------------------------------------------------------------------------
// PROCESS 6 — MMseqs2 easy-search all-vs-all on SCOPe40 single-domain seqs
// ---------------------------------------------------------------------------

process mmseqs2Search {
    publishDir "${params.outdir}/mmseqs2", mode: 'copy', pattern: '*.tsv.gz'

    input:
    path fasta

    output:
    path "mmseqs2_scope40.tsv.gz"

    script:
    """
    mkdir -p mmseqs2_tmp

    ${params.mmseqs} easy-search \\
        ${fasta} \\
        ${fasta} \\
        mmseqs2_scope40.tsv \\
        mmseqs2_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report} \\
        -s 7.5

    gzip -c mmseqs2_scope40.tsv > mmseqs2_scope40.tsv.gz
    rm -f mmseqs2_scope40.tsv
    rm -rf mmseqs2_tmp
    """
}


// ---------------------------------------------------------------------------
// PROCESS 7 — Evaluate MMseqs2 hits → .rocx file
// ---------------------------------------------------------------------------

process evaluateMMseqs2 {
    publishDir params.outdir, mode: 'copy'

    input:
    path mmseqs2_hits
    path scope_domains

    output:
    path "mmseqs2_scope40.rocx"
    path "mmseqs2_scope40_auc.txt"

    script:
    """
    ${projectDir}/bin/evaluate_foldseek_scope.py \\
        --hits    ${mmseqs2_hits} \\
        --domains ${scope_domains} \\
        --label   mmseqs2_scope40 \\
        --outdir  .
    """
}


// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {
    fasta_ch = Channel.fromPath(params.scope_fasta, checkIfExists: true)

    parsed = parseScopeHeaders(fasta_ch)

    // FoldSeek branch (structure-based, real PDB chains)
    structs_ch = downloadScopeStructures(parsed.single_ids)
    fs_hits_ch = foldseekSearch(structs_ch)
    evaluateFoldseek(fs_hits_ch, parsed.domains)

    // MMseqs2 branch (sequence-based, single-domain FASTA subset)
    single_fa_ch = extractSingleDomainFasta(fasta_ch)
    mm_hits_ch   = mmseqs2Search(single_fa_ch)
    evaluateMMseqs2(mm_hits_ch, parsed.domains)
}
