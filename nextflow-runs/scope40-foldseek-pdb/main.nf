#!/usr/bin/env nextflow

/*
 * scope40-foldseek-pdb/main.nf
 *
 * Run FoldSeek on all SCOPe40 2.08 domain structures and evaluate
 * against the SCOPe hierarchy.
 *
 * Requires the SCOPe pdbstyle tarball to be downloaded and extracted first:
 *   curl -O http://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz
 *   tar -xzf pdbstyle-sel-gs-bib-40-2.08.tgz -C ~/data/scope/pdbstyle-2.08 --strip-components=1
 * This gives pre-cut domain .ent files for all 15,177 SCOPe40 domains.
 *
 * Pipeline:
 *   1. parseScopeHeaders        — parse FASTA headers → scope_domains.tsv (all 15,177)
 *   2. collectPdbstyleStructures — copy all .ent files from pdbstyle dir → structures/
 *   3. foldseekSearch           — foldseek easy-search all-vs-all
 *   4. evaluateFoldseek         — sensitivity-to-first-FP → foldseek_pdb.rocx
 *
 * Usage:
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf -resume
 */

nextflow.enable.dsl=2

def home = System.getProperty('user.home')

params.scope_fasta    = "${home}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir         = "${home}/data/scope/results-foldseek-pdb"
params.pdbstyle_dir   = "${home}/data/scope/pdbstyle-2.08"
params.foldseek       = "${home}/anaconda3/envs/foldseek-10.941cd33/bin/foldseek"
params.evalue_report  = 10.0


// ---------------------------------------------------------------------------
// PROCESS 1 — Parse SCOPe FASTA headers into a domain metadata table (all domains)
// ---------------------------------------------------------------------------

process parseScopeHeaders {
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path "scope_domains.tsv", emit: domains

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import re

    # Header: >d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Ciliate ...}
    pat = re.compile(r'^>(\\S+)\\s+(\\S+)')

    with open('${fasta}') as fi, open('scope_domains.tsv', 'w') as fo:
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
    """
}


// ---------------------------------------------------------------------------
// PROCESS 2 — Collect pre-cut domain structures from SCOPe pdbstyle directory
// ---------------------------------------------------------------------------

process collectPdbstyleStructures {
    // Reads .ent files from params.pdbstyle_dir (extracted pdbstyle tarball).
    // SCOPe pdbstyle layout: {pdbstyle_dir}/{2-char-subdir}/d{domain_id}.ent
    // Output: structures/ with all files renamed to .pdb for FoldSeek.

    output:
    path "structures"

    script:
    """
    mkdir -p structures
    find ${params.pdbstyle_dir} -name "*.ent" | while read f; do
        base=\$(basename "\$f" .ent)
        ln -s "\$f" "structures/\${base}.pdb"
    done
    n=\$(ls structures | wc -l | tr -d ' ')
    echo "Collected \${n} domain structures from ${params.pdbstyle_dir}"
    if [ "\$n" -eq 0 ]; then
        echo "ERROR: no .ent files found in ${params.pdbstyle_dir}" >&2
        exit 1
    fi
    """
}


// ---------------------------------------------------------------------------
// PROCESS 3 — FoldSeek all-vs-all on all SCOPe40 domain structures
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


// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {
    fasta_ch = Channel.fromPath(params.scope_fasta, checkIfExists: true)

    parsed = parseScopeHeaders(fasta_ch)

    // FoldSeek branch (structure-based, all pdbstyle domains)
    structs_ch = collectPdbstyleStructures()
    fs_hits_ch = foldseekSearch(structs_ch)
    evaluateFoldseek(fs_hits_ch, parsed.domains)

}
