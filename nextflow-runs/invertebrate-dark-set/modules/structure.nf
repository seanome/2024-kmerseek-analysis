/*
 * Foldseek and ProstT5 on the dark set: the structure tools, beside phmmer, jackhmmer and
 * mmseqs2, against the same clade-excluded Swiss-Prot reference. `-entry darkStructure`
 * in main.nf wires these; nothing else uses them.
 *
 *   foldseekTargetDb   Foldseek's AlphaFold DB Swiss-Prot database cut down to the
 *                      accessions of one clade-excluded reference, so Foldseek searches
 *                      the sequence tools' targets and no others. Also writes which
 *                      reference proteins have a model.
 *   afdbQueryModels    the AlphaFold DB models of one query proteome, from the proteome
 *                      archive and per-accession downloads, and which proteins have one.
 *   foldseekQueryDb    Foldseek database of those models, plus their amino-acid and 3Di
 *                      strings for the decoys.
 *   foldseekDecoyFasta (amino acid, 3Di) pairs dipeptide-shuffled, one decoy per model.
 *   foldseekDecoyDb    Foldseek database of the decoys.
 *   prostt5Fasta       the proteome cut at --prostt5_max_len, and its dipeptide-shuffled copy.
 *   prostt5Db          ProstT5 3Di predicted from sequence, on one GPU; real and decoy.
 *   foldseekSearch     one query chunk, real or decoy models, against the cut database.
 *   prostt5Search      one query chunk, real or decoy ProstT5 3Di, against the same database.
 *
 * Every search output has the QfO region benchmark's eight columns, no header:
 * query, target, qstart, qend, tstart, tend, bits, evalue, with AlphaFold DB names reduced
 * to the bare accession and fragment hits moved to protein coordinates. A chunk whose
 * proteins have no model writes an empty file; coverage.tsv says those proteins were not
 * run, which is different from run with no hit.
 */

// Resources for every process here are set in nextflow.config (withName), because a
// config selector beats a process-body directive; the body values are the standard
// profile's defaults.

process foldseekTargetDb {
    tag "minus_${clade}"
    container "${params.foldseek_image}"
    cpus 2
    memory '8 GB'
    storeDir "${params.structure_cache ?: params.outdir + '/structure_db'}/target"

    input:
    tuple val(clade), path(ref_dir), path(afdb, stageAs: 'afdb')

    output:
    tuple val(clade), path("minus_${clade}_afdb")

    script:
    """
    set -euo pipefail
    out=minus_${clade}_afdb
    mkdir -p \$out
    # Database keys of every model whose accession is in the reference, and for each
    # reference protein whether it has a model. Names look like AF-P12345-F1-model_v6.
    awk -F'\\t' -v out=\$out '
        NR == FNR { ref[\$1] = 1; n_ref++; next }
        {
            n_db++
            if (!match(\$2, /^AF-[A-Za-z0-9]+-F[0-9]+-model_v[0-9]+/)) { bad++; next }
            split(substr(\$2, 1, RLENGTH), p, "-")
            if (!(p[2] in ref)) next
            print \$1 > (out "/ids.txt")
            if (!(p[2] in has)) { has[p[2]] = substr(p[4], 8); n_has++ }
            ver[substr(p[4], 8)]++
        }
        END {
            print "accession\\thas_model\\tmodel_version" > (out "/coverage.tsv")
            for (a in ref) print a "\\t" ((a in has) ? 1 : 0) "\\t" ((a in has) ? "v" has[a] : "") > (out "/coverage.tsv")
            vs = ""
            for (v in ver) vs = vs (vs == "" ? "" : ", ") "\\"v" v "\\": " ver[v]
            printf "{\\"excluded_clade\\": \\"${clade}\\", \\"n_reference\\": %d, \\"n_with_model\\": %d, " \\
                   "\\"share_with_model\\": %.6f, \\"n_afdb_entries\\": %d, \\"n_afdb_names_unparsed\\": %d, " \\
                   "\\"model_versions_of_kept_entries\\": {%s}}\\n", \\
                   n_ref, n_has, (n_ref ? n_has / n_ref : 0), n_db, bad, vs > (out "/summary.json")
        }' ${ref_dir}/reference_accessions.txt afdb/db.lookup
    test -s \$out/ids.txt || { echo "no AlphaFold DB model matches minus_${clade}" >&2; exit 1; }
    # createsubdb in Foldseek 10 cuts the amino-acid, 3Di, C-alpha and header databases
    # together. The keys stay the database's own, so the lookup is cut to the same keys.
    foldseek createsubdb \$out/ids.txt afdb/db \$out/db --subdb-mode 0
    awk -F'\\t' 'NR == FNR { k[\$1]; next } \$1 in k' \$out/ids.txt afdb/db.lookup > \$out/db.lookup
    cp afdb/db.version \$out/afdb_db.version 2>/dev/null || true
    for s in '' _ss _ca _h; do test -e \$out/db\${s}.dbtype || { echo "createsubdb wrote no db\${s}" >&2; exit 1; }; done
    cat \$out/summary.json >&2
    """

    stub:
    """
    mkdir -p minus_${clade}_afdb
    printf '0\\tAF-Q6GZX4-F1-model_v6\\t0\\n' > minus_${clade}_afdb/db.lookup
    printf 'accession\\thas_model\\tmodel_version\\nQ6GZX4\\t1\\tv6\\nP12345\\t0\\t\\n' > minus_${clade}_afdb/coverage.tsv
    echo '{"n_reference": 2, "n_with_model": 1}' > minus_${clade}_afdb/summary.json
    """
}

process afdbQueryModels {
    tag "${species}"
    label 'python_scoring'
    cpus 1
    memory '4 GB'
    time '4h'

    input:
    tuple val(species), path(proteome), path(archives, stageAs: 'archives/*'), path(extra, stageAs: 'extra')

    output:
    tuple val(species), path("${species}_afdb_models")

    script:
    def arch = archives instanceof List ? archives : [archives]
    def archArgs = arch.findAll { it.name != 'NO_ARCHIVE' }.collect { "--archive ${it}" }.join(' ')
    // A species with no per-accession downloads gets the NO_EXTRA placeholder file, which
    // the script skips because it is not a directory.
    def extraArg = "--extra-dir extra"
    """
    set -euo pipefail
    mkdir -p ${species}_afdb_models
    afdb_query_models.py --species ${species} --proteome ${proteome} \\
        ${archArgs} ${extraArg} \\
        --out-dir ${species}_afdb_models/models \\
        --coverage ${species}_afdb_models/coverage.tsv \\
        --summary ${species}_afdb_models/summary.json
    """

    stub:
    """
    mkdir -p ${species}_afdb_models/models
    printf 'accession\\tlength\\thas_model\\tn_fragments\\tmodel_version\\tsource\\n${species}_A\\t16\\t1\\t1\\tv6\\tstub\\n${species}_B\\t17\\t0\\t0\\t\\tnone\\n' > ${species}_afdb_models/coverage.tsv
    echo '{"species": "${species}", "n_queries": 2, "n_with_model": ${species == 'botryllus' ? 0 : 1}}' > ${species}_afdb_models/summary.json
    """
}

process foldseekQueryDb {
    tag "${species}"
    container "${params.foldseek_image}"
    cpus 8
    memory '16 GB'
    time '2h'

    input:
    tuple val(species), path(models)

    output:
    tuple val(species), path("${species}_foldseek_query")

    script:
    """
    set -euo pipefail
    out=${species}_foldseek_query
    mkdir -p \$out
    foldseek createdb ${models}/models/ \$out/db --threads ${task.cpus}
    # The amino-acid and 3Di strings, for the decoys (Foldseek wiki: convert2fasta on db_ss
    # with its own header link).
    foldseek lndb \$out/db_h \$out/db_ss_h
    foldseek convert2fasta \$out/db \$out/aa.fasta
    foldseek convert2fasta \$out/db_ss \$out/ss.fasta
    cp ${models}/coverage.tsv ${models}/summary.json \$out/
    n_db=\$(wc -l < \$out/db.lookup)
    n_files=\$(ls ${models}/models | wc -l)
    echo "${species}: \${n_db} database entries from \${n_files} model files" >&2
    test "\$n_db" -eq "\$n_files" || { echo "createdb read \${n_db} of \${n_files} model files" >&2; exit 1; }
    """

    stub:
    """
    mkdir -p ${species}_foldseek_query
    printf '0\\tAF-${species}_A-F1-model_v6\\t0\\n' > ${species}_foldseek_query/db.lookup
    printf '>AF-${species}_A-F1-model_v6\\nMKLLVLLAAGGLLSAQ\\n' > ${species}_foldseek_query/aa.fasta
    printf '>AF-${species}_A-F1-model_v6\\nDDPVVVVVVCCVVVPD\\n' > ${species}_foldseek_query/ss.fasta
    cp ${models}/coverage.tsv ${models}/summary.json ${species}_foldseek_query/
    """
}

process foldseekDecoyFasta {
    tag "${species}"
    label 'python_scoring'
    cpus 1
    memory '8 GB'
    time '1h'

    input:
    tuple val(species), path(qdb), path(decoys_module, stageAs: 'decoys.py')

    output:
    tuple val(species), path("${species}_decoy_aa.fasta"), path("${species}_decoy_ss.fasta")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    structure_decoys.py columns --aa ${qdb}/aa.fasta --ss ${qdb}/ss.fasta \\
        --out-aa ${species}_decoy_aa.fasta --out-ss ${species}_decoy_ss.fasta \\
        --seed ${params.decoy_seed}
    """

    stub:
    """
    printf '>AF-${species}_A-F1-model_v6\\nMLKLVLAAGLLGSLAQ\\n' > ${species}_decoy_aa.fasta
    printf '>AF-${species}_A-F1-model_v6\\nDPVVVDVVCVVVCVPD\\n' > ${species}_decoy_ss.fasta
    """
}

process foldseekDecoyDb {
    tag "${species}"
    container "${params.foldseek_image}"
    cpus 2
    memory '8 GB'
    time '1h'

    input:
    tuple val(species), path(aa), path(ss)

    output:
    tuple val(species), path("${species}_foldseek_decoy")

    script:
    // Foldseek wiki, "Create a Foldseek database from 3Di and amino acid sequences".
    """
    set -euo pipefail
    out=${species}_foldseek_decoy
    mkdir -p \$out
    foldseek base:createdb ${aa} \$out/db --shuffle 0
    foldseek base:createdb ${ss} \$out/db_ss --shuffle 0
    test \$(wc -l < \$out/db.lookup) -eq \$(grep -c '^>' ${aa})
    """

    stub:
    """
    mkdir -p ${species}_foldseek_decoy
    printf '0\\tAF-${species}_A-F1-model_v6\\t0\\n' > ${species}_foldseek_decoy/db.lookup
    """
}

process prostt5Fasta {
    tag "${species}"
    label 'python_scoring'
    cpus 1
    memory '8 GB'
    time '1h'

    input:
    tuple val(species), path(proteome), path(decoys_module, stageAs: 'decoys.py')

    output:
    tuple val(species), path("${species}.prostt5_real.fasta"), path("${species}.prostt5_decoy.fasta"),
          path("${species}.prostt5_skipped.tsv")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    structure_decoys.py sequence --in ${proteome} --max-len ${params.prostt5_max_len} \\
        --out-real ${species}.prostt5_real.fasta --out-decoy ${species}.prostt5_decoy.fasta \\
        --skipped ${species}.prostt5_skipped.tsv --seed ${params.decoy_seed}
    """

    stub:
    """
    cp ${proteome} ${species}.prostt5_real.fasta
    cp ${proteome} ${species}.prostt5_decoy.fasta
    printf 'accession\\tlength\\n' > ${species}.prostt5_skipped.tsv
    """
}

process prostt5Db {
    tag "${species}.${kind}"
    container "${params.foldseek_image}"
    cpus 8
    memory '16 GB'
    time '2h'
    publishDir "${params.outdir}/${species}/structure", mode: 'copy', pattern: '*.prostt5_*.log'

    input:
    tuple val(species), val(kind), path(fasta), path(weights, stageAs: 'prostt5_weights')

    output:
    tuple val(species), val(kind), path("${species}_prostt5_${kind}"), emit: db
    path "${species}.prostt5_${kind}.log", emit: log

    script:
    // createdb is where ProstT5 runs. A memory kill inside it can exit with a code the
    // global retry already handles; nothing here needs its own retry rule.
    """
    set -euo pipefail
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > ${species}.prostt5_${kind}.log 2>&1 || echo "nvidia-smi not available" > ${species}.prostt5_${kind}.log
    mkdir -p ${species}_prostt5_${kind}
    foldseek createdb ${fasta} ${species}_prostt5_${kind}/db \\
        --prostt5-model prostt5_weights \\
        ${params.prostt5_gpu ? '--gpu 1' : ''} \\
        --threads ${task.cpus} 2>&1 | tail -n 20 >> ${species}.prostt5_${kind}.log
    test \$(wc -l < ${species}_prostt5_${kind}/db.lookup) -eq \$(grep -c '^>' ${fasta})
    """

    stub:
    """
    mkdir -p ${species}_prostt5_${kind}
    awk '/^>/ {print n++ "\\t" substr(\$1, 2) "\\t0"}' ${fasta} > ${species}_prostt5_${kind}/db.lookup
    echo stub > ${species}.prostt5_${kind}.log
    """
}

// The search body shared by both tools: cut the query database to this chunk's proteins,
// search, write QfO's columns. name_awk turns a lookup name into the protein's accession.
def structureSearchScript(String qdb, String tdb, String chunk, String out_name, int cpus) {
    """
    set -euo pipefail
    awk '/^>/ { print substr(\$1, 2) }' ${chunk} > chunk_ids.txt
    awk -F'\\t' 'NR == FNR { a[\$1]; next }
        { acc = \$2
          if (match(\$2, /^AF-[A-Za-z0-9]+-F[0-9]+/)) { split(substr(\$2, 1, RLENGTH), p, "-"); acc = p[2] }
          if (acc in a) print \$1 }' chunk_ids.txt ${qdb}/db.lookup > keys.txt
    if [ ! -s keys.txt ]; then
        echo "no protein of ${chunk} is in ${qdb}: nothing to search" >&2
        : | gzip -c > ${out_name}
        exit 0
    fi
    mkdir -p sub tmp
    foldseek createsubdb keys.txt ${qdb}/db sub/db --subdb-mode 0
    foldseek search sub/db ${tdb}/db res tmp \\
        --threads ${cpus} -e ${params.evalue_report} --max-seqs ${params.structure_max_seqs}
    foldseek convertalis sub/db ${tdb}/db res out.tsv --threads ${cpus} \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
    # AlphaFold DB names (AF-<acc>-F<n>-model_v<k>) become the bare accession. A model over
    # 2_700 residues is split into fragments on a 200-residue stride, each numbered from 1,
    # so a hit on F<n> moves (n - 1) * 200 residues (QfO region benchmark, foldseekSearch).
    awk -F'\\t' 'BEGIN { OFS = "\\t" } {
        for (i = 1; i <= 2; i++) {
            if (match(\$i, /AF-[A-Za-z0-9]+-F[0-9]+/)) {
                tok = substr(\$i, RSTART, RLENGTH)
                split(tok, p, "-")
                off = (substr(p[3], 2) + 0 - 1) * 200
                if (off) { \$(2 * i + 1) += off; \$(2 * i + 2) += off }
                \$i = p[2]
            }
        }
        print
    }' out.tsv | gzip -c > ${out_name}
    echo "\$(wc -l < keys.txt) query entries, \$(wc -l < out.tsv) hits" >&2
    """
}

process foldseekSearch {
    tag "${species}.${chunk.simpleName}.${kind}"
    container "${params.foldseek_image}"
    cpus 8
    memory '16 GB'
    time '4h'
    publishDir "${params.outdir}/${species}/structure", mode: 'copy'

    input:
    tuple val(species), val(kind), val(out_name), path(chunk), path(qdb), path(tdb)

    output:
    tuple val(species), val(kind), path(out_name)

    script:
    structureSearchScript(qdb.toString(), tdb.toString(), chunk.toString(), out_name, task.cpus as int)

    stub:
    """
    printf '${species}_A\\tQ6GZX4\\t1\\t10\\t1\\t10\\t50\\t1e-9\\n' | gzip -c > ${out_name}
    """
}

process prostt5Search {
    tag "${species}.${chunk.simpleName}.${kind}"
    container "${params.foldseek_image}"
    cpus 8
    memory '16 GB'
    time '4h'
    publishDir "${params.outdir}/${species}/structure", mode: 'copy'

    input:
    tuple val(species), val(kind), val(out_name), path(chunk), path(qdb), path(tdb)

    output:
    tuple val(species), val(kind), path(out_name)

    script:
    structureSearchScript(qdb.toString(), tdb.toString(), chunk.toString(), out_name, task.cpus as int)

    stub:
    """
    printf '${species}_A\\tQ6GZX4\\t1\\t10\\t1\\t10\\t40\\t1e-5\\n' | gzip -c > ${out_name}
    """
}

// Coverage files, one place: what was run and what was not.
process structureCoverage {
    tag "${species}"
    publishDir "${params.outdir}/${species}/structure", mode: 'copy'
    cpus 1
    memory '1 GB'
    executor 'local'

    input:
    tuple val(species), path(query_cov, stageAs: 'query_coverage.tsv'),
          path(query_summary, stageAs: 'query_summary.json'),
          path(target_cov, stageAs: 'target_coverage.tsv'),
          path(target_summary, stageAs: 'target_summary.json'),
          path(prostt5_skipped, stageAs: 'prostt5_skipped.tsv')

    output:
    tuple val(species), path("${species}_afdb_query_coverage.tsv"), path("${species}_afdb_query_summary.json"),
          path("${species}_afdb_target_coverage.tsv"), path("${species}_afdb_target_summary.json"),
          path("${species}_prostt5_skipped.tsv")

    script:
    """
    cp query_coverage.tsv ${species}_afdb_query_coverage.tsv
    cp query_summary.json ${species}_afdb_query_summary.json
    cp target_coverage.tsv ${species}_afdb_target_coverage.tsv
    cp target_summary.json ${species}_afdb_target_summary.json
    cp prostt5_skipped.tsv ${species}_prostt5_skipped.tsv
    """
}
