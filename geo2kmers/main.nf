#!/usr/bin/env nextflow
/*
 * geo2kmers: fetch -> sequences -> sketch for every sample of one GEO series.
 *
 *   nextflow run main.nf -profile small --gse GSE2183 --outdir results/GSE2183
 *
 * Each process runs an inline Python script with an absolute interpreter path in its shebang
 * line, as the nextflow-conda skill says. The `conda` directive is not used: it does not
 * reliably reach an inline script. Set NCBI_EMAIL in the environment before launching.
 */
nextflow.enable.dsl = 2

params.gse      = null
params.outdir   = null
params.gpl      = ''      // needed when a series uses several platforms
params.sample_tables = false // also download every GSM table, which hold detection calls
// alphabet:k pairs, and the modes to run for each
params.sketches = 'dna:31,protein:10,dayhoff:19,hp:30'
params.modes    = 'weighted,present'
params.proteome = ''      // comma-separated absolute paths: pick protein and hp frames with these proteins
// absolute paths to the python of each process's conda environment
params.python_fetch     = "/scratch/users/olgabot/nb361-pec/env/bin/python"
params.python_sequences = params.python_fetch
params.python_sketch    = params.python_fetch

if (!params.gse || !params.outdir) error "set --gse and --outdir"
params.proteome.tokenize(',').each { if (!file(it).exists()) error "no proteome file ${it}" }

// A task killed from outside: no status, or Integer.MAX_VALUE when the scheduler killed it, or
// 128 + signal. These are worth another try. Anything else is a real error.
def killedFromOutside(status) {
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

process FETCH {
    tag "${gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    storeDir "${params.outdir}/fetch"

    input:
    val gse

    output:
    path "${gse}"

    script:
    """
    #!${params.python_fetch}
    import sys
    sys.path.insert(0, "${projectDir}")
    from geo2kmers.cli import main
    raise SystemExit(main(["fetch", "${gse}", "--outdir", "${gse}"${params.gpl ? ', "--gpl", "' + params.gpl + '"' : ''}${params.sample_tables ? ', "--sample-tables"' : ''}]))
    """.stripIndent()
}

process SEQUENCES {
    tag "${params.gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    storeDir "${params.outdir}/sequences"

    input:
    path series

    output:
    path "probes.fasta", emit: fasta
    path "probe_sequence_source.tsv"
    path "genbank_cache.fasta", optional: true

    script:
    """
    #!${params.python_sequences}
    import glob
    import sys
    sys.path.insert(0, "${projectDir}")
    from geo2kmers.cli import main
    tables = glob.glob("${series}/GPL*.txt")
    assert len(tables) == 1, tables
    raise SystemExit(main(["sequences", tables[0], "--out", "probes.fasta"]))
    """.stripIndent()
}

process SKETCH {
    tag "${gsm} ${dye} ${alphabet} k${k} ${mode}"
    // 'finish': a failed sketch stops new work and the run exits with an error, so it cannot pass unseen
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'finish' }
    maxRetries 3
    // holding a proteome's k-mers peaked at 3.3 GB on GENCODE human, 2.2 GB on the mussel proteome
    memory { (alphabet != 'dna' && params.proteome ? 4.GB : 2.GB) * task.attempt }
    publishDir "${params.outdir}/sketches", mode: 'copy'

    input:
    tuple val(gsm), val(dye), val(alphabet), val(k), val(mode)
    path series
    path fasta

    output:
    path "${gsm}*.tsv*"

    script:
    def stem = "${gsm}${dye ? '.' + dye : ''}.${alphabet}.k${k}.${mode}"
    def extra = dye ? "'--channel', '${dye}'," : ""
    if (alphabet != 'dna' && params.proteome) {
        extra += " '--proteome', " + params.proteome.tokenize(',').collect { "'${it}'" }.join(', ') + ","
    }
    """
    #!${params.python_sketch}
    import glob
    import sys
    sys.path.insert(0, "${projectDir}")
    from geo2kmers.cli import main
    matrix = glob.glob("${series}/*_series_matrix.txt.gz")
    source = matrix[0] if matrix else "${series}/gsm"
    raise SystemExit(main(["sketch", "--fasta", "${fasta}", "--intensities", source, "--sample", "${gsm}",
                           ${extra} "--k", "${k}", "--alphabet", "${alphabet}", "--mode", "${mode}",
                           "--out", "${stem}.tsv"]))
    """.stripIndent()
}

workflow {
    series = FETCH(Channel.of(params.gse))
    probes = SEQUENCES(series)

    def sketches = params.sketches.tokenize(',').collect { it.tokenize(':') }
    def modes = params.modes.tokenize(',')
    // a two-colour sample gives one sketch per channel
    jobs = series.flatMap { dir ->
        def out = []
        dir.resolve('samples.tsv').splitCsv(header: true, sep: '\t').each { row ->
            def dyes = row.channel_count == '1' ? [''] : ['ch1', 'ch2']
            dyes.each { ch ->
                sketches.each { a, k ->
                    modes.each { m -> out << tuple(row.gsm, ch, a, k, m) }
                }
            }
        }
        out
    }
    SKETCH(jobs, series.first(), probes.fasta.first())
}
