nextflow.enable.dsl=2
import groovy.json.JsonSlurper

// Script paths are staged inputs so edits invalidate their dependent tasks.
def scripts = file("${projectDir}/../../scripts/botryllus_shazam")

process VALIDATE_PAIR {
    publishDir "${params.outdir}/input-validation", mode: 'copy', pattern: '*.validated.json'
    tag "$run"
    cpus 3
    input:
    tuple val(run), path(reads), val(record)
    path helper
    path validator
    output:
    tuple val(run), path(reads), path("${run}.validated.json")
    script:
    """
    '${params.python}' '${helper}' pair '${run}' '${reads[0]}' '${reads[1]}' '${record.files[0].sha256}' '${record.files[1].sha256}' '${record.pairs}' '${validator}'
    """
}

process BUILD_VALIDATOR {
    input: path source
    output: path 'validate_pairs'
    script:
    """
    module load devel rust/1.90.0
    rustc -O '${source}' -o validate_pairs
    """
}

process COUNT_KMERS {
    tag "$run"
    cpus 8
    memory '32 GB'
    time '12h'
    input: tuple val(run), path(reads), path(validation)
    output: tuple val(run), path("${run}.counts.txt.gz")
    script:
    """
    '${params.dekupl}/bin/jellyfish' count -C -c 63 --out-counter-len 8 -L 2 -m 31 -s 10000 -t ${task.cpus} -F 2 -o counts.jf <(gzip -dc '${reads[0]}') <(gzip -dc '${reads[1]}')
    '${params.dekupl}/bin/jellyfish' dump -c counts.jf | LC_ALL=C sort -k 1 -S 3000M --parallel=${task.cpus} | gzip -c > '${run}.counts.txt.gz'
    """
}

process JOIN_KMERS {
    publishDir "${params.outdir}/counts", mode: 'copy'
    memory '32 GB'
    time '12h'
    input: path counts, arity: '1..*'
    output: path 'raw-counts.tsv.gz'
    script:
    def ordered = counts.sort { it.name }
    def names = ordered.collect { it.name.replace('.counts.txt.gz','') }.join('\\t')
    def args = ordered.collect { "'${it}'" }.join(' ')
    """
    printf 'tag\\t${names}\\n' | gzip -c > raw-counts.tsv.gz
    '${params.dekupl}/share/dekupl/bin/joinCounts' -r 2 -a 5 ${args} | gzip -c >> raw-counts.tsv.gz
    """
}

process KALLISTO_INDEX {
    memory '16 GB'
    input: path reference
    output: path 'reference.idx'
    script:
    """
    '${params.dekupl}/share/dekupl/bin/kallisto' index -i reference.idx '${reference}'
    """
}

process QUANTIFY_GENES {
    tag "$run"
    memory '8 GB'
    time '12h'
    input:
    tuple val(run), path(reads), path(validation)
    path index
    output: tuple val(run), path("${run}.quant")
    script:
    """
    '${params.dekupl}/share/dekupl/bin/kallisto' quant -i '${index}' -o '${run}.quant' '${reads[0]}' '${reads[1]}'
    """
}

process GENE_COUNTS {
    publishDir "${params.outdir}/counts", mode: 'copy'
    memory '16 GB'
    input:
    path quantifications, arity: '1..*'
    path reference
    path gtf
    path helper
    output:
    path 'genes.tsv.gz', emit: counts
    path 'transcript-to-gene.tsv', emit: mapping
    script:
    def args = quantifications.sort { it.name }.collect { "'${it}'" }.join(' ')
    """
    '${params.python}' '${helper}' genes '${reference}' '${gtf}' ${args}
    """
}

process POOL_COUNTS {
    memory '16 GB'
    time '12h'
    input:
    path kmers
    path genes
    path design
    path pool_script
    output: path 'models/*/*', arity: '1..*'
    script:
    """
    '${params.python}' '${pool_script}' '${kmers}' '${genes}' '${design}' models
    """
}

process NORMALIZE {
    tag "$contrast:$assay"
    input: tuple val(contrast), val(assay), path(folder)
    output: tuple val(contrast), val(assay), path('normalized')
    script:
    """
    cp -RL -- '${folder}' normalized
    '${params.dekupl}/share/dekupl/bin/computeNF' normalized/pooled.tsv.gz > normalized/normalization.tsv
    """
}

process BLOCKED_MODEL {
    tag "$contrast:$assay"
    memory '32 GB'
    time '24h'
    publishDir "${params.outdir}/models/${contrast}/${assay}", mode: 'copy'
    input:
    tuple val(contrast), val(assay), path(folder)
    path model_script
    output: tuple val(contrast), val(assay), path('result')
    script:
    """
    cp -RL -- '${folder}' result
    '${params.dekupl}/bin/Rscript' '${model_script}' result '${contrast}' '${assay}'
    """
}

process SELECT_DE_SEEDS {
    publishDir "${params.outdir}/selected-dekupl", mode: 'copy'
    input:
    path models, stageAs: 'models/??/*', arity: '3'
    path helper
    output:
    path 'selected', emit: all
    path 'selected/contrasts/*', arity: '3', emit: contrasts
    script:
    def args = models.collect { "'${it}'" }.join(' ')
    """
    '${params.python}' '${helper}' seeds ${args}
    """
}

process MERGE_TAGS {
    publishDir "${params.outdir}/native-contigs/${contrast}", mode: 'copy'
    tag "$contrast"
    input:
    tuple val(contrast), path(selected)
    path helper
    path merge_script
    output: tuple val(contrast), path('native')
    script:
    """
    '${params.python}' '${helper}' merge '${selected}' '${params.dekupl}/share/dekupl/bin/mergeTags'
    """
}

process BUILD_RECRUITER {
    input: path source
    output: path 'recruit_reads'
    script:
    """
    module load devel rust/1.90.0
    rustc -O '${source}' -o recruit_reads
    """
}

process RECRUIT_READS {
    publishDir "${params.outdir}/recruited/${run}", mode: 'copy'
    tag "$run"
    cpus 5
    memory '16 GB'
    time '12h'
    input:
    tuple val(run), path(reads), path(validation)
    path selected
    path binary
    path helper
    output: tuple val(run), path("${run}_*.fq.gz"), path("${run}.hits.tsv.gz"), path("${run}.counts.json")
    script:
    """
    './${binary}' '${selected}/seeds.tsv' '${reads[0]}' '${reads[1]}' '${run}_1.fq.gz' '${run}_2.fq.gz' '${run}.hits.tsv.gz' > '${run}.counts.json'
    '${params.python}' '${helper}' check_recruit '${validation}' '${run}.counts.json'
    """
}

process SPLASH {
    cpus 16
    memory '96 GB'
    time '48h'
    input:
    path reads, stageAs: 'reads/*'
    path helper
    output: path 'splash'
    script:
    """
    '${params.python}' '${helper}' splash '${params.software}' ${task.cpus} reads
    """
}

process SELECT_SPLASH {
    publishDir "${params.outdir}/selected-splash", mode: 'copy'
    input:
    path splash
    path selector
    output: path 'splash-selected'
    script:
    """
    '${params.python}' '${selector}' '${splash}/full.after_correction.all_anchors.tsv' splash-selected
    cp -- '${splash}/full.after_correction.all_anchors.tsv' splash-selected/all-tested-anchors.tsv
    """
}

process COMPACT {
    tag "$source"
    cpus 16
    memory '96 GB'
    time '48h'
    input:
    tuple val(source), path(reads, stageAs: 'reads/*'), path(selected)
    path helper
    output: tuple val(source), path('context')
    script:
    """
    '${params.python}' '${helper}' compact '${source}' reads '${selected}' '${params.software}/compactors' ${task.cpus}
    """
}

process TRANSLATE {
    publishDir "${params.outdir}/queries/${source}", mode: 'copy'
    tag "$source"
    input:
    tuple val(source), path(context)
    path native_contexts, stageAs: 'native/??/*', arity: '3'
    path helper
    path translator
    output: tuple val(source), path('queries')
    script:
    """
    '${params.python}' '${helper}' translate '${source}' '${context}' ${native_contexts.collect { "'${it}'" }.join(' ')}
    """
}

process SKETCH {
    publishDir "${params.outdir}/sketches/${source}/${mode}", mode: 'copy'
    tag "$source:$mode"
    cpus 4
    memory '16 GB'
    time '12h'
    input:
    tuple val(source), path(queries), val(mode), val(k)
    path helper
    output: tuple val(source), val(mode), val(k), path('query.sig.zip'), path('coverage.json')
    script:
    """
    '${params.python}' '${helper}' sketch '${queries}' '${mode}' ${k} ${task.cpus}
    """
}

process SEARCH {
    tag "$source:$database:$mode:$scale"
    cpus 4
    memory '32 GB'
    time '12h'
    publishDir "${params.outdir}/matches/${source}/${database}/${mode}/${scale}", mode: 'copy'
    input:
    tuple val(source), val(mode), val(k), path(query), path(coverage), val(database), val(scale), path(target), path(protocol)
    path helper
    output: tuple val(source), val(database), val(mode), val(scale), path('search-result')
    script:
    """
    '${params.python}' '${helper}' search '${query}' '${target}' '${protocol}' '${source}' '${database}' '${mode}' ${k} ${scale} ${task.cpus}
    mkdir search-result
    mv raw-matches.csv matches.csv search.json search-result/
    """
}

process PREPARE_TARGET {
    tag "$database:$mode:$scale"
    cpus 1
    memory '32 GB'
    time '24h'
    publishDir "${params.outdir}/targets/${database}/${mode}/${scale}", mode: 'copy'
    input:
    tuple val(database), val(kind), val(mode), val(k), val(scale), path(sources, arity: '1..*')
    path helper
    path translator
    path validator
    output: tuple val(database), val(mode), val(k), val(scale), path('target.sig.zip'), path('target.protocol.json')
    script:
    def args = sources.collect { "'${it}'" }.join(' ')
    """
    '${params.python}' '${helper}' target '${database}' '${kind}' '${mode}' ${k} ${scale} '${validator}' ${args}
    """
}

process JOIN_EVIDENCE {
    memory '32 GB'
    time '12h'
    publishDir "${params.outdir}/evidence", mode: 'copy'
    input:
    path folders, stageAs: 'evidence/????/*'
    path reference
    path mapping
    path paper
    path annotations
    path helper
    path recovery_helper
    output: path 'gene-evidence.json'
    script:
    def args = folders.collect { "'${it}'" }.join(' ')
    """
    '${params.python}' '${helper}' evidence '${reference}' '${mapping}' '${paper}' '${annotations}' ${args}
    """
}

process ALIGN_CONTEXT {
    tag "$source"
    cpus 4
    memory '32 GB'
    time '12h'
    publishDir "${params.outdir}/sequence-evidence/${source}", mode: 'copy'
    input:
    tuple val(source), path(queries)
    path genome
    output: tuple val(source), path('alignment')
    script:
    def command = params.genome ? "'${params.minimap2}' -x splice -c --cs=long --secondary=yes -t ${task.cpus} '${genome}' '${queries}/dna.fasta' > alignment/context.paf" : "printf 'not measured: genome not supplied\\n' > alignment/status.txt"
    """
    mkdir alignment
    ${command}
    """
}

process QUERY_PFAM {
    tag "$source"
    cpus 4
    memory '16 GB'
    time '24h'
    publishDir "${params.outdir}/query-pfam/${source}", mode: 'copy'
    input:
    tuple val(source), path(queries)
    path hmm_files
    output: tuple val(source), path('pfam')
    script:
    def command = params.pfam_hmm ? "if test -s '${queries}/peptides.faa'; then '${params.hmmscan}' --cpu ${task.cpus} --cut_ga --domtblout pfam/domains.domtblout '${params.pfam_hmm.tokenize('/').last()}' '${queries}/peptides.faa' > pfam/hmmscan.txt; else printf 'no eligible peptides\\n' > pfam/status.txt; fi" : "printf 'not measured: Pfam HMM database not supplied\\n' > pfam/status.txt"
    """
    mkdir pfam
    ${command}
    """
}

workflow {
    for (name in ['inputs','design','samples','reference','gtf','targets','paper','annotations']) {
        if (!params[name]) error "Required parameter: --${name}"
    }
    if (params.pfam_hmm && !params.hmmscan) error 'A pinned --hmmscan executable is required with --pfam_hmm'
    def records = new JsonSlurper().parse(file(params.inputs))
    def design = new JsonSlurper().parse(file(params.design))
    if (records.size() != params.expected_libraries || records.collect{it.run}.unique().size() != records.size()) error 'Incomplete or duplicate validated libraries'
    def sampleIds = file(params.samples).readLines().drop(1).collect { it.split('\t')[0] }.toSet()
    if (sampleIds != records.collect{it.run}.toSet()) error 'Sample metadata and validated libraries differ'
    def sampleHash = java.security.MessageDigest.getInstance('SHA-256').digest(file(params.samples).bytes).encodeHex().toString()
    if (sampleHash != design.samples_sha256) error 'Frozen sample metadata changed'
    def inputs = Channel.fromList(records).map { r -> tuple(r.run, r.files.collect { file(it.path, checkIfExists:true) }, r) }
    def helper = file("${scripts}/nextflow_stages.py")
    BUILD_VALIDATOR(file("${scripts}/validate_pairs.rs"))
    VALIDATE_PAIR(inputs, helper, BUILD_VALIDATOR.out)
    COUNT_KMERS(VALIDATE_PAIR.out)
    JOIN_KMERS(COUNT_KMERS.out.map{run, counts -> counts}.collect())
    KALLISTO_INDEX(file(params.reference))
    QUANTIFY_GENES(VALIDATE_PAIR.out, KALLISTO_INDEX.out)
    GENE_COUNTS(QUANTIFY_GENES.out.map{run, q -> q}.collect(), file(params.reference), file(params.gtf), helper)
    POOL_COUNTS(JOIN_KMERS.out, GENE_COUNTS.out.counts, file(params.design), file("${scripts}/pool_abundance.py"))
    def pooled = POOL_COUNTS.out.flatten().map { p -> tuple(p.parent.name, p.name, p) }
    NORMALIZE(pooled)
    BLOCKED_MODEL(NORMALIZE.out, file("${scripts}/blocked_abundance.R"))
    SELECT_DE_SEEDS(BLOCKED_MODEL.out.filter{c,a,p -> a == 'kmers'}.map{c,a,p -> p}.collect(), helper)
    MERGE_TAGS(SELECT_DE_SEEDS.out.contrasts.flatten().map{p -> tuple(p.name,p)}, helper, file("${scripts}/dekupl_contigs.py"))
    BUILD_RECRUITER(file("${scripts}/recruit_dekupl_reads.rs"))
    RECRUIT_READS(VALIDATE_PAIR.out, SELECT_DE_SEEDS.out.all, BUILD_RECRUITER.out, helper)
    def allReads = VALIDATE_PAIR.out.map{r,reads,v -> reads}.flatten().collect()
    SPLASH(allReads, helper)
    SELECT_SPLASH(SPLASH.out, file("${scripts}/select_anchors.py"))
    // Both sequence sources enter the same translation and search processes.
    def contexts = Channel.of('splash').combine(allReads.map{[it]}).combine(SELECT_SPLASH.out)
        .mix(Channel.of('dekupl').combine(RECRUIT_READS.out.map{r,reads,h,c -> reads}.flatten().collect().map{[it]}).combine(SELECT_DE_SEEDS.out.all))
    COMPACT(contexts, helper)
    TRANSLATE(COMPACT.out, MERGE_TAGS.out.map{c,p -> p}.collect(), helper, file("${scripts}/dekupl_translation.py"))
    def settings = Channel.of(tuple('dna',31), tuple('protein',7), tuple('dayhoff',12), tuple('hp',30))
    SKETCH(TRANSLATE.out.combine(settings), helper)
    ALIGN_CONTEXT(TRANSLATE.out, params.genome ? file(params.genome,checkIfExists:true) : [])
    QUERY_PFAM(TRANSLATE.out, params.pfam_hmm ? ['', '.h3f', '.h3i', '.h3m', '.h3p'].collect { file(params.pfam_hmm+it,checkIfExists:true) } : [])
    def targetInputs = Channel.fromPath(params.targets).splitCsv(header:true,sep:'\t').map { r ->
        def targetSettings = [dna:31,protein:7,dayhoff:12,hp:30]
        if (!(r.mode in targetSettings.keySet()) || (r.k as int) != targetSettings[r.mode]) error 'Unsupported target sketch setting'
        if (!(r.kind in ['reads','protein']) || (r.kind == 'protein' && r.mode == 'dna')) error 'Unsupported target source'
        if (!(r.scaled as int in [5,10,100])) error 'Unsupported target scaled value'
        tuple(r.database, r.kind, r.mode, r.k as int, r.scaled as int, r.paths.split(';').collect{file(it,checkIfExists:true)})
    }
    PREPARE_TARGET(targetInputs, helper, file("${scripts}/dekupl_translation.py"), BUILD_VALIDATOR.out)
    def targets = PREPARE_TARGET.out.map{db,m,k,scale,target,protocol -> tuple(m,db,scale,target,protocol)}
    def searches = SKETCH.out.map{s,m,k,q,c -> tuple(m,s,k,q,c)}.combine(targets,by:0)
        .map{m,s,k,q,c,db,scale,target,protocol -> tuple(s,m,k,q,c,db,scale,target,protocol)}
    SEARCH(searches, helper)
    def evidenceInputs = TRANSLATE.out.map{s,q -> q}.mix(SEARCH.out.map{s,d,m,scale,p -> p})
        .mix(BLOCKED_MODEL.out.filter{c,a,p -> a=='genes'}.map{c,a,p -> p})
        .mix(ALIGN_CONTEXT.out.map{s,p -> p}).mix(QUERY_PFAM.out.map{s,p -> p}).collect()
    JOIN_EVIDENCE(evidenceInputs, file(params.reference), GENE_COUNTS.out.mapping, file(params.paper), file(params.annotations), helper, file("${scripts}/figure4_recovery.py"))
}
