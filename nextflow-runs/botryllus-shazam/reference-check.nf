nextflow.enable.dsl=2
include { ALIGN_CONTEXT; QUERY_PFAM; SPLIT_PFAM } from './main.nf'
workflow {
    def inputs = Channel.of(tuple('positive', file(params.positive)), tuple('empty', file(params.empty)))
    def helper = file("${projectDir}/../../scripts/botryllus_shazam/nextflow_stages.py")
    ALIGN_CONTEXT(inputs, file(params.genome, checkIfExists:true), helper)
    SPLIT_PFAM(inputs, helper)
    QUERY_PFAM(SPLIT_PFAM.out.flatMap{s, files -> files.collect{p -> tuple(s,p.baseName,p)}}, ['', '.h3f', '.h3i', '.h3m', '.h3p'].collect { file(params.pfam_hmm+it, checkIfExists:true) }, helper)
}
