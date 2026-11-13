#!/usr/bin/env nextflow
// Re-score saved search output without invoking simulation, assembly, DE, or searches.
nextflow.enable.dsl = 2
params.key_dir = '/scratch/users/olgabot/360-de-kmer-landing/key'
params.organisms = 'ciona,zebrafish'
params.only_settings = 'gbmr7.k11'
params.settings = "${projectDir}/assets/kmerseek_settings.tsv"
params.labellings = "${projectDir}/assets/labellings.tsv"
params.outdir = "${launchDir}/results"
include { landings; collectSummaries } from './main.nf'

workflow {
    def wanted = params.only_settings.tokenize(',')
    def lines = file(params.settings).readLines().findAll { it.trim() && !it.startsWith('#') }
    def head = lines[0].split('\t') as List
    def settings = lines.drop(1).collect { line ->
        [head, line.split('\t') as List].transpose().collectEntries { it }
    }.findAll { "${it.alphabet}.k${it.ksize}".toString() in wanted }
    if (settings.size() != wanted.size()) error 'Unknown repair setting'
    def organisms = params.organisms.tokenize(',')
    inputs = Channel.fromList(organisms).combine(Channel.fromList(settings)).map { org, s ->
        def slug = "${org}.${s.alphabet}.k${s.ksize}"
        tuple(org, s,
              file("${params.outdir}/${org}/kmerseek/${slug}.regions.csv.zst", checkIfExists: true),
              file("${params.outdir}/${org}/kmerseek/${slug}.status", checkIfExists: true),
              file("${params.outdir}/${org}/kmer_path/${org}.peptide_map.tsv", checkIfExists: true),
              file("${params.key_dir}/${org}/search_db.pfam.tsv", checkIfExists: true),
              file("${params.key_dir}/${org}/answer_key.tsv", checkIfExists: true),
              file(params.labellings, checkIfExists: true))
    }
    repaired = landings(inputs).summary
    existing = Channel.fromPath("${params.outdir}/*/{standard,landings}/*.summary.tsv", checkIfExists: true).filter { f ->
        !organisms.any { org -> wanted.any { setting -> f.name == "${org}.${setting}.summary.tsv" } }
    }
    collectSummaries(existing.mix(repaired).collect())
}
