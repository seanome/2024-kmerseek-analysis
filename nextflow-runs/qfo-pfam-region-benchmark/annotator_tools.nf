#!/usr/bin/env nextflow
/*
 * Annotator tools as leaderboard rows (prompt G of the 2026-10-16 leaderboard plan).
 *
 * G1, best-hit transfer: what most genome annotation pipelines do first. The human
 * midi-plus queries (998 proteins) are searched against Swiss-Prot with Mammalia removed,
 * by DIAMOND (--sensitive, --ultra-sensitive) and MMseqs2 (-s 7.5, one round). Each
 * query's best hit gives its Pfam domains, placed on the query through the alignment.
 * The calls are scored by evaluate_domain_calls.py against the midi-plus Pfam truth,
 * the same truth and covariates as the main run's rows.
 *
 *   queries ──► search (one module per tool) ──► best hit per query
 *                                                   │
 *   Swiss-Prot ─► minus Mammalia ──────────────────►│──► hmmscan the hit proteins
 *                                                   ▼
 *                         place hit domains on the query ──► score ──► rows, cards
 *
 * Launch: `make run-annotator-tools` (Sherlock); `make stub` checks the wiring on a
 * laptop. Rows and cards: `make annotator-rows` after the run.
 */

nextflow.enable.dsl = 2

params.queries       = null   // midi-plus human query FASTA (998 proteins)
params.truth         = null   // human_domain_truth.parquet from the midi-plus run
params.covariates    = null   // human_query_covariates.parquet from the midi-plus run
params.swissprot_dat = null   // uniprot_sprot.dat.gz, release 2026_03
params.pfam_hmm      = null   // Pfam-A.hmm, pressed (.h3f/.h3i/.h3m/.h3p beside it)
params.outdir        = null

params.exclude_clade      = 'Mammalia'
params.species_label      = 'swissprot_minus_mammalia'
params.diamond_settings   = 'sensitive,ultra-sensitive'
params.mmseqs_sensitivity = 7.5
// The pipeline's transfer rule: a hit must cover half of a target domain to transfer it.
params.min_overlap = 0.5
params.strict_iou  = 0.8
// Which tools run (comma-separated).
params.tools = 'diamond,mmseqs2'

// After the params, not before: a module reads params when it is included, so a param
// set below an include is undefined inside it (the stub run caught --min-overlap null).
include { cladeExcludedReference; hitTargets; hmmscanHitTargets;
          placeBestHitDomains; scoreAnnotatorTool } from './modules/annotator_common'
include { DIAMOND_BEST_HIT } from './modules/diamond_best_hit'
include { MMSEQS_BEST_HIT }  from './modules/mmseqs_best_hit'

def required(name) {
    if (params[name] == null) {
        error "--${name} is required (see `make run-annotator-tools` for the Sherlock paths)"
    }
    file(params[name], checkIfExists: true)
}

workflow {
    queries    = Channel.value(required('queries'))
    truth      = Channel.value(required('truth'))
    covariates = Channel.value(required('covariates'))
    dat        = required('swissprot_dat')
    hmm        = required('pfam_hmm')
    if (params.outdir == null) error "--outdir is required"
    hmm_aux = file("${params.pfam_hmm}.h3*")
    if (hmm_aux.size() != 4) {
        error "expected 4 hmmpress files beside ${params.pfam_hmm}, found ${hmm_aux.size()}"
    }
    tools = params.tools.tokenize(',')
    def unknown = tools - ['diamond', 'mmseqs2']
    if (unknown) error "unknown --tools: ${unknown}"

    ref = cladeExcludedReference(dat, params.exclude_clade)

    hits = Channel.empty()
    if ('diamond' in tools) {
        hits = hits.mix(DIAMOND_BEST_HIT(ref.fasta, queries,
                                         params.diamond_settings.tokenize(',')).hits)
    }
    if ('mmseqs2' in tools) {
        hits = hits.mix(MMSEQS_BEST_HIT(ref.fasta, queries, params.mmseqs_sensitivity).hits)
    }

    targets = hitTargets(hits.map { it[2] }.collect(), ref.fasta)
    domains = hmmscanHitTargets(targets.map { [it, hmm, hmm_aux] })

    placed = placeBestHitDomains(hits.combine(domains).combine(queries))
    scoreAnnotatorTool(placed.regions.combine(truth).combine(covariates))
}
