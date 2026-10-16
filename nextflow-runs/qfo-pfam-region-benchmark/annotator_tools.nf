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
 * G2, reference-database tools: InterProScan (every member database) and eggNOG-mapper
 * (default DIAMOND search) annotate the queries against their own databases, which
 * contain human. Their rows are "graded on their own models", as hmmscan's is.
 *
 * G3, embedding search: TM-Vec (with DeepBLAST's structural alignment) and PLMSearch,
 * the queries against each QfO target proteome; best hit's Pfam domains from the
 * target's domain map. GPU tasks.
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
// G2, reference-database tools. Each runs only when named in --tools, and needs its data.
params.interproscan_dir = null   // unpacked interproscan-5.78-109.0 (make interproscan-data)
params.eggnog_data_dir  = null   // emapperdb 5.0.2 (make eggnog-data)
params.refdb_label      = 'own_models'
// Which tools run (comma-separated).
// G3, embedding search against each QfO target proteome of the midi-plus run.
// The nine QfO targets of the midi-plus run that have a domain map in
// data/midi-plus/results/truth (botryllus has none there; checked 2026-10-16 by -preview).
params.embedding_targets  = 'mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli'
params.species_registry   = "${projectDir}/assets/qfo_species.tsv"
params.qfo_dir            = null   // data/midi-plus/qfo
params.domain_map_dir     = null   // data/midi-plus/results/truth (<species>_domain_map.parquet)
params.tmvec_image        = null   // built from Dockerfile.tmvec
params.tmvec_model        = null   // TM-Vec checkpoint (.ckpt)
params.tmvec_config       = null   // its params .json
params.prott5_dir         = null   // ProtT5-XL-UniRef50 weights folder
params.deepblast_model    = null   // DeepBLAST checkpoint; null = no alignment, no positions
params.plmsearch_image    = null   // built from Dockerfile.plmsearch
params.plmsearch_model_dir = null  // plmsearch_data/model (plmsearch.sav, esm/)
params.tmvec_setting      = params.deepblast_model ? 'cath_deepblast' : 'cath'
params.plmsearch_setting  = 'ss_predictor'
// Which tools run (comma-separated).
params.tools = 'diamond,mmseqs2,interproscan,eggnog,tmvec,plmsearch'

// After the params, not before: a module reads params when it is included, so a param
// set below an include is undefined inside it (the stub run caught --min-overlap null).
include { cladeExcludedReference; hitTargets; hmmscanHitTargets;
          placeBestHitDomains; scoreAnnotatorTool } from './modules/annotator_common'
include { DIAMOND_BEST_HIT } from './modules/diamond_best_hit'
include { MMSEQS_BEST_HIT }  from './modules/mmseqs_best_hit'
include { INTERPROSCAN }     from './modules/interproscan'
include { EGGNOG_MAPPER }    from './modules/eggnog_mapper'
include { shortHeaders; TMVEC; PLMSEARCH } from './modules/embedding_search'

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
    def unknown = tools - ['diamond', 'mmseqs2', 'interproscan', 'eggnog', 'tmvec', 'plmsearch']
    if (unknown) error "unknown --tools: ${unknown}"

    hits = Channel.empty()
    def best_hit_tools = tools.intersect(['diamond', 'mmseqs2'])
    ref = best_hit_tools ? cladeExcludedReference(dat, params.exclude_clade) : null
    if ('diamond' in tools) {
        hits = hits.mix(DIAMOND_BEST_HIT(ref.fasta, queries,
                                         params.diamond_settings.tokenize(',')).hits)
    }
    if ('mmseqs2' in tools) {
        hits = hits.mix(MMSEQS_BEST_HIT(ref.fasta, queries, params.mmseqs_sensitivity).hits)
    }

    regions = Channel.empty()
    if (best_hit_tools) {
        targets = hitTargets(hits.map { it[2] }.collect(), ref.fasta)
        domains = hmmscanHitTargets(targets.map { [it, hmm, hmm_aux] })
        placed = placeBestHitDomains(hits.combine(domains).combine(queries))
        regions = placed.regions
            .map { tool, setting, r -> [tool, setting, params.species_label, r] }
    }

    // Reference-database tools: their calls are already on the query, from their own
    // models, so there is no reference, no best hit and no placement step.
    if ('interproscan' in tools) {
        if (params.interproscan_dir == null) error "--interproscan_dir is required for interproscan"
        file("${params.interproscan_dir}/data", checkIfExists: true)
        regions = regions.mix(INTERPROSCAN(queries).regions
            .map { tool, setting, r -> [tool, setting, params.refdb_label, r] })
    }
    if ('eggnog' in tools) {
        def edir = required('eggnog_data_dir')
        regions = regions.mix(EGGNOG_MAPPER(queries, Channel.value(edir), Channel.value(hmm)).regions
            .map { tool, setting, r -> [tool, setting, params.refdb_label, r] })
    }

    // Embedding search: one task per target proteome; the species is the row's target.
    def emb_tools = tools.intersect(['tmvec', 'plmsearch'])
    if (emb_tools) {
        def species = params.embedding_targets.tokenize(',')
        def registry = file(params.species_registry).readLines().drop(1)
            .collect { it.split('\t') }
            .collectEntries { [(it[0]): it] }
        def target_rows = species.collect { sp ->
            def r = registry[sp]
            if (r == null) error "${sp} is not in ${params.species_registry}"
            def fasta = file("${required('qfo_dir')}/${r[3]}/${r[2]}_${r[1]}.fasta", checkIfExists: true)
            def map = file("${required('domain_map_dir')}/${sp}_domain_map.parquet", checkIfExists: true)
            [sp, fasta, map]
        }
        shorts = shortHeaders(
            Channel.fromList(target_rows.collect { [it[0], it[1]] })
                .mix(queries.map { ['query', it] })
        )
        q_short = shorts.filter { it[0] == 'query' }.map { it[1] }
        maps = Channel.fromList(target_rows.collect { [it[0], it[2]] })
        t_short = shorts.filter { it[0] != 'query' }.combine(maps, by: 0)

        if ('tmvec' in emb_tools) {
            if (params.tmvec_image == null) error "--tmvec_image is required for tmvec (build Dockerfile.tmvec)"
            def models = Channel.value([
                required('tmvec_model'), required('tmvec_config'), required('prott5_dir'),
                params.deepblast_model ? required('deepblast_model')
                                       : file("${projectDir}/assets/NO_DEEPBLAST")])
            regions = regions.mix(TMVEC(t_short, q_short, models).regions)
        }
        if ('plmsearch' in emb_tools) {
            if (params.plmsearch_image == null) error "--plmsearch_image is required for plmsearch (build Dockerfile.plmsearch)"
            regions = regions.mix(PLMSEARCH(t_short, q_short,
                                             Channel.value(required('plmsearch_model_dir'))).regions)
        }
    }

    scoreAnnotatorTool(regions.combine(truth).combine(covariates))
}
