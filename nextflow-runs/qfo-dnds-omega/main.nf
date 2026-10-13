#!/usr/bin/env nextflow

/*
 * qfo-dnds-omega/main.nf
 *
 * Per-ortholog-pair omega (pairwise dN/dS, PAML codeml) for human against QfO
 * target species, computed from the cDNA that ships with the QfO release rather
 * than fetched from Ensembl. Feeds --omega_file in the qfo-pfam-region-benchmark
 * pipeline, which reads it through bin/build_query_covariates.py.
 *
 * This supersedes nextflow-runs/human-mouse-dnds-omega for three reasons: that
 * pipeline is human-mouse only, its ortholog pairing comes from MGI and so does not
 * generalise past mouse, and its codeml parser reports omega in the dS column (see
 * bin/compute_omega_chunk.py for the exact cause and the fix). That pipeline is left
 * untouched; notebook 206 still cites its numbers.
 *
 * WHICH SPECIES THIS IS RUN FOR, AND WHY IT IS NOT ALL NINE
 *
 * omega is dN/dS, so it is only as meaningful as dS. Synonymous sites saturate: once
 * they have been hit often enough that the observed difference between two sequences
 * stops growing with time, no correction recovers the substitution count, and dS
 * becomes an extrapolation from a number that has stopped carrying information.
 *
 * Measured on 120 sampled 1:1 OMA ortholog pairs per species from this release,
 * using Nei-Gojobori synonymous-site counting. `undefined` is the fraction of pairs
 * where the observed proportion of synonymous differences reached the 3/4 ceiling at
 * which the Jukes-Cantor correction has no solution at all:
 *
 *   species      MYA    median protein id   median dS   dS undefined
 *   mouse        100          0.904            0.61          0.8%
 *   chicken      300          0.765            1.49         14.2%
 *   zebrafish    430          0.679            2.14         71.7%
 *   ciona        550          0.475            3.5+         98.3%
 *   fly          600          0.459            1.92         68.3%
 *   worm         650          0.422            2.31         93.3%
 *   yeast        900          0.384            3.13         95.8%
 *   arabidopsis 1500          0.390            2.47         95.8%
 *   ecoli       2000            n/a             n/a           n/a
 *
 * The mouse median dS of 0.61 matches the published human-mouse figure and matches
 * what the older pipeline's codeml run gives once its dS is reconstructed, which is
 * a check on the measurement rather than a claim about the measurement.
 *
 * From zebrafish outward the median pair is at or past the ceiling: median observed
 * synonymous difference is 0.787 for zebrafish and 0.82-0.85 beyond it, against a
 * random-sequence expectation near 0.75. codeml will not refuse these. It returns a
 * dS, and an omega, and they look like data. They are not, so the default species
 * list stops at chicken.
 *
 * ecoli is excluded for a second, independent reason: no ortholog source in the
 * release pairs it to human at all. Its .idmapping carries no OrthoDB, OMA, GeneTree
 * or TreeFam cross-references, and its eggNOG groups share none with human's. The
 * only overlap is KEGG Orthology, where 206 groups happen to hold exactly one human
 * and one ecoli protein. That co-occurrence is a shared enzymatic function, not an
 * orthology call, and inventing pairs from it would not be defensible.
 *
 * --species overrides the default list for anyone who wants the saturated species
 * anyway. Those rows arrive with dS_saturated=true, which is the point of the column.
 *
 * Two stages:
 *   1. buildPairs        - one task per species; reads the QfO release directly,
 *                          emits 1:1 ortholog pairs plus chunked codon FASTAs.
 *   2. computeOmegaChunk - one task per chunk of ~250 pairs; MAFFT, codon
 *                          back-translation, codeml runmode=-2.
 *
 * Output, under --outdir:
 *   omega_all_species.tsv          every species, with a species column
 *   omega.<species>.tsv            one species; this is what --omega_file wants
 *   pairs.<species>.tsv            the pair manifest, including pairs that failed
 *   summary.<species>.json         per-species counts from the pairing step
 *
 * build_query_covariates.py joins omega onto a single accession per human protein and
 * keeps the first match, so handing it the combined file would let it pick a species
 * arbitrarily. Point --omega_file at one of the per-species files.
 */

nextflow.enable.dsl = 2

params.qfo_dir         = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir          = 'results'
params.omega_container = 'qfo-dnds-omega:1.0'
params.species         = null   // comma-separated; default is the defensible set below
params.ortholog_source = 'OMA'
params.chunk_size      = 250
params.min_codons      = 50
params.max_pairs       = 0      // 0 = no cap; >0 truncates each species, for smoke tests
params.ds_saturated_above = 2.0

// Human is the query for every pair. kmerseek scores regions query-side, so the
// benchmark's covariates are per human protein and human cannot be a target here.
def HUMAN = [proteome: 'UP000005640_9606', subdir: 'Eukaryota']

// icode is PAML's genetic code index: 0 is the universal code, 10 is the bacterial
// and plant-plastid code (NCBI translation table 11). Only ecoli would need 10, and
// it is unreachable for other reasons, but carrying the field keeps the assumption
// visible instead of hard-coded.
def ALL_SPECIES = [
    [label: 'mouse',       proteome: 'UP000000589_10090',  subdir: 'Eukaryota', mya: 100,  icode: 0],
    [label: 'chicken',     proteome: 'UP000000539_9031',   subdir: 'Eukaryota', mya: 300,  icode: 0],
    [label: 'zebrafish',   proteome: 'UP000000437_7955',   subdir: 'Eukaryota', mya: 430,  icode: 0],
    [label: 'ciona',       proteome: 'UP000008144_7719',   subdir: 'Eukaryota', mya: 550,  icode: 0],
    [label: 'fly',         proteome: 'UP000000803_7227',   subdir: 'Eukaryota', mya: 600,  icode: 0],
    [label: 'worm',        proteome: 'UP000001940_6239',   subdir: 'Eukaryota', mya: 650,  icode: 0],
    [label: 'yeast',       proteome: 'UP000002311_559292', subdir: 'Eukaryota', mya: 900,  icode: 0],
    [label: 'arabidopsis', proteome: 'UP000006548_3702',   subdir: 'Eukaryota', mya: 1500, icode: 0],
    [label: 'ecoli',       proteome: 'UP000000625_83333',  subdir: 'Bacteria',  mya: 2000, icode: 10],
]

// Where dS is still informative. See the saturation table in the header.
def DEFENSIBLE = ['mouse', 'chicken']

def requested = params.species ? params.species.tokenize(',')*.trim() : DEFENSIBLE
def SPECIES = ALL_SPECIES.findAll { it.label in requested }

if (SPECIES.isEmpty()) {
    error "No target species matched '${requested.join(',')}'. Known targets: " +
          "${ALL_SPECIES*.label.join(', ')} (human is the query and is not a target)"
}

def saturated = SPECIES*.label.findAll { !(it in DEFENSIBLE) }
if (saturated) {
    log.warn "Synonymous sites are saturated for ${saturated.join(', ')}; " +
             "dS and therefore omega are not interpretable for those pairs. " +
             "Rows will carry dS_saturated=true."
}

log.info """
qfo-dnds-omega
  qfo_dir         : ${params.qfo_dir}
  species         : ${SPECIES*.label.join(', ')}
  ortholog_source : ${params.ortholog_source}
  chunk_size      : ${params.chunk_size}
  outdir          : ${params.outdir}
"""

process buildPairs {
    tag "${species}"
    label 'pairing'
    container params.omega_container
    publishDir params.outdir, mode: 'copy', pattern: '{pairs,summary}.*'

    input:
    tuple val(species), val(mya), val(icode),
          path(human_dna), path(human_protein), path(human_idmapping),
          path(target_dna), path(target_idmapping)

    output:
    tuple val(species), val(icode), path("pairs.${species}.tsv"), emit: manifest
    path "pairs.${species}.tsv",    emit: manifest_only
    path "summary.${species}.json", emit: summary
    // arity '1..*' is load-bearing, not decoration. A glob output emits a bare Path when
    // exactly one file matches and a List when several do, so `transpose(by: 3)` below
    // died with "Not a valid transpose element at index: 3" the first time a species
    // produced a single chunk. That is the normal case for `make test`, which uses 20
    // pairs against a chunk_size of 20, and it would also hit any real species whose pair
    // count lands under one chunk. Declaring the arity makes it a List either way.
    tuple val(species), val(icode), path("pairs.${species}.tsv"),
          path("chunk_*.${species}.fa", arity: '1..*'), emit: chunks

    script:
    def cap = params.max_pairs ? "--max_pairs ${params.max_pairs}" : ''
    """
    build_ortholog_pairs.py \\
        --human_dna ${human_dna} \\
        --human_protein ${human_protein} \\
        --human_idmapping ${human_idmapping} \\
        --target_dna ${target_dna} \\
        --target_idmapping ${target_idmapping} \\
        --species ${species} \\
        --mya ${mya} \\
        --ortholog_source ${params.ortholog_source} \\
        --min_codons ${params.min_codons} \\
        --chunk_size ${params.chunk_size} \\
        ${cap} \\
        --outdir .
    """
}

process computeOmegaChunk {
    tag "${species}:${chunk.simpleName}"
    label 'codeml'
    container params.omega_container
    // A chunk that dies takes ~250 pairs with it, which is worth a retry but not
    // worth stopping a nine-species run over.
    errorStrategy { task.attempt <= 2 ? 'retry' : 'ignore' }
    maxRetries 2

    input:
    tuple val(species), val(icode), path(manifest), path(chunk)

    output:
    tuple val(species), path("omega.${species}.${chunk.simpleName}.tsv")

    script:
    """
    compute_omega_chunk.py \\
        --chunk_fasta ${chunk} \\
        --manifest ${manifest} \\
        --icode ${icode} \\
        --ds_saturated_above ${params.ds_saturated_above} \\
        --outfile omega.${species}.${chunk.simpleName}.tsv \\
        --workdir codeml_work
    """
}

workflow {
    def qfo = file(params.qfo_dir, checkIfExists: true)
    def human_dna        = file("${qfo}/${HUMAN.subdir}/${HUMAN.proteome}_DNA.fasta",   checkIfExists: true)
    def human_protein    = file("${qfo}/${HUMAN.subdir}/${HUMAN.proteome}.fasta",       checkIfExists: true)
    def human_idmapping  = file("${qfo}/${HUMAN.subdir}/${HUMAN.proteome}.idmapping",   checkIfExists: true)

    species_ch = Channel.fromList(SPECIES).map { s ->
        tuple(
            s.label, s.mya, s.icode,
            human_dna, human_protein, human_idmapping,
            file("${qfo}/${s.subdir}/${s.proteome}_DNA.fasta", checkIfExists: true),
            file("${qfo}/${s.subdir}/${s.proteome}.idmapping", checkIfExists: true),
        )
    }

    pairs = buildPairs(species_ch)

    // buildPairs emits all of a species' chunks in one tuple; transpose turns that
    // into one tuple per chunk so codeml runs fan out.
    chunk_ch = pairs.chunks.transpose(by: 3)

    omega_ch = computeOmegaChunk(chunk_ch)

    // One file per species, which is the shape --omega_file expects.
    omega_ch
        .collectFile(storeDir: params.outdir, keepHeader: true, skip: 1, sort: true) { species, tsv ->
            ["omega.${species}.tsv", tsv]
        }
        .subscribe { f -> log.info "Wrote ${f}" }

    // Plus one combined file. Every row carries a species column, so this is the
    // file to read for anything comparing across species.
    omega_ch
        .map { species, tsv -> tsv }
        .collectFile(name: 'omega_all_species.tsv', storeDir: params.outdir,
                     keepHeader: true, skip: 1, sort: true)
        .subscribe { f -> log.info "Wrote ${f}" }
}
