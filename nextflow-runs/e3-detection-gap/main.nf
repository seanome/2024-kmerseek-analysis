#!/usr/bin/env nextflow

/*
 * e3-detection-gap
 *
 * The baseline arms that notebook 340 (E3, domain-level detection-gap benchmark)
 * needs but that are NOT already on disk.  Arms that already have data live in
 * ../pfam-benchmark-tools (DIAMOND, MMseqs2 seq-seq, MMseqs2 iterative, phmmer)
 * and ../qfo-pfam-benchmark (kmerseek HP) — this pipeline does not duplicate them.
 *
 * Arms produced here, in the priority order set by the experiment plan:
 *
 *   1. jackhmmer  — 3 iterations against a large sequence database (UniRef30/UniRef50),
 *                   then the converged profile is searched against each species
 *                   proteome.  This is the sequence-only state of the art and the
 *                   comparison the hypothesis lives or dies on.
 *                   (HHblits is the sibling arm; it does NOT need new code — see
 *                   README.md, it only needs ../pfam-benchmark-tools rerun with
 *                   --hhblits_db pointing at a real UniRef30.)
 *
 *   2. interproscan — Pfam member-database scan of each species proteome.  This is
 *                   the ANNOTATION CEILING, not a competing arm: notebook 340's
 *                   ground truth is itself Pfam-derived, so this recovers ~100% by
 *                   construction.  It is run so the ceiling is measured rather than
 *                   assumed.
 *
 *   3. foldseek   — structure arm over the AlphaFold models already downloaded to
 *                   results/pfam_benchmark/alphafold_structures/all_species
 *                   (54_339 .cif files, no new download needed).
 *
 *   4. esm2       — windowed per-residue PLM embedding search.  GPU arm; scaffolded
 *                   here and handed off, deliberately not run on the laptop.
 *
 * Every arm emits the same 4-column TSV that notebook 340's harness reads:
 *
 *     human_accession <TAB> species_accession <TAB> score <TAB> evalue
 *
 * gzipped to ${params.outdir}/<arm>/human_vs_<species>.<arm>.tsv.gz
 *
 * Usage (single dash on -resume; --resume is a pipeline param, not a Nextflow flag):
 *
 *   nextflow run main.nf -profile local -resume \
 *       --arms foldseek
 *
 *   nextflow run main.nf -profile local -resume \
 *       --arms jackhmmer --jackhmmer_db $HOME/data/uniref/uniref50.fasta
 *
 * NOTE ON CONTAINERS: no ENTRYPOINT in any image used here — Nextflow runs
 * /bin/bash inside the container and an ENTRYPOINT breaks it.  No `/usr/bin/time -l`
 * anywhere either; that flag is macOS-only and Debian silently ignores it, which
 * shows up later as a missing-output error.
 */

nextflow.enable.dsl = 2

def home = System.getProperty('user.home')

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

params.qfo_dir      = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.repo_dir     = "${projectDir}/../.."
params.structure_dir = "${params.repo_dir}/results/pfam_benchmark/alphafold_structures/all_species"
params.accession_list = "${params.repo_dir}/results/pfam_benchmark/pairs_stratified/all_accessions_for_af2.tsv"
params.outdir       = "${home}/data/e3-detection-gap"

// Comma-separated subset of: jackhmmer,interproscan,foldseek,esm2
params.arms         = "foldseek"

// Comma-separated species labels, or "all"
params.species      = "ciona,fly,worm"

// jackhmmer / HHblits background database.  UniRef50 is ~15 GB and adequate;
// UniRef30 (the HH-suite clustered DB) is ~50 GB.  Egress, not compute, is the
// budget constraint on this project — prefer UniRef50 unless the reviewer asks
// specifically for UniRef30.
params.jackhmmer_db = null
params.jackhmmer_iterations = 3
params.evalue_report = 10.0

// InterProScan 5 installation directory (interproscan.sh lives here)
params.interproscan_dir = "${home}/data/interproscan"

// ESM-2 checkpoint; the 650M model is the usual size/quality compromise
params.esm2_model   = "esm2_t33_650M_UR50D"
params.esm2_window  = 32
params.esm2_stride  = 16

params.python       = "${home}/anaconda3/envs/2025-kmerseek-analysis/bin/python3"

// ---------------------------------------------------------------------------
// Species table — must match ../pfam-benchmark-tools/main.nf exactly, otherwise
// the arms cannot be joined on species label in notebook 340.
// ---------------------------------------------------------------------------

def SPECIES = [
    [label: "mouse",       taxon: "10090",  proteome: "UP000000589", subdir: "Eukaryota", mya: 100],
    [label: "chicken",     taxon: "9031",   proteome: "UP000000539", subdir: "Eukaryota", mya: 320],
    [label: "zebrafish",   taxon: "7955",   proteome: "UP000000437", subdir: "Eukaryota", mya: 450],
    [label: "ciona",       taxon: "7719",   proteome: "UP000008144", subdir: "Eukaryota", mya: 550],
    [label: "fly",         taxon: "7227",   proteome: "UP000000803", subdir: "Eukaryota", mya: 800],
    [label: "worm",        taxon: "6239",   proteome: "UP000001940", subdir: "Eukaryota", mya: 800],
    [label: "yeast",       taxon: "559292", proteome: "UP000002311", subdir: "Eukaryota", mya: 1100],
    [label: "arabidopsis", taxon: "3702",   proteome: "UP000006548", subdir: "Eukaryota", mya: 1500],
    [label: "ecoli",       taxon: "83333",  proteome: "UP000000625", subdir: "Bacteria",  mya: 2000],
]

def HUMAN = [label: "human", taxon: "9606", proteome: "UP000005640", subdir: "Eukaryota"]

def speciesFasta(rec) {
    file("${params.qfo_dir}/${rec.subdir}/${rec.proteome}_${rec.taxon}.fasta")
}

// ---------------------------------------------------------------------------
// PROCESSES — FoldSeek (structure arm)
// ---------------------------------------------------------------------------

process foldseekCollectStructures {
    /*
     * Symlink the AlphaFold .cif models belonging to one proteome into a flat
     * directory so `foldseek createdb` can consume it.
     *
     * The models are already on disk; nothing is downloaded here.  Accessions with
     * no model are reported in the .missing file and are counted in notebook 340 as
     * FoldSeek misses — never as "outside the benchmark".
     */
    tag "${label}"
    label 'low_cpu'
    publishDir "${params.outdir}/foldseek/inventory", mode: 'copy', pattern: '*.missing.txt'

    input:
    tuple val(label), path(fasta)

    output:
    tuple val(label), path("${label}_structures"), emit: dir
    path "${label}.missing.txt",                   emit: missing

    script:
    """
    #!${params.python}
    import os
    import re
    from pathlib import Path

    struct_dir = Path("${params.structure_dir}")
    out = Path("${label}_structures")
    out.mkdir(exist_ok=True)

    # UniProt fasta headers look like  >sp|Q9NR96|TLR9_HUMAN ...
    accs = []
    with open("${fasta}") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m = re.match(r">(?:sp|tr)\\|([A-Z0-9]+)\\|", line)
            if m:
                accs.append(m.group(1))
    accs = sorted(set(accs))

    missing = []
    n_linked = 0
    for acc in accs:
        hits = list(struct_dir.glob(f"AF-{acc}-F1-model_*.cif"))
        if not hits:
            missing.append(acc)
            continue
        os.symlink(hits[0], out / hits[0].name)
        n_linked += 1

    with open("${label}.missing.txt", "w") as fh:
        fh.write("\\n".join(missing) + ("\\n" if missing else ""))

    print(f"${label}: linked {n_linked} structures, {len(missing)} accessions have no AFDB model")
    """
}

process foldseekCreateDB {
    tag "${label}"
    container 'quay.io/biocontainers/foldseek:10.941cd33--h1e1f9c8_0'
    label 'medium_cpu'

    input:
    tuple val(label), path(struct_dir)

    output:
    tuple val(label), path("${label}_db*")

    script:
    """
    foldseek createdb ${struct_dir} ${label}_db --threads ${task.cpus}
    """
}

process foldseekSearch {
    /*
     * Structure-structure search, human queries against one species proteome.
     * -e is deliberately loose: notebook 340 calibrates every arm to a matched
     * false-positive rate itself, so pre-filtering here would silently hand
     * FoldSeek a different operating point than the other arms.
     */
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/foldseek:10.941cd33--h1e1f9c8_0'
    label 'high_cpu'
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_db), path(human_db)

    output:
    tuple val(species), val("foldseek"), path("human_vs_${species}.foldseek.tsv.gz")

    script:
    """
    mkdir -p tmp
    foldseek search human_db ${species}_db result tmp \\
        -e ${params.evalue_report} \\
        --max-seqs 4000 \\
        --threads ${task.cpus}

    foldseek convertalis human_db ${species}_db result result.tsv \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus}

    # AF-<ACC>-F1-model_v6.cif -> <ACC> so the ids join to the Pfam ground truth
    sed -E 's/AF-([A-Z0-9]+)-F1-model_v[0-9]+(\\.cif)?/\\1/g' result.tsv \\
        | gzip -c > human_vs_${species}.foldseek.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — jackhmmer (priority-1 sequence-only baseline)
// ---------------------------------------------------------------------------

process jackhmmerBuildProfiles {
    /*
     * Iterate the human queries against a large background database to build
     * converged profiles.  This is the expensive step: ~20k human queries x N
     * iterations against UniRef50.  Run it once, reuse for all 9 species.
     */
    container 'quay.io/biocontainers/hmmer:3.4--hdbdd923_1'
    label 'very_high_cpu'
    publishDir "${params.outdir}/jackhmmer/profiles", mode: 'copy'

    input:
    path human_fasta
    path background_db

    output:
    path "human_converged.hmm"

    script:
    """
    jackhmmer \\
        -N ${params.jackhmmer_iterations} \\
        --chkhmm human_chk \\
        --noali --notextw \\
        --cpu ${task.cpus} \\
        -o /dev/null \\
        ${human_fasta} ${background_db}

    # keep only the last checkpoint per query (the converged profile)
    cat human_chk-${params.jackhmmer_iterations}.hmm > human_converged.hmm
    """
}

process jackhmmerSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hmmer:3.4--hdbdd923_1'
    label 'high_cpu'
    publishDir "${params.outdir}/jackhmmer", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta), path(human_hmm)

    output:
    tuple val(species), val("jackhmmer"), path("human_vs_${species}.jackhmmer.tsv.gz")

    script:
    """
    hmmsearch \\
        --tblout hits.tbl \\
        --noali --notextw \\
        -E ${params.evalue_report} \\
        --cpu ${task.cpus} \\
        -o /dev/null \\
        ${human_hmm} ${species_fasta}

    # tblout columns: target(0) accession(1) query(2) accession(3) fullseq-evalue(4) score(5)
    # emit query=human, target=species, then strip sp|ACC|NAME -> ACC
    grep -v '^#' hits.tbl \\
        | awk '{print \$3 "\\t" \$1 "\\t" \$6 "\\t" \$5}' \\
        | sed -E 's/(sp|tr)\\|([A-Z0-9]+)\\|[A-Za-z0-9_]+/\\2/g' \\
        | gzip -c > human_vs_${species}.jackhmmer.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — InterProScan Pfam scan (annotation ceiling, NOT a competing arm)
// ---------------------------------------------------------------------------

process interproscanPfam {
    /*
     * Domain annotation of one species proteome with the Pfam member database.
     *
     * Circular with notebook 340's ground truth by construction — that is the
     * point.  Running it measures the ceiling instead of assuming it, and the
     * difference between "UniProt says this protein has PF00069" and "a fresh
     * hmmscan finds PF00069 here" is itself worth reporting.
     *
     * Uses the local InterProScan install (${params.interproscan_dir}) rather than
     * a container, because the member-database data directory is ~50 GB and is
     * already unpacked at ~/data/interproscan and ~/data/interproscan6.
     */
    tag "${species}"
    label 'very_high_cpu'
    publishDir "${params.outdir}/interproscan", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta)

    output:
    tuple val(species), val("interproscan"), path("${species}.pfam.tsv.gz")

    script:
    """
    # InterProScan rejects '*' and other non-standard residues
    sed 's/\\*//g' ${species_fasta} > clean.fasta

    ${params.interproscan_dir}/interproscan.sh \\
        --input clean.fasta \\
        --applications Pfam \\
        --formats TSV \\
        --disable-precalc \\
        --cpu ${task.cpus} \\
        --outfile ${species}.pfam.tsv

    gzip -c ${species}.pfam.tsv > ${species}.pfam.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — ESM-2 windowed embedding search (GPU arm; scaffold)
// ---------------------------------------------------------------------------

process esm2Embed {
    /*
     * Per-residue ESM-2 embeddings, mean-pooled over sliding windows, so the
     * comparison stays at the domain unit rather than the whole protein.
     *
     * Deliberately NOT run on the laptop.  On a single A100 the human proteome is
     * roughly a few GPU-hours at the 650M checkpoint; the 9 target proteomes are
     * comparable.  Hand this off to a GPU box or an AWS Batch GPU queue.
     */
    tag "${label}"
    container 'quay.io/biocontainers/fair-esm:2.0.0--pyhdfd78af_0'
    label 'gpu'

    input:
    tuple val(label), path(fasta)

    output:
    tuple val(label), path("${label}.esm2_windows.npz")

    script:
    """
    ${params.python} ${projectDir}/bin/esm2_window_embed.py \\
        --fasta ${fasta} \\
        --model ${params.esm2_model} \\
        --window ${params.esm2_window} \\
        --stride ${params.esm2_stride} \\
        --out ${label}.esm2_windows.npz
    """
}

process esm2Search {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/fair-esm:2.0.0--pyhdfd78af_0'
    label 'gpu'
    publishDir "${params.outdir}/plm", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_npz), path(human_npz)

    output:
    tuple val(species), val("esm2"), path("human_vs_${species}.esm2.tsv.gz")

    script:
    """
    ${params.python} ${projectDir}/bin/esm2_window_search.py \\
        --query ${human_npz} \\
        --target ${species_npz} \\
        --top-k 1000 \\
        --out human_vs_${species}.esm2.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// WORKFLOW
// ---------------------------------------------------------------------------

workflow {

    def wanted_arms = params.arms.tokenize(',').collect { it.trim() }
    def wanted_species = params.species == 'all'
        ? SPECIES.collect { it.label }
        : params.species.tokenize(',').collect { it.trim() }

    def selected = SPECIES.findAll { wanted_species.contains(it.label) }
    if (!selected) {
        error "No species matched --species ${params.species}"
    }

    human_fasta = speciesFasta(HUMAN)
    species_ch  = Channel.fromList(selected.collect { tuple(it.label, speciesFasta(it)) })

    // ---------------------------------------------------------------- foldseek
    if (wanted_arms.contains('foldseek')) {
        // accessions come straight off the fasta headers; no new download
        proteomes = species_ch.mix(Channel.of(tuple('human', human_fasta)))
        struct_dirs = foldseekCollectStructures(proteomes).dir

        dbs = foldseekCreateDB(struct_dirs)

        human_db  = dbs.filter { label, db -> label == 'human' }.map { label, db -> db }
        target_db = dbs.filter { label, db -> label != 'human' }

        foldseekSearch(target_db.combine(human_db))
    }

    // --------------------------------------------------------------- jackhmmer
    if (wanted_arms.contains('jackhmmer')) {
        if (!params.jackhmmer_db) {
            error "--jackhmmer_db is required for the jackhmmer arm (e.g. UniRef50 fasta)"
        }
        human_hmm = jackhmmerBuildProfiles(human_fasta, file(params.jackhmmer_db))
        jackhmmerSearch(species_ch.combine(human_hmm))
    }

    // ------------------------------------------------------------ interproscan
    if (wanted_arms.contains('interproscan')) {
        interproscanPfam(species_ch)
    }

    // -------------------------------------------------------------------- esm2
    if (wanted_arms.contains('esm2')) {
        embeds = esm2Embed(
            species_ch.mix(Channel.of(tuple('human', human_fasta)))
        )
        human_npz  = embeds.filter { label, npz -> label == 'human' }.map { label, npz -> npz }
        target_npz = embeds.filter { label, npz -> label != 'human' }
        esm2Search(target_npz.combine(human_npz))
    }
}
