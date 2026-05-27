#!/usr/bin/env nextflow

/*
 * pfam_benchmark_tools.nf
 *
 * Run HMMER3, MMseqs2 (seq-seq + iterative), DIAMOND --ultra-sensitive,
 * HHblits, and PSALM against the QfO Pfam subdomain benchmark ground truth.
 *
 * Each tool searches human proteins against 9 QfO species spanning
 * 100–2000 MYA evolutionary distance, then each run is evaluated against
 * ground-truth pairs (proteins sharing ≥1 Pfam domain but different overall
 * architectures = TP; no shared Pfam domains = TN).
 *
 * Sensitivity ladder:
 *   DIAMOND → MMseqs2-seq → HMMER3-phmmer → MMseqs2-iterative → HHblits
 *   Plus annotation-based: HMMER3-hmmscan, PSALM
 *
 * Prerequisites:
 *   - DIAMOND, MMseqs2, HMMER3, HH-suite Docker images
 *   - Pfam-A.hmm (hmmpress'd) for hmmscan (optional; set params.pfam_hmm)
 *   - HHblits uniclust30/UniRef30 database (optional; set params.hhblits_db)
 *   - PSALM model weights (optional; set params.psalm_model)
 *
 * Usage (from this directory):
 *   nextflow run main.nf
 *   nextflow run main.nf --resume
 *   nextflow run main.nf --skip_hhblits --skip_psalm   # minimal run
 *
 * Download Pfam-A.hmm:
 *   wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
 *   gunzip Pfam-A.hmm.gz && hmmpress Pfam-A.hmm
 *
 * Download UniRef30 for HHblits (large, ~50 GB):
 *   https://wwwuser.gwdg.de/~compbiol/uniclust/current_release/
 */

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

def home = System.getProperty('user.home')

params.qfo_dir          = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.ground_truth_dir = "${projectDir}/../../results/pfam_benchmark/pairs"
params.outdir           = "${home}/data/pfam-benchmark-tools/results"

// Pfam HMM database (optional; required for hmmscan mode)
params.pfam_hmm         = "${home}/data/pfam/Pfam-A.hmm"
params.run_hmmscan      = file(params.pfam_hmm).exists()

// HHblits background database (optional; required for full sensitivity)
// Set to null to use single-sequence profiles (lower sensitivity, no DB needed)
params.hhblits_db       = null    // e.g. "${home}/data/uniclust30/UniRef30_2023_02"
params.skip_hhblits     = false   // set true to skip HHblits entirely

// PSALM model checkpoint (optional; set to enable PSALM)
params.psalm_model      = null
params.skip_psalm       = (params.psalm_model == null)

// Search parameters
params.mmseqs2_sensitivity  = 7       // MMseqs2 -s setting (1-7.5; 7 = most sensitive)
params.mmseqs2_iterations   = 3       // >1 enables iterative profile refinement
params.evalue_report        = 10.0    // loose E-value cutoff for reporting

// ---------------------------------------------------------------------------
// Species: [label, taxon_id, proteome_id, subdir, mya]
// Must match QfO_release_2020_04_with_updated_UP000008143 filenames
// ---------------------------------------------------------------------------

def SPECIES = [
    [label: "mouse",       taxon: "10090",  proteome: "UP000000589", subdir: "Eukaryota", mya: 100],
    [label: "chicken",     taxon: "9031",   proteome: "UP000000539", subdir: "Eukaryota", mya: 300],
    [label: "zebrafish",   taxon: "7955",   proteome: "UP000000437", subdir: "Eukaryota", mya: 430],
    [label: "ciona",       taxon: "7719",   proteome: "UP000008144", subdir: "Eukaryota", mya: 550],
    [label: "fly",         taxon: "7227",   proteome: "UP000000803", subdir: "Eukaryota", mya: 600],
    [label: "worm",        taxon: "6239",   proteome: "UP000001940", subdir: "Eukaryota", mya: 650],
    [label: "yeast",       taxon: "559292", proteome: "UP000002311", subdir: "Eukaryota", mya: 900],
    [label: "arabidopsis", taxon: "3702",   proteome: "UP000006548", subdir: "Eukaryota", mya: 1500],
    [label: "ecoli",       taxon: "83333",  proteome: "UP000000625", subdir: "Bacteria",  mya: 2000],
]

// ---------------------------------------------------------------------------
// PROCESSES — DIAMOND
// ---------------------------------------------------------------------------

process diamondMakeDB {
    tag "${species}"
    container 'quay.io/biocontainers/diamond:2.1.24--hf93d47f_0'
    label 'medium_cpu'

    input:
    tuple val(species), path(fasta)

    output:
    tuple val(species), path("${species}.dmnd")

    script:
    """
    diamond makedb --in ${fasta} --db ${species} --threads ${task.cpus}
    """
}

process diamondSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/diamond:2.1.24--hf93d47f_0'
    label 'medium_cpu'
    publishDir "${params.outdir}/diamond", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(diamond_db), path(human_fasta)

    output:
    tuple val(species), val("diamond"), path("human_vs_${species}.diamond.tsv.gz")

    script:
    // outfmt 6: query_id, target_id, bitscore, evalue
    """
    diamond blastp \\
        --query      ${human_fasta} \\
        --db         ${diamond_db}  \\
        --ultra-sensitive            \\
        --outfmt 6 qseqid sseqid bitscore evalue \\
        --evalue     ${params.evalue_report} \\
        --max-target-seqs 0          \\
        --threads    ${task.cpus}    \\
        --block-size 2               \\
        --out        /dev/stdout     \\
    | gzip -c > human_vs_${species}.diamond.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — MMseqs2
// ---------------------------------------------------------------------------

process mmseqs2EasySearch {
    // seq-seq search (no profile building)
    tag "human_vs_${species} [${mode}]"
    container 'quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0'
    label 'medium_cpu'
    publishDir "${params.outdir}/${mode}", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(mode), val(num_iter), path(species_fasta), path(human_fasta)

    output:
    tuple val(species), val(mode), path("human_vs_${species}.${mode}.tsv.gz")

    script:
    def iter_flag = num_iter > 1 ? "--num-iterations ${num_iter}" : ""
    """
    mkdir -p mmseqs_tmp
    mmseqs easy-search \\
        ${human_fasta} \\
        ${species_fasta} \\
        human_vs_${species}.${mode}.tsv \\
        mmseqs_tmp \\
        --threads  ${task.cpus}                   \\
        -s         ${params.mmseqs2_sensitivity}  \\
        ${iter_flag}                               \\
        --max-seqs 1000                            \\
        --format-output "query,target,bits,evalue"
    gzip -c human_vs_${species}.${mode}.tsv > human_vs_${species}.${mode}.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — HMMER3 phmmer (seq-seq with query profiling)
// ---------------------------------------------------------------------------

process hmmer3Phmmer {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hmmer:3.4--hb6cb901_4'
    label 'medium_cpu'
    publishDir "${params.outdir}/hmmer3_phmmer", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta), path(human_fasta)

    output:
    tuple val(species), val("hmmer3_phmmer"), path("human_vs_${species}.hmmer3_phmmer.tsv.gz")

    script:
    // phmmer --tblout: columns are target_name target_acc query_name query_acc evalue score bias ...
    // We extract: query_name(col3) target_name(col1) score(col6) evalue(col5)
    """
    mkfifo tbl_pipe
    phmmer \\
        --tblout  tbl_pipe                \\
        --noali                           \\
        -E        ${params.evalue_report} \\
        --cpu     ${task.cpus}            \\
        ${human_fasta} ${species_fasta}   \\
        > /dev/null &
    grep -v '^#' tbl_pipe \\
    | awk 'NF >= 6 {print \$3 "\\t" \$1 "\\t" \$6 "\\t" \$5}' \\
    | gzip -c > human_vs_${species}.hmmer3_phmmer.tsv.gz
    wait
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — HMMER3 hmmscan (seq → Pfam domain annotation)
// ---------------------------------------------------------------------------

process hmmer3Hmmscan {
    // Annotate proteins with Pfam domains (run on human AND each species)
    tag "${label}"
    container 'quay.io/biocontainers/hmmer:3.4--hb6cb901_4'
    label 'high_cpu'
    publishDir "${params.outdir}/hmmer3_hmmscan/annotations", mode: 'copy', pattern: '*.domtblout.gz'

    input:
    tuple val(label), path(fasta)
    path pfam_hmm

    output:
    tuple val(label), path("${label}.hmmscan.domtblout.gz")

    script:
    // domtblout columns: target_name target_acc tlen query_name query_acc qlen evalue score bias ...
    // domain-level: col 1 = target_domain, col 4 = query_protein
    """
    hmmscan \\
        --domtblout /dev/stdout \\
        --noali                 \\
        -E         1.0          \\
        --domE     1.0          \\
        --cpu      ${task.cpus} \\
        ${pfam_hmm}             \\
        ${fasta}                \\
    | grep -v '^#' \\
    | gzip -c > ${label}.hmmscan.domtblout.gz
    """
}

process hmmer3HmmscanPairScores {
    // Compute similarity score for each ground-truth pair based on shared Pfam domains.
    // Score = max bitscore of shared domains (0 if no shared domains).
    tag "human_vs_${species}"
    label 'python'
    publishDir "${params.outdir}/hmmer3_hmmscan", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species),
          path(human_annot),
          path(species_annot),
          path(ground_truth)

    output:
    tuple val(species), val("hmmer3_hmmscan"), path("human_vs_${species}.hmmer3_hmmscan.tsv.gz")

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import gzip, polars as pl

    def load_domtblout(gz_path):
        rows = []
        with gzip.open(gz_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 22:
                    continue
                # col 0: pfam_domain_name, col 3: query_protein_name, col 7: evalue, col 13: domain_bitscore
                domain   = parts[0]
                protein  = parts[3]
                score    = float(parts[13])   # domain bitscore
                rows.append({'protein': protein, 'domain': domain, 'score': score})
        return pl.DataFrame(rows) if rows else pl.DataFrame({'protein': [], 'domain': [], 'score': []})

    def extract_accession(s):
        if '|' in s:
            parts = s.split('|')
            if len(parts) >= 2:
                return parts[1]
        return s

    human_df   = load_domtblout('${human_annot}')
    species_df = load_domtblout('${species_annot}')

    human_df   = human_df.with_columns(pl.col('protein').map_elements(extract_accession, return_dtype=pl.String))
    species_df = species_df.with_columns(pl.col('protein').map_elements(extract_accession, return_dtype=pl.String))

    gt = pl.read_parquet('${ground_truth}').select(['human_accession', 'species_accession'])

    # Join human domains and species domains to find shared Pfam domains per pair
    h  = human_df.rename({'protein': 'human_accession', 'score': 'human_score'})
    s  = species_df.rename({'protein': 'species_accession', 'score': 'species_score'})

    shared = h.join(s, on='domain', how='inner')

    # Score = max(min(human_score, species_score)) per pair — conservative domain evidence
    pair_scores = (
        shared
        .with_columns(pl.min_horizontal('human_score', 'species_score').alias('pair_domain_score'))
        .group_by(['human_accession', 'species_accession'])
        .agg(pl.col('pair_domain_score').max().alias('score'))
    )

    # Only output pairs present in ground truth
    result = gt.join(pair_scores, on=['human_accession', 'species_accession'], how='left').fill_null(0.0)
    result = result.with_columns([
        pl.lit('sp|').add(pl.col('human_accession')).add(pl.lit('|x')).alias('query_id'),
        pl.lit('sp|').add(pl.col('species_accession')).add(pl.lit('|x')).alias('target_id'),
    ]).select(['query_id', 'target_id', 'score'])

    import io
    buf = io.BytesIO()
    result.write_csv(buf, separator='\\t', has_header=False)
    import gzip as gz2
    with gz2.open('human_vs_${species}.hmmer3_hmmscan.tsv.gz', 'wb') as out:
        out.write(buf.getvalue())

    print(f"hmmscan pairs: {len(result)}, with score>0: {(result['score'] > 0).sum()}")
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — HHblits / HHsearch (profile–profile)
// ---------------------------------------------------------------------------

process hhblitsBuildDB {
    /*
     * Build an HH-suite target database from species proteins.
     *
     * If params.hhblits_db is provided (uniclust30 / UniRef30), hhblits is used
     * to build MSA profiles for each sequence — this is the recommended mode
     * for maximum sensitivity.
     *
     * If params.hhblits_db is null, single-sequence a3m files are used.
     * Sensitivity will be similar to phmmer in that case.
     *
     * Requires: hh-suite Docker image with ffindex_from_fasta, hhmake, cstranslate.
     */
    tag "${species}"
    container 'quay.io/biocontainers/hhsuite:3.3.0--h503566f_15'
    label 'high_cpu'

    input:
    tuple val(species), path(fasta)

    output:
    tuple val(species), path("${species}_hhdb")

    script:
    def uniclust_flag = params.hhblits_db ? "-d ${params.hhblits_db}" : ""
    def n_iter = params.hhblits_db ? "2" : "0"
    """
    mkdir -p ${species}_hhdb

    # Convert FASTA to ffindex for batch processing
    ffindex_from_fasta -s ${species}_hhdb/seq.ffdata \\
                          ${species}_hhdb/seq.ffindex \\
                       ${fasta}

    if [ -n "${params.hhblits_db ?: ''}" ]; then
        # Build MSA profiles via hhblits against background DB (better sensitivity)
        ffindex_apply \\
            ${species}_hhdb/seq.ffdata \\
            ${species}_hhdb/seq.ffindex \\
            -d ${species}_hhdb/a3m.ffdata \\
            -i ${species}_hhdb/a3m.ffindex \\
            -- hhblits -i stdin -d ${params.hhblits_db} \\
               -oa3m stdout -n ${n_iter} -cpu 1 -v 0
    else
        # Use single-sequence a3m (all uppercase = all consensus, no MSA)
        ffindex_apply \\
            ${species}_hhdb/seq.ffdata \\
            ${species}_hhdb/seq.ffindex \\
            -d ${species}_hhdb/a3m.ffdata \\
            -i ${species}_hhdb/a3m.ffindex \\
            -- awk '/^>/{print} !/^>/{print toupper(\$0)}'
    fi

    # Build HMM profiles from a3m files
    ffindex_apply \\
        ${species}_hhdb/a3m.ffdata \\
        ${species}_hhdb/a3m.ffindex \\
        -d ${species}_hhdb/hhm.ffdata \\
        -i ${species}_hhdb/hhm.ffindex \\
        -- hhmake -i stdin -o stdout -v 0

    # Build cs219 context-state database for fast prefiltering
    cstranslate \\
        -f -x 0.3 -c 4 -I a3m \\
        -i ${species}_hhdb/a3m \\
        -o ${species}_hhdb/cs219

    sort -k3 -n ${species}_hhdb/cs219.ffindex | cut -f1 > ${species}_hhdb/db.sort
    """
}

process hhblitsBuildQueryProfiles {
    /*
     * Build a3m profiles for human query proteins.
     * Analogous to hhblitsBuildDB but for the human proteome.
     */
    container 'quay.io/biocontainers/hhsuite:3.3.0--h503566f_15'
    label 'high_cpu'

    input:
    path human_fasta

    output:
    path "human_hhdb"

    script:
    def n_iter = params.hhblits_db ? "2" : "0"
    """
    mkdir -p human_hhdb

    ffindex_from_fasta -s human_hhdb/seq.ffdata \\
                          human_hhdb/seq.ffindex \\
                       ${human_fasta}

    if [ -n "${params.hhblits_db ?: ''}" ]; then
        ffindex_apply \\
            human_hhdb/seq.ffdata \\
            human_hhdb/seq.ffindex \\
            -d human_hhdb/a3m.ffdata \\
            -i human_hhdb/a3m.ffindex \\
            -- hhblits -i stdin -d ${params.hhblits_db} \\
               -oa3m stdout -n ${n_iter} -cpu 1 -v 0
    else
        ffindex_apply \\
            human_hhdb/seq.ffdata \\
            human_hhdb/seq.ffindex \\
            -d human_hhdb/a3m.ffdata \\
            -i human_hhdb/a3m.ffindex \\
            -- awk '/^>/{print} !/^>/{print toupper(\$0)}'
    fi
    """
}

process hhblitsSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hhsuite:3.3.0--h503566f_15'
    label 'high_cpu'
    publishDir "${params.outdir}/hhblits", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_hhdb), path(human_hhdb)

    output:
    tuple val(species), val("hhblits"), path("human_vs_${species}.hhblits.tsv.gz")

    script:
    // hhsearch -blasttab: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    // We use cols: qseqid(0) sseqid(1) bitscore(11) evalue(10)
    """
    ffindex_apply \\
        ${human_hhdb}/a3m.ffdata \\
        ${human_hhdb}/a3m.ffindex \\
        -d results.ffdata \\
        -i results.ffindex \\
        -- hhsearch \\
               -i stdin \\
               -d ${species_hhdb}/hhm \\
               -blasttab /dev/stdout \\
               -cpu 1 -v 0

    # Extract blasttab lines from ffindex and reformat to (query, target, bitscore, evalue)
    ffindex_apply results.ffdata results.ffindex -- cat \\
    | awk 'NF >= 12 {print \$1 "\\t" \$2 "\\t" \$12 "\\t" \$11}' \\
    | gzip -c > human_vs_${species}.hhblits.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — PSALM (LM-based domain annotation)
// ---------------------------------------------------------------------------

process psalmAnnotate {
    /*
     * Annotate proteins with Pfam domains using PSALM
     * (Protein Sequence domain Annotation using Language Models).
     *
     * Paper: https://www.mlsb.io/papers_2024/Protein_Sequence_Domain_Annotation_using_Language_Models.pdf
     * GitHub: check paper for repository link
     *
     * Requires:
     *   params.psalm_model: path to PSALM model checkpoint (ESM2-based)
     *   PSALM Docker image with ESM2, PyTorch, PSALM installed
     *
     * Output: TSV with columns: protein_id, pfam_domain, score
     */
    tag "${label}"
    container 'psalm:latest'    // build from PSALM GitHub repo
    label 'gpu_or_high_cpu'

    when:
    !params.skip_psalm

    input:
    tuple val(label), path(fasta)
    path psalm_model

    output:
    tuple val(label), path("${label}.psalm.tsv.gz")

    script:
    """
    psalm predict \\
        --fasta       ${fasta} \\
        --model       ${psalm_model} \\
        --output      ${label}.psalm.tsv \\
        --format      tsv \\
        --device      cpu

    gzip -c ${label}.psalm.tsv > ${label}.psalm.tsv.gz
    """
}

process psalmPairScores {
    // Same as hmmer3HmmscanPairScores but using PSALM annotations
    tag "human_vs_${species}"
    label 'python'

    when:
    !params.skip_psalm

    input:
    tuple val(species),
          path(human_annot),
          path(species_annot),
          path(ground_truth)

    output:
    tuple val(species), val("psalm"), path("human_vs_${species}.psalm.tsv.gz")

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import gzip, io, polars as pl

    def load_psalm(gz_path):
        with gzip.open(gz_path, 'rt') as f:
            # Expected format: protein_id <tab> pfam_domain <tab> score
            return pl.read_csv(f, separator='\\t', has_header=False,
                               new_columns=['protein', 'domain', 'score'],
                               infer_schema_length=10000)

    def extract_accession(s):
        if '|' in s:
            parts = s.split('|')
            if len(parts) >= 2:
                return parts[1]
        return s

    human_df   = load_psalm('${human_annot}')
    species_df = load_psalm('${species_annot}')

    human_df   = human_df.with_columns(pl.col('protein').map_elements(extract_accession, return_dtype=pl.String))
    species_df = species_df.with_columns(pl.col('protein').map_elements(extract_accession, return_dtype=pl.String))

    gt = pl.read_parquet('${ground_truth}').select(['human_accession', 'species_accession'])

    h = human_df.rename({'protein': 'human_accession', 'score': 'human_score'})
    s = species_df.rename({'protein': 'species_accession', 'score': 'species_score'})

    shared = h.join(s, on='domain', how='inner')
    pair_scores = (
        shared
        .with_columns(pl.min_horizontal('human_score', 'species_score').alias('pair_domain_score'))
        .group_by(['human_accession', 'species_accession'])
        .agg(pl.col('pair_domain_score').max().alias('score'))
    )

    result = gt.join(pair_scores, on=['human_accession', 'species_accession'], how='left').fill_null(0.0)
    result = result.with_columns([
        pl.lit('sp|').add(pl.col('human_accession')).add(pl.lit('|x')).alias('query_id'),
        pl.lit('sp|').add(pl.col('species_accession')).add(pl.lit('|x')).alias('target_id'),
    ]).select(['query_id', 'target_id', 'score'])

    buf = io.BytesIO()
    result.write_csv(buf, separator='\\t', has_header=False)
    import gzip as gz2
    with gz2.open('human_vs_${species}.psalm.tsv.gz', 'wb') as out:
        out.write(buf.getvalue())
    """
}

// ---------------------------------------------------------------------------
// PROCESSES — Evaluation
// ---------------------------------------------------------------------------

process evaluateBenchmark {
    tag "${tool} vs ${species}"
    label 'python'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.parquet'

    input:
    tuple val(species), val(tool), path(results_tsv_gz), path(ground_truth)

    output:
    path "${tool}.${species}.metrics.parquet"
    path "${tool}.${species}.pr_curve.parquet"

    script:
    """
    ${projectDir}/bin/evaluate_benchmark.py \\
        ${results_tsv_gz} \\
        ${species} \\
        ${ground_truth} \\
        ${tool} \\
        ${tool}.${species}.metrics.parquet \\
        ${tool}.${species}.pr_curve.parquet
    """
}

process aggregateMetrics {
    label 'python'
    publishDir params.outdir, mode: 'copy'

    input:
    path 'metrics/*'

    output:
    path "all_tools_metrics.parquet"
    path "all_tools_pr_curves.parquet"

    script:
    """
    ${projectDir}/bin/aggregate_metrics.py metrics all_tools_metrics.parquet all_tools_pr_curves.parquet
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    // Channel of (species_label, species_fasta) tuples
    species_ch = Channel.fromList(
        SPECIES.collect { s ->
            tuple(s.label, file("${params.qfo_dir}/${s.subdir}/${s.proteome}_${s.taxon}.fasta"))
        }
    )

    // Ground truth channel: (species_label, ground_truth_parquet)
    gt_ch = Channel.fromList(
        SPECIES.collect { s ->
            tuple(s.label, file("${params.ground_truth_dir}/human_vs_${s.label}_ground_truth.parquet"))
        }
    )

    // -----------------------------------------------------------------------
    // DIAMOND
    // -----------------------------------------------------------------------
    diamond_db_ch = diamondMakeDB(species_ch)

    diamond_search_input = diamond_db_ch.map { species, db ->
        tuple(species, db, human_fasta)
    }
    diamond_results = diamondSearch(diamond_search_input)

    // -----------------------------------------------------------------------
    // MMseqs2 — seq-seq and iterative
    // -----------------------------------------------------------------------
    mmseqs2_input = species_ch.flatMap { species, fasta ->
        [
            tuple(species, "mmseqs2_seqseq",    1,                             fasta, human_fasta),
            tuple(species, "mmseqs2_iterative",  params.mmseqs2_iterations,    fasta, human_fasta),
        ]
    }
    mmseqs2_results = mmseqs2EasySearch(mmseqs2_input)

    // -----------------------------------------------------------------------
    // HMMER3 phmmer
    // -----------------------------------------------------------------------
    phmmer_input = species_ch.map { species, fasta ->
        tuple(species, fasta, human_fasta)
    }
    phmmer_results = hmmer3Phmmer(phmmer_input)

    // -----------------------------------------------------------------------
    // HMMER3 hmmscan (optional; requires Pfam-A.hmm)
    // -----------------------------------------------------------------------
    hmmscan_results = Channel.empty()
    if (params.run_hmmscan) {
        def pfam_hmm = file(params.pfam_hmm)

        // Annotate human + all 9 species in one channel call
        all_for_hmmscan = species_ch.mix(Channel.of(tuple("human", human_fasta)))
        all_annot_ch = hmmer3Hmmscan(all_for_hmmscan, pfam_hmm)

        // Split into human vs species annotations
        human_annot_ch   = all_annot_ch.filter { label, _annot -> label == "human" }
        species_annot_ch = all_annot_ch.filter { label, _annot -> label != "human" }

        // Combine: for each species, emit (species, human_annot, species_annot, ground_truth)
        hmmscan_pair_input = species_annot_ch
            .combine(human_annot_ch.map { _label, annot -> annot })
            .map { species_label, sp_annot, hu_annot -> tuple(species_label, hu_annot, sp_annot) }
            .join(gt_ch, by: 0)

        hmmscan_results = hmmer3HmmscanPairScores(hmmscan_pair_input)
    }

    // -----------------------------------------------------------------------
    // HHblits (optional; skip if params.skip_hhblits)
    // -----------------------------------------------------------------------
    hhblits_results = Channel.empty()
    if (!params.skip_hhblits) {
        human_hhdb   = hhblitsBuildQueryProfiles(human_fasta)
        species_hhdb = hhblitsBuildDB(species_ch)

        hhblits_input = species_hhdb.combine(human_hhdb)
        hhblits_results = hhblitsSearch(hhblits_input)
    }

    // -----------------------------------------------------------------------
    // PSALM (optional; skip if params.psalm_model is null)
    // -----------------------------------------------------------------------
    psalm_results = Channel.empty()
    if (!params.skip_psalm) {
        def psalm_model = file(params.psalm_model)

        all_proteins_ch = species_ch.mix(Channel.of(tuple("human", human_fasta)))
        psalm_annot_ch  = psalmAnnotate(all_proteins_ch, psalm_model)

        human_psalm_annot_ch   = psalm_annot_ch.filter { label, _annot -> label == "human" }
        species_psalm_annot_ch = psalm_annot_ch.filter { label, _annot -> label != "human" }

        psalm_pair_input = species_psalm_annot_ch
            .combine(human_psalm_annot_ch.map { _label, annot -> annot })
            .map { species_label, sp_annot, hu_annot -> tuple(species_label, hu_annot, sp_annot) }
            .join(gt_ch, by: 0)

        psalm_results = psalmPairScores(psalm_pair_input)
    }

    // -----------------------------------------------------------------------
    // Collect all tool results and evaluate against ground truth
    // -----------------------------------------------------------------------
    all_results = diamond_results
        .mix(mmseqs2_results)
        .mix(phmmer_results)
        .mix(hmmscan_results)
        .mix(hhblits_results)
        .mix(psalm_results)

    // Join each (species, tool, results_tsv) with its ground truth
    eval_input = all_results.join(gt_ch, by: 0)
        .map { species, tool, tsv, gt -> tuple(species, tool, tsv, gt) }

    eval_out = evaluateBenchmark(eval_input)

    // Collect all .parquet outputs (metrics + pr_curve) and aggregate
    aggregateMetrics(eval_out[0].mix(eval_out[1]).collect())

    // Print a summary of completed evaluations
    all_results.subscribe { species, tool, _tsv ->
        log.info "Completed: ${tool} human_vs_${species}"
    }
}
