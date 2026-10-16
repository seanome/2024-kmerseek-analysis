#!/usr/bin/env nextflow
/*
 * Pilot: does an index of Swiss-Prot's annotated regions alone let kmerseek call short and
 * disordered regions that it cannot call against whole proteins, and does it help kmerseek
 * more than MMseqs2?
 *
 * Queries: reviewed human Swiss-Prot entries on chromosome 6 (assets/chr6_queries.*.tsv).
 * Truth:   each query's own Swiss-Prot features (bin/build_query_truth.py).
 * Indexes, both from reviewed Swiss-Prot with every Mammalia entry removed:
 *   the whole-protein index  every remaining entry, whole
 *   the regions index        every DOMAIN, REGION, MOTIF, REPEAT, ZN_FING, COMPBIAS, COILED,
 *                            TRANSMEM and INTRAMEM of those entries, cut out with
 *                            (k_max - 1) residues each side; 30 aa or more clustered with
 *                            MMseqs2 (50% identity, 80% coverage of both), shorter ones
 *                            deduplicated by exact sequence
 * Each index is searched together with its decoy copy (residues shuffled within 10-residue
 * windows, names prefixed DECOY_), so calls can be compared at the same decoy error rate.
 * A second decoy copy, shuffled within 20-residue windows, repeats everything as a check.
 *
 * Entries:
 *   -entry build   the queries, truth, both indexes and their decoys
 *   (default)      the build, then kmerseek (four settings), MMseqs2 and the
 *                  composition-only classifier against both indexes, the scoring at 5%
 *                  decoy error, and the comparison tables (bin/compare_indexes.py)
 * --query_limit 20 runs the searches on 20 queries spread over the accession list.
 */

nextflow.enable.dsl = 2


params.swissprot_dat = "${System.getenv('HOME')}/data/uniprot/uniprot_sprot.2026_03.dat.gz"
params.chr6_queries  = "${projectDir}/assets/chr6_queries.2026_03.tsv"
params.disprot       = "${projectDir}/assets/disprot_human_disorder.tsv"
params.outdir        = null
params.exclude_clade = 'Mammalia'
// The largest k-size among the kmerseek settings searched. A regions-index entry carries
// k_max - 1 residues each side of its feature, so a k-mer of that size can start on the
// feature's first residue and end on its last. 22 is the largest k among the settings
// (hp_thomas_dill2, hp_thomas_dill_no_c2 and hp_kyte_doolittle2 at 22; PREDICTIONS.md).
params.k_max          = 22
params.min_cluster_aa = 30
params.cluster_min_seq_id = 0.5
params.cluster_coverage   = 0.8
// The decoys' shuffle window. The first is the headline; the others repeat everything as a
// check (Olga, 2026-10-07: keep 10, add 20).
params.decoy_windows = '10,20'
params.decoy_seed    = 20261006

// kmerseek settings, alphabet:k:scaled (Olga, 2026-10-13; PREDICTIONS.md): every alphabet
// with a measured kappa, at the k that carries about the 20.9 bits of a protein20 5-mer
// (bits per letter from real amino-acid frequencies), inside the alphabet's tested k range
// (notebook 274); polarity4 at 11 to match the dark set; gbmr7 at 16, because at its bits-
// matched k (its range floor, 13) the dark set's searches needed 379-500 GB without decoys.
// Each searches with extension at its alphabet's own mismatch penalty, c_opt in
// assets/kappa_by_alphabet.tsv (copied from olgabot/dark-set-kmerseek-0.4), X-drop 4 x C,
// low-complexity mask off.
params.kmerseek_settings = [
    'hp_lehninger2:21:1', 'hp_lehninger_c_nonpolar2:21:1', 'hp_pbotc_1st_ed2:21:1',
    'hp_thomas_dill2:22:1', 'hp_thomas_dill_no_c2:22:1', 'hp_kyte_doolittle2:22:1',
    'hp_lehninger_hpc3:19:1', 'gbmr4:14:1', 'polarity4:11:1', 'wwmj5:10:1', 'dayhoff6:9:1',
    'gbmr7:16:1', 'funcgroups8:8:1', 'mmseqs12:6:1', 'sdm12:7:1', 'wass14:6:1',
    'hsdm17:6:1', 'uniprot18:5:1', 'protein20:5:1',
].join(',')
// Search the regions index a second time with the whole-protein index's fit (same setting
// and decoy window), so the two indexes differ only in size n ("regionsshared"; Olga,
// 2026-10-13). Needs kmerseek with --ka-r-database (0.4.1-rc1).
params.shared_fit = true
// Memory. Index builds: assets/kmerseek_index_first_ask.tsv, else the default. Searches:
// the dark set's model from the index's own k-mer spectrum, headroom x (36 + 2.34 x
// sqrt(load)) GB, where load = sum(occ^2 x n) / sum(occ x n) over spectrum.csv.gz
// (olgabot/dark-set-kmerseek-0.4 main.nf, kmerseekSearchMemory), at least the floor.
// Retries double, up to the largest node.
params.kmerseek_index_first_ask    = "${projectDir}/assets/kmerseek_index_first_ask.tsv"
params.kmerseek_index_memory_default_gb = 120
params.kmerseek_search_memory_base     = 36
params.kmerseek_search_memory_slope    = 2.34
params.kmerseek_search_memory_headroom = 1.5
params.kmerseek_search_memory_floor_gb = 24
// The largest node: a setting whose search model asks more is skipped and named in the
// log. Separate from the cap on any one ask, which the laptop profile lowers.
params.kmerseek_skip_above_gb          = 500
params.kmerseek_memory_max_gb          = 500
params.kappa_table       = "${projectDir}/assets/kappa_by_alphabet.tsv"
params.xdrop_per_penalty = 4
params.chain_max_gap     = 30
params.chain_max_shift   = 10
// Index entries the Karlin-Altschul fit searches with, per index. 500 whole proteins are about
// 175_000 residues; regions-index entries are about a fifth as long (mean 71 aa: 45_736_962 residues over
// 639_854 entries), so the
// regions index gets 2_500 for about the same residues. With 500, protein20 k5 on the
// regions index (10-residue decoys) was refused for too few score bins on 2026-10-07, while
// the same index with 20-residue decoys fitted: a fit at the edge of what 500 entries give.
params.ka_queries         = 500
params.ka_queries_regions = 2500
// The dark set's search filters (olgabot/dark-set-kmerseek-0.4 main.nf, 7913404).
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3
// Rows above this region_evalue are dropped in the search's own pipe, as in the dark set.
// The 5% decoy thresholds are expected far below it; compare_indexes.py checks.
params.max_region_evalue = 10000
// The dark set's mmseqs2Search: -s 7, 3 iterations, report E <= 10.
params.mmseqs_sensitivity = 7
params.mmseqs_iterations  = 3
params.mmseqs_evalue      = 10
params.composition_window = 30
params.composition_step   = 10
params.max_decoy_rate = 0.05
params.min_landing    = 0.5
params.query_chunk_size = 100
// 0 searches every query. A positive number searches that many, spread evenly over the
// accession-sorted list, for a test run.
params.query_limit = 0
// The Pfam index's inputs; null leaves it out.
params.pfam_regions = null
params.pfam_hmm_dat = null
// A command put in front of each kmerseek and MMseqs2 call to record its peak memory. Empty on
// Sherlock, where the trace has it; `/usr/bin/time -l` on the laptop (profile local), where
// Nextflow reports no memory on macOS. Its report goes to the task's log.
params.time_cmd = ''

// The interpreter for every bin/ script, and the MMseqs2 binary. The container's on
// Sherlock; full paths on the laptop (nextflow.config, profile local), because a
// `#!/usr/bin/env python3` script inside Nextflow finds the system Python, not the conda env.
def py(String script) { "${params.python} \$(command -v ${script})" }

process parseFeatures {
    label 'python'
    publishDir "${params.outdir}/swissprot", mode: 'copy'

    input:
    path dat

    output:
    path 'sprot_features.parquet', emit: features
    path 'sprot_sequences.parquet', emit: sequences
    path 'parse_features.log'

    script:
    """
    ${py('parse_swissprot_features.py')} --swissprot-dat ${dat} --out-prefix sprot 2> parse_features.log
    cat parse_features.log >&2
    """
}

// Reused from invertebrate-dark-set (olgabot/dark-set-kmerseek-0.4 at 7913404), changed only
// by black formatting.
process buildWholeReference {
    label 'python'
    publishDir "${params.outdir}/whole_protein_index", mode: 'copy'

    input:
    path dat

    output:
    path 'reference.fasta', emit: fasta
    path 'reference_accessions.txt', emit: accessions
    path 'reference_summary.json'

    script:
    """
    ${py('build_clade_excluded_reference.py')} --swissprot-dat ${dat} \\
        --exclude-clade ${params.exclude_clade} --out-prefix reference \\
        --summary-out reference_summary.json
    """
}

process buildQueryTruth {
    label 'python'
    publishDir "${params.outdir}/queries", mode: 'copy'

    input:
    path features
    path sequences
    path chr6
    path disprot

    output:
    path 'queries.fasta', emit: queries
    path 'truth.parquet', emit: truth
    path 'truth_summary.tsv'
    path 'truth.log'

    script:
    """
    ${py('build_query_truth.py')} --features ${features} --sequences ${sequences} \\
        --chr6 ${chr6} --disprot ${disprot} --out-dir . 2> truth.log
    cat truth.log >&2
    """
}

process extractRegions {
    label 'python'
    publishDir "${params.outdir}/regions_index", mode: 'copy', pattern: 'regions.parquet'

    input:
    path features
    path sequences
    path accessions

    output:
    path 'regions.parquet', emit: regions
    path 'regions_long.fasta', emit: long_fasta
    path 'regions_short.fasta', emit: short_fasta

    script:
    """
    ${py('extract_regions.py')} --features ${features} --sequences ${sequences} \\
        --accessions ${accessions} --k-max ${params.k_max} \\
        --min-cluster-aa ${params.min_cluster_aa} --out-dir .
    """
}

process clusterRegions {
    label 'mmseqs'
    publishDir "${params.outdir}/regions_index", mode: 'copy', pattern: '*.log'

    input:
    path long_fasta

    output:
    path 'clu_cluster.tsv', emit: tsv
    path 'cluster.log'

    script:
    """
    set -euo pipefail
    ${params.mmseqs} easy-cluster ${long_fasta} clu tmp \\
        --min-seq-id ${params.cluster_min_seq_id} -c ${params.cluster_coverage} --cov-mode 0 \\
        --threads ${task.cpus} > cluster.log 2>&1
    ${params.mmseqs} version >> cluster.log
    """
}

process finishRegionsIndex {
    label 'python'
    publishDir "${params.outdir}/regions_index", mode: 'copy'

    input:
    path regions
    path cluster_tsv

    output:
    path 'regions_index.fasta', emit: fasta
    path 'cluster_members.parquet'
    path 'cluster_summary.tsv'

    script:
    """
    ${py('finish_regions_index.py')} --regions ${regions} --cluster-tsv ${cluster_tsv} --out-dir .
    """
}

// The Pfam index (Olga, 2026-10-15): Pfam's domains on the whole-protein index's entries,
// cut out, clustered and given decoys the way the regions index is. Built only when
// --pfam_regions and --pfam_hmm_dat are given (one Pfam release: Pfam-A.regions.tsv.gz and
// Pfam-A.hmm.dat.gz). Separate processes, so the regions index's tasks are untouched.
process extractPfamRegions {
    label 'python'
    publishDir "${params.outdir}/pfam_index", mode: 'copy', pattern: '*.{parquet,json}'

    input:
    path pfam_regions
    path pfam_hmm_dat
    path sequences
    path accessions

    output:
    path 'regions.parquet', emit: regions
    path 'regions_long.fasta', emit: long_fasta
    path 'regions_short.fasta', emit: short_fasta
    path 'pfam_filter.json'

    script:
    """
    ${py('extract_pfam_regions.py')} --pfam-regions ${pfam_regions} --pfam-hmm-dat ${pfam_hmm_dat} \\
        --sequences ${sequences} --accessions ${accessions} --k-max ${params.k_max} \\
        --min-cluster-aa ${params.min_cluster_aa} --out-dir .
    """
}

process clusterPfamRegions {
    label 'mmseqs'
    publishDir "${params.outdir}/pfam_index", mode: 'copy', pattern: '*.log'

    input:
    path long_fasta

    output:
    path 'clu_cluster.tsv', emit: tsv
    path 'cluster.log'

    script:
    """
    set -euo pipefail
    ${params.mmseqs} easy-cluster ${long_fasta} clu tmp \\
        --min-seq-id ${params.cluster_min_seq_id} -c ${params.cluster_coverage} --cov-mode 0 \\
        --threads ${task.cpus} > cluster.log 2>&1
    ${params.mmseqs} version >> cluster.log
    """
}

process finishPfamIndex {
    label 'python'
    publishDir "${params.outdir}/pfam_index", mode: 'copy'

    input:
    path regions
    path cluster_tsv

    output:
    path 'regions_index.fasta', emit: fasta
    path 'cluster_members.parquet'
    path 'cluster_summary.tsv'

    script:
    """
    ${py('finish_regions_index.py')} --regions ${regions} --cluster-tsv ${cluster_tsv} --out-dir .
    """
}

process makeDecoys {
    label 'python'
    tag "${name}.w${window}"
    publishDir "${params.outdir}/${name}", mode: 'copy', pattern: '*.{md5,log}'

    input:
    tuple val(name), path(fasta), val(window)

    output:
    tuple val(name), val(window), path("${name}.w${window}.target_decoy.fasta"), emit: combined
    path "${name}.w${window}.decoy.fasta.md5"
    path "${name}.w${window}.decoys.log"

    script:
    """
    ${py('make_window_decoys.py')} --fasta ${fasta} --decoy-out ${name}.w${window}.decoy.fasta \\
        --combined-out ${name}.w${window}.target_decoy.fasta \\
        --window ${window} --seed ${params.decoy_seed} 2> ${name}.w${window}.decoys.log
    cat ${name}.w${window}.decoys.log >&2
    """
}

process indexSummary {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path queries
    tuple val(name_a), path(fasta_a), val(name_b), path(fasta_b)

    output:
    path 'index_summary.json'
    path 'index_summary.log'

    script:
    """
    ${py('index_summary.py')} --queries ${queries} --index ${name_a} ${fasta_a} \\
        --index ${name_b} ${fasta_b} --out index_summary.json 2> index_summary.log
    cat index_summary.log >&2
    """
}

// ---------------------------------------------------------------------------------------
// Searches, scoring and comparison
// ---------------------------------------------------------------------------------------

// alphabet -> c_opt, from the kappa table's header-named columns.
def kappaPenalties() {
    def rows = file(params.kappa_table).readLines().findAll { it && !it.startsWith('#') }
    def head = rows[0].split('\t') as List
    def ia = head.indexOf('alphabet')
    def ic = head.indexOf('c_opt')
    if (ia < 0 || ic < 0) error "${params.kappa_table} has no alphabet or c_opt column"
    rows.drop(1).collectEntries { r -> def f = r.split('\t'); [(f[ia]): f[ic]] }
}

// The dark set's penaltyString: two decimals, trailing zeros dropped.
def penaltyString(double c) {
    def sf = String.format('%.2f', c)
    sf.contains('.') ? sf.replaceAll(/0+$/, '').replaceAll(/\.$/, '') : sf
}

def kmerseekSettings() {
    def pens = kappaPenalties()
    params.kmerseek_settings.toString().tokenize(',')*.trim().collect { spec ->
        def (alphabet, k, scaled) = spec.tokenize(':')
        def c = pens[alphabet]
        if (c == null) error "no c_opt for ${alphabet} in ${params.kappa_table}"
        def x = penaltyString((params.xdrop_per_penalty as double) * (c as double))
        [alphabet, k as int, scaled as int, c, x, "${alphabet}_k${k}_s${scaled}".toString()]
    }
}

// whole_protein_index -> whole, regions_index -> regions: the index part of a label.
def indexKind(String name) {
    name.startsWith('whole') ? 'whole' : (name.startsWith('pfam') ? 'pfam' : 'regions')
}

def indexFirstAskGb(String alphabet, int ksize) {
    def rows = file(params.kmerseek_index_first_ask).readLines().findAll { it && !it.startsWith('#') }
    def head = rows[0].split('\t') as List
    def hit = rows.drop(1).collect { it.split('\t') }.find {
        it[head.indexOf('alphabet')] == alphabet && it[head.indexOf('ksize')] as int == ksize
    }
    hit ? (hit[head.indexOf('first_ask_gb')] as double) : (params.kmerseek_index_memory_default_gb as double)
}

// load = sum(occ^2 x n_kmers) / sum(occ x n_kmers): the expected posting-list length hit by
// a random k-mer of the index. Read in the workflow, where the index path is real; a
// process directive sees only the staged name.
def spectrumLoad(Path idx) {
    def f = idx.resolve('spectrum.csv.gz')
    if (!f.exists()) error "no spectrum.csv.gz in ${idx}; kmerseekIndex writes one"
    double num = 0.0d
    double den = 0.0d
    new java.util.zip.GZIPInputStream(f.newInputStream()).withReader('UTF-8') { r ->
        r.eachLine { line ->
            if (line.startsWith('#') || line.startsWith('moltype')) return
            def c = line.split(',')
            double occ = c[2] as double
            double n = c[3] as double
            num += occ * occ * n
            den += occ * n
        }
    }
    den > 0 ? num / den : 0.0d
}

def searchFirstAskGb(double load) {
    double gb = (params.kmerseek_search_memory_headroom as double) *
        ((params.kmerseek_search_memory_base as double) +
         (params.kmerseek_search_memory_slope as double) * Math.sqrt(load))
    Math.max(gb, params.kmerseek_search_memory_floor_gb as double)
}

def memoryAsk(double firstGb, int attempt) {
    long gb = Math.min(Math.ceil(firstGb * Math.pow(2, attempt - 1)) as long,
                       params.kmerseek_memory_max_gb as long)
    "${gb} GB"
}

// The fit an index stored for `penalty`, read from the survival CSV the build wrote: K and
// the slope (the lambda scale; r_database in later kmerseek). Null when the fit was refused.
def storedFit(Path idx, String penalty) {
    def f = idx.resolve("ka_survival.C${penalty}.csv")
    if (!f.exists()) return null
    def lines = f.readLines()
    def head = lines[0].split(',') as List
    def row = lines[1].split(',') as List
    if (head.contains('fitted') && row[head.indexOf('fitted')] != 'true') return null
    [k: row[head.indexOf('k')], slope: row[head.indexOf('slope')]]
}

process splitQueries {
    label 'python'

    input:
    path queries

    output:
    path 'chunk_*.fasta', emit: chunks
    path 'searched_queries.fasta', emit: searched

    script:
    """
    ${params.python} - <<'PYEOF'
    recs, name, seq = [], None, []
    for line in open("${queries}"):
        line = line.rstrip("\\n")
        if line.startswith(">"):
            if name: recs.append((name, "".join(seq)))
            name, seq = line[1:].split()[0], []
        elif line:
            seq.append(line)
    if name: recs.append((name, "".join(seq)))
    recs.sort()
    limit = ${params.query_limit}
    if limit and limit < len(recs):
        step = len(recs) / limit
        recs = [recs[int(i * step)] for i in range(limit)]
    size = ${params.query_chunk_size}
    with open("searched_queries.fasta", "w") as all_fh:
        for i in range(0, len(recs), size):
            with open(f"chunk_{i // size:03d}.fasta", "w") as fh:
                for n, s in recs[i:i + size]:
                    fh.write(f">{n}\\n{s}\\n")
                    all_fh.write(f">{n}\\n{s}\\n")
    PYEOF
    """
}

process kmerseekIndex {
    label 'kmerseek_index'
    tag "${name}.w${window}.${slug}"
    // Body directive, so no withLabel selector in the config may set memory (a selector
    // beats a body directive).
    memory { memoryAsk(indexFirstAskGb(alphabet, ksize as int), task.attempt) }

    input:
    tuple val(name), val(window), path(fasta), val(alphabet), val(ksize), val(scaled),
          val(penalty), val(xdrop), val(slug)

    output:
    tuple val(name), val(window), val(alphabet), val(ksize), val(penalty), val(xdrop),
          val(slug), path("${name}.w${window}.${slug}.kmerseek.rocksdb"), emit: index
    path "${name}.w${window}.${slug}.index.log", emit: log

    script:
    def idx = "${name}.w${window}.${slug}.kmerseek.rocksdb"
    def kq  = name.startsWith('regions') ? params.ka_queries_regions : params.ka_queries
    """
    set -euo pipefail
    ${params.time_cmd} ${params.kmerseek} index --alphabet ${alphabet} --ksize ${ksize} --scaled ${scaled} \\
        --input ${fasta} --output ${idx} \\
        --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} \\
        --ka-queries ${kq} --ka-survival-out ${idx}/ka_survival.C${penalty}.csv \\
        --kmer-stats-out ${idx}/spectrum.csv.gz > ${name}.w${window}.${slug}.index.log 2>&1
    ${params.kmerseek} --version >> ${name}.w${window}.${slug}.index.log
    """
}

process kmerseekSearch {
    label 'kmerseek_search'
    tag "${name}.w${window}.${slug}.${chunk.simpleName}"

    memory { memoryAsk(mem_gb as double, task.attempt) }

    input:
    tuple val(name), val(window), val(alphabet), val(ksize), val(penalty), val(xdrop),
          val(slug), path(index_dir), val(mem_gb), val(index_label), val(fit_flags),
          path(chunk)

    output:
    tuple val("kmerseek.${index_label}.w${window}.${slug}"),
          path("${chunk.simpleName}.calls.tsv.gz"), emit: calls
    path "${chunk.simpleName}.search.log"

    script:
    // kmerseek writes regions 0-based and end-exclusive on both proteins; the shared call
    // format is 1-based and inclusive, so start + 1 and the end as written. Columns are
    // found by header name. Names hold no commas (queries are bare accessions; regions-index
    // descriptions had commas replaced), so splitting on commas is safe.
    """
    # Nextflow runs every task under bash -e, which would stop the script at the failed
    # pipe before the no-fit check below can read its status (seen 2026-10-07: protein20
    # k5 on the regions index, a refused fit, failed the run).
    set -uo pipefail
    set +e
    ${params.time_cmd} ${params.kmerseek} search --alphabet ${alphabet} --ksize ${ksize} \\
        --query ${chunk} --target ${index_dir} \\
        --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} --ka-queries 0 ${fit_flags} \\
        --chain-max-gap ${params.chain_max_gap} --chain-max-shift ${params.chain_max_shift} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} \\
        --min-region-score ${params.min_region_score} 2> ${chunk.simpleName}.search.log \\
      | awk -F, -v OFS='\\t' -v max=${params.max_region_evalue} '
          NR == 1 { for (i = 1; i <= NF; i++) c[\$i] = i
                    split("query_name target_name region_start region_end target_start target_end region_evalue region_subseq target_subseq", need, " ")
                    for (n in need) if (!(need[n] in c)) { print "missing column " need[n] > "/dev/stderr"; exit 3 }
                    print "query", "qstart", "qend", "target", "tstart", "tend", "evalue", "qseq", "tseq"; next }
          (\$c["region_evalue"] + 0) <= max {
                    print \$c["query_name"], \$c["region_start"] + 1, \$c["region_end"],
                          \$c["target_name"], \$c["target_start"] + 1, \$c["target_end"],
                          \$c["region_evalue"], \$c["region_subseq"], \$c["target_subseq"] }' \\
      | gzip -c > ${chunk.simpleName}.calls.tsv.gz
    status=(\${PIPESTATUS[@]})
    set -e
    if [ "\${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" ${chunk.simpleName}.search.log; then
        # The index-time fit was refused, so this setting has no E-value on this index.
        # Recorded as such, never as zero calls: the scorer reads the #nofit line.
        { printf '#nofit\\n'; printf 'query\\tqstart\\tqend\\ttarget\\ttstart\\ttend\\tevalue\\tqseq\\ttseq\\n'; } \\
            | gzip -c > ${chunk.simpleName}.calls.tsv.gz
        exit 0
    fi
    for s in "\${status[@]}"; do [ "\$s" -eq 0 ] || exit 1; done
    """
}

process mmseqsSearch {
    label 'mmseqs'
    tag "${name}.w${window}"

    input:
    tuple val(name), val(window), path(fasta), path(queries)

    output:
    tuple val("mmseqs2.${indexKind(name)}.w${window}.mmseqs2"), path("calls.tsv.gz"), emit: calls
    path "mmseqs.${name}.w${window}.log"

    script:
    """
    set -euo pipefail
    ${params.mmseqs} createdb ${queries} qdb > mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} createdb ${fasta} tdb >> mmseqs.${name}.w${window}.log 2>&1
    ${params.time_cmd} ${params.mmseqs} search qdb tdb res tmp -s ${params.mmseqs_sensitivity} \\
        --num-iterations ${params.mmseqs_iterations} -e ${params.mmseqs_evalue} \\
        --threads ${task.cpus} >> mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} convertalis qdb tdb res out.tsv \\
        --format-output 'query,qstart,qend,target,tstart,tend,evalue,qaln,taln' \\
        >> mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} version >> mmseqs.${name}.w${window}.log
    { printf 'query\\tqstart\\tqend\\ttarget\\ttstart\\ttend\\tevalue\\tqseq\\ttseq\\n'; cat out.tsv; } \\
        | gzip -c > calls.tsv.gz
    """
}

process compositionClassifier {
    label 'python'
    tag "w${window}"

    input:
    tuple val(name), val(window), path(fasta), path(queries)

    output:
    tuple val("composition.regions.w${window}.composition"), path("calls.tsv.gz"), emit: calls

    script:
    """
    ${py('composition_classifier.py')} --queries ${queries} --index ${fasta} \\
        --window-aa ${params.composition_window} --step ${params.composition_step} \\
        --out calls.tsv.gz
    """
}

process scoreCalls {
    label 'python'
    tag "${label}"
    publishDir "${params.outdir}/scored", mode: 'copy'

    input:
    tuple val(label), path(calls, stageAs: 'calls_??????.tsv.gz'), path(truth),
          path(searched), path(target_features)

    output:
    path "*.features.parquet", emit: features
    path "*.threshold.json", emit: threshold
    path "*.call_hits.parquet", emit: call_hits, optional: true

    script:
    def parts = label.tokenize('.')
    // Every index but the whole-protein one names its entries accession|type|desc|start-end.
    def kind  = parts[1] in ['whole', 'wholeannot'] ? parts[1] : 'regions'
    // Whole-protein calls are scored a second time keeping only calls whose target side
    // lies in an annotated feature (index label "wholeannot"; Olga, 2026-10-13). The
    // composition classifier is scored on all its calls, ranked, for the comparison at
    // kmerseek's call count.
    def annot = (parts + []).with { it[1] = 'wholeannot'; it.join('.') }
    def common = "--calls calls_*.tsv.gz --truth ${truth} --searched ${searched} " +
                 "--target-features ${target_features} --flank \$(( ${params.k_max} - 1 )) " +
                 "--max-decoy-rate ${params.max_decoy_rate} --min-landing ${params.min_landing}"
    def first  = parts[0] == 'composition' ? '--all-calls' : ''
    def second = parts[1] == 'whole'
        ? "${py('score_calls.py')} ${common} --index-kind whole --annotated-targets-only --label ${annot} --prefix ${annot}"
        : ''
    """
    set -euo pipefail
    ${py('score_calls.py')} ${common} --index-kind ${kind} --label ${label} --prefix ${label} \\
        ${first}
    ${second}
    """
}

// Each searched query feature's identity to its closest same-type regions-index entry, from
// a search outside the compared tools, for the identity split (bin/feature_identity.py).
process featureFasta {
    label 'python'

    input:
    path truth
    path searched

    output:
    path 'features.fasta'

    script:
    """
    ${py('feature_identity.py')} --truth ${truth} --searched ${searched} --write-fasta features.fasta
    """
}

process featureSearch {
    label 'mmseqs'

    input:
    path features
    path regions_fasta

    output:
    path 'feature_hits.tsv'

    script:
    """
    set -euo pipefail
    ${params.mmseqs} easy-search ${features} ${regions_fasta} feature_hits.tsv tmp \\
        -s 7.5 --max-seqs 1000 -e 10 --threads ${task.cpus} \\
        --format-output 'query,target,pident,alnlen,evalue' > feature_search.log 2>&1
    """
}

process compareIndexes {
    label 'python'
    publishDir "${params.outdir}/comparison", mode: 'copy'

    input:
    path scored
    path truth
    path searched
    path feature_hits

    output:
    path '*.tsv'
    path 'feature_calls.parquet'

    script:
    """
    ${py('feature_identity.py')} --truth ${truth} --searched ${searched} \\
        --read-hits ${feature_hits} --out feature_identity.tsv
    ${py('compare_indexes.py')} --scored ${scored} --truth ${truth} \\
        --feature-identity feature_identity.tsv
    """
}

workflow build {
    main:
    if (!params.outdir) error "--outdir is required"
    dat = file(params.swissprot_dat, checkIfExists: true)
    windows = params.decoy_windows.toString().tokenize(',')*.trim()*.toInteger()

    parseFeatures(dat)
    buildWholeReference(dat)
    buildQueryTruth(parseFeatures.out.features, parseFeatures.out.sequences,
                    file(params.chr6_queries, checkIfExists: true),
                    file(params.disprot, checkIfExists: true))
    extractRegions(parseFeatures.out.features, parseFeatures.out.sequences,
                   buildWholeReference.out.accessions)
    clusterRegions(extractRegions.out.long_fasta)
    finishRegionsIndex(extractRegions.out.regions, clusterRegions.out.tsv)

    indexes = buildWholeReference.out.fasta.map { ['whole_protein_index', it] }
        .mix(finishRegionsIndex.out.fasta.map { ['regions_index', it] })
    if (params.pfam_regions) {
        if (!params.pfam_hmm_dat) error "--pfam_regions needs --pfam_hmm_dat from the same Pfam release"
        extractPfamRegions(file(params.pfam_regions, checkIfExists: true),
                           file(params.pfam_hmm_dat, checkIfExists: true),
                           parseFeatures.out.sequences, buildWholeReference.out.accessions)
        clusterPfamRegions(extractPfamRegions.out.long_fasta)
        finishPfamIndex(extractPfamRegions.out.regions, clusterPfamRegions.out.tsv)
        indexes = indexes.mix(finishPfamIndex.out.fasta.map { ['pfam_index', it] })
    }
    makeDecoys(indexes.combine(Channel.fromList(windows)))
    // The size summary reads the headline window's files, whole-protein and regions index.
    pair = makeDecoys.out.combined.filter { it[1] == windows[0] && it[0] != 'pfam_index' }
        .map { n, w, f -> [n, f] }
        .toSortedList { a, b -> b[0] <=> a[0] }
        .map { l -> l.flatten() }
    indexSummary(buildQueryTruth.out.queries, pair)

    emit:
    combined  = makeDecoys.out.combined
    queries   = buildQueryTruth.out.queries
    truth     = buildQueryTruth.out.truth
    features  = parseFeatures.out.features
    regions   = finishRegionsIndex.out.fasta
}

workflow {
    build()
    splitQueries(build.out.queries)

    settings = Channel.fromList(kmerseekSettings())
    kmerseekIndex(build.out.combined.combine(settings))

    // Each index with its search memory, from its own k-mer spectrum. A setting whose
    // search would need more than the largest node is left out and named in the log.
    sized = kmerseekIndex.out.index
        .map { n, w, a, k, p, x, s, idx -> [n, w, a, k, p, x, s, idx, searchFirstAskGb(spectrumLoad(idx))] }
        .filter { row ->
            def ok = row[8] <= (params.kmerseek_skip_above_gb as double)
            if (!ok) log.warn "skipping ${row[0]}.w${row[1]}.${row[6]}: its search would need ${Math.round(row[8])} GB"
            ok
        }
    own = sized.map { row -> row + [indexKind(row[0]), ''] }
    // The regions index searched with the whole-protein index's fit for the same setting
    // and decoy window. No row when the whole-protein fit was refused.
    wholeFit = sized.filter { row -> row[0].startsWith('whole') }
        .map { row -> [[row[1], row[6]], storedFit(row[7], row[4])] }
        .filter { key, fit -> fit != null }
    shared = sized.filter { row -> !row[0].startsWith('whole') }
        .map { row -> [[row[1], row[6]], row] }
        .join(wholeFit)
        .map { key, row, fit -> row + [indexKind(row[0]) + 'shared', "--ka-k ${fit.k} --ka-r-database ${fit.slope}".toString()] }
    searches = params.shared_fit ? own.mix(shared) : own
    kmerseekSearch(searches.combine(splitQueries.out.chunks.flatten()))

    mmseqsSearch(build.out.combined.combine(splitQueries.out.searched))
    compositionClassifier(build.out.combined.filter { it[0] == 'regions_index' }
        .combine(splitQueries.out.searched))

    calls = kmerseekSearch.out.calls.groupTuple()
        .mix(mmseqsSearch.out.calls.map { l, f -> [l, [f]] })
        .mix(compositionClassifier.out.calls.map { l, f -> [l, [f]] })
    scoreCalls(calls.combine(build.out.truth).combine(splitQueries.out.searched)
                    .combine(build.out.features))
    featureFasta(build.out.truth, splitQueries.out.searched)
    featureSearch(featureFasta.out, build.out.regions)
    compareIndexes(scoreCalls.out.features.mix(scoreCalls.out.threshold)
                       .mix(scoreCalls.out.call_hits).collect(),
                   build.out.truth, splitQueries.out.searched, featureSearch.out)
}
