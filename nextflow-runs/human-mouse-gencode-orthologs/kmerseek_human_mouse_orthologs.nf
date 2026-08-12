#!/usr/bin/env nextflow

/*
 * Nextflow pipeline to evaluate kmerseek ortholog detection using human-mouse orthologs
 *
 * Runs all HP alphabet encodings × k=18-30 against mouse GENCODE proteins.
 * (2026-08-11: floor lowered from 20 to 18 -- this absorbs the sibling
 * ../human-mouse-gencode-orthologs-hp-v040/ pipeline's job for k18-19, which was only ever a
 * partial, hand-listed repair of specific truncated combos, not a real k15-19 sweep; that
 * pipeline had also never actually been run, only `-preview`. k15-17 deliberately NOT included
 * yet -- extrapolating from the k20-23 size trend (66G/35G/32G/19G per ksize across all 6
 * variants), full k15-19 coverage could need on the order of 1-1.7TB, well past the ~144GB free
 * at the time of this change. k18-19 alone is a more modest, examined step down.
 *
 * Usage:
 *   nextflow run kmerseek_human_mouse_orthologs.nf
 */

// FASTA files for human and mouse protein sequences
params.human_fasta = "${System.getProperty('user.home')}/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
params.mouse_fasta = "${System.getProperty('user.home')}/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa"
params.outdir = "${System.getProperty('user.home')}/data/gencode/results-human-mouse-orthologs"

// Ortholog mapping URL
params.ortholog_url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"

// OrthoFinder's own human-vs-mouse orthologue calls -- the sequence-similarity baseline
// computeRbhF1 re-scores per combo (notebook 200 §1's "lesson 3": OrthoFinder must be scored on
// the same gene universe as the kmerseek combo it's compared to, not its own full workload).
params.of_tsv = "${System.getProperty('user.home')}/data/gencode/data-for-orthofinder/OrthoFinder/Results_Mar03/Orthologues/Orthologues_gencode.v49.pc_translations/gencode.v49.pc_translations__v__gencode.vM38.pc_translations.tsv"

// Minimum containment threshold — 0.0 keeps all hits (large CSVs; polars handles them)
params.threshold = 0.0

// kmerseek main's search-time filters (added after 0.3.1) reject low-probability matches
// before they're ever written to disk, instead of writing everything and filtering in the
// analysis notebooks afterward. Defaults match kmerseek's own CLI defaults.
params.min_shared_kmers = 2
params.max_pvalue = 0.05

process downloadOrthologMapping {
    publishDir params.outdir, mode: 'copy'

    output:
    path 'HOM_MouseHumanSequence.rpt'

    script:
    """
    curl -o HOM_MouseHumanSequence.rpt "${params.ortholog_url}"
    """
}

process parseOrthologMapping {
    publishDir params.outdir, mode: 'copy'

    input:
    path ortholog_file

    output:
    path 'ortholog_pairs.tsv'
    path 'ortholog_stats.txt'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import csv
    from collections import defaultdict

    # Parse ortholog file and extract human-mouse gene symbol pairs
    # Group by DB Class Key to identify ortholog groups
    ortholog_groups = defaultdict(lambda: {'human': set(), 'mouse': set()})

    with open('${ortholog_file}', 'r') as f:
        reader = csv.DictReader(f, delimiter='\\t')
        for row in reader:
            db_class_key = row['DB Class Key']
            organism = row['Common Organism Name']
            symbol = row['Symbol']

            if organism == 'human':
                # Human gene symbols are uppercase in GENCODE
                ortholog_groups[db_class_key]['human'].add(symbol.upper())
            elif organism == 'mouse, laboratory':
                # Mouse gene symbols are capitalized (first letter uppercase) in GENCODE
                # But the JAX file already has proper capitalization
                ortholog_groups[db_class_key]['mouse'].add(symbol)

    # Create pairs: for each human gene, list all mouse orthologs
    human_to_mouse = defaultdict(set)
    mouse_to_human = defaultdict(set)

    for group in ortholog_groups.values():
        for human_gene in group['human']:
            for mouse_gene in group['mouse']:
                human_to_mouse[human_gene].add(mouse_gene)
                mouse_to_human[mouse_gene].add(human_gene)

    # Write pairs file (human_gene, mouse_gene)
    with open('ortholog_pairs.tsv', 'w') as f:
        f.write('human_gene\\tmouse_gene\\n')
        for human_gene, mouse_genes in sorted(human_to_mouse.items()):
            for mouse_gene in sorted(mouse_genes):
                f.write(f'{human_gene}\\t{mouse_gene}\\n')

    # Write stats
    with open('ortholog_stats.txt', 'w') as f:
        f.write(f'Number of ortholog groups: {len(ortholog_groups)}\\n')
        f.write(f'Number of human genes with mouse orthologs: {len(human_to_mouse)}\\n')
        f.write(f'Number of mouse genes with human orthologs: {len(mouse_to_human)}\\n')
        f.write(f'Total human-mouse pairs: {sum(len(v) for v in human_to_mouse.values())}\\n')

        # Check for one-to-many and many-to-many relationships
        one_to_one = sum(1 for v in human_to_mouse.values() if len(v) == 1)
        one_to_many = sum(1 for v in human_to_mouse.values() if len(v) > 1)
        f.write(f'Human genes with exactly one mouse ortholog: {one_to_one}\\n')
        f.write(f'Human genes with multiple mouse orthologs: {one_to_many}\\n')
    """
}

process indexDatabase {
    tag "${species}_${encoding}_k${ksize}"
    // Deliberately NOT kmerseek:main -- indexing doesn't need main's search-time filters, and
    // main also removed `--scaled` from `kmerseek index` entirely (crashes below), plus
    // swapping this container tag changes every combo's task hash, so -resume would try to
    // rebuild EVERY already-built index (not just the missing ones) and collide trying to
    // `mv` onto a non-empty already-populated storeDir target. Keeping this on 0.3.1 leaves
    // existing indices alone; only searchHumanVsMouse needs the new binary.
    container 'kmerseek:0.3.1'
    containerOptions '--entrypoint ""'
    storeDir "${params.outdir}/indices"

    input:
    tuple val(species), path(fasta), val(encoding), val(ksize)

    output:
    tuple val(species), val(encoding), val(ksize), path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir')

    script:
    // kmerseek normalises the encoding to underscores when auto-naming the database
    // (e.g. hp-thomas-dill -> hp_thomas_dill); the declared output must use that
    // normalised name.
    //
    // Deliberately ONE declared output (just the dir), not dir+log as separate outputs.
    // storeDir only skips re-running a task when ALL its declared outputs already exist
    // at the target; with two outputs, one going missing/stale while the other survives
    // (dash/underscore mismatch in 2026-07, a log that predated storeDir tracking it in
    // 2026-08) silently forces a rebuild, which then crashes trying to `mv` the rebuilt
    // dir onto the already-populated target ("Directory not empty"). With a single dir
    // output, storeDir's presence check is binary and this class of collision can't
    // happen: either the dir's there and the task is skipped outright (no mv, no rebuild),
    // or it's not and a normal mv into an empty target succeeds. The log is nested INSIDE
    // the dir (as kmerseek_index.log) instead of a sibling path so it travels with the
    // one output that matters, rather than being a second thing that can go missing on
    // its own.
    def enc_fname = encoding.replace('-', '_')
    def db_dir   = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.rocksdb"
    """
    LOG_TMP=\$(mktemp)
    echo "=== Indexing: ${species} ${encoding} k=${ksize} ===" | tee \$LOG_TMP
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a \$LOG_TMP
    echo "" | tee -a \$LOG_TMP

    kmerseek index \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --scaled 1 \\
        --input ${fasta} \\
        2>&1 | tee -a \$LOG_TMP || true

    # Ensure output directory exists even if k-mer space is saturated
    mkdir -p ${db_dir}

    echo "" | tee -a \$LOG_TMP
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a \$LOG_TMP

    # Filename can't collide with anything RocksDB itself writes (*.sst, CURRENT,
    # IDENTITY, LOCK, LOG, LOG.old.*, MANIFEST-*, OPTIONS-*).
    cp \$LOG_TMP ${db_dir}/kmerseek_index.log
    """
}

process searchHumanVsMouse {
    tag "${encoding}_k${ksize}"
    container 'kmerseek:main'
    containerOptions '--entrypoint ""'
    storeDir params.outdir
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(encoding), val(ksize), path(human_fasta), path(mouse_index)

    output:
    tuple val(encoding), val(ksize), path("human_vs_mouse.${encoding}.k${ksize}.results.csv.zst")
    path "human_vs_mouse.${encoding}.k${ksize}.search.log"

    script:
    def output_zst = "human_vs_mouse.${encoding}.k${ksize}.results.csv.zst"
    def log_file   = "human_vs_mouse.${encoding}.k${ksize}.search.log"
    """
    set -o pipefail

    echo "=== Searching: human vs mouse ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    # NOTE: no `|| true` and no fallback `touch` here anymore -- both used to swallow a
    # crashed/OOM-killed `kmerseek search` (e.g. SIGKILL, no stderr) and let the task exit 0
    # with a truncated or empty results file, which storeDir then cached as a permanent
    # "success". Some HP encodings (hp-kyte-doolittle, hp-thomas-dill, hp-thomas-dill-no-c) at
    # low ksize produce dense hub k-mers that exceed real available memory (Docker Desktop's
    # VM, not the `memory` directive below, is the actual ceiling) and get killed. Now that
    # failure propagates (`pipefail` + Nextflow's default `set -e`) so the task is marked
    # FAILED and produces no cached output -- see withName: searchHumanVsMouse's
    # errorStrategy 'ignore' in nextflow.config, which lets the rest of the sweep continue.
    #
    # No `--encoding` here (2026-08-10 fix): kmerseek 0.4.0's search added a strict check that
    # the flag must literally match the database's stored encoding, and errors otherwise --
    # "Encoding mismatch: database has encoding=hp, but you specified --encoding=HpLehninger."
    # hp-lehninger is byte-identical to plain hp (verified in notebook 200) and kmerseek's own
    # indexer normalizes it to "hp" internally at index time, so every hp-lehninger search
    # failed under the new check even though nothing was actually wrong. Dropping the flag lets
    # kmerseek auto-detect encoding from the database the same way it already does for
    # ksize/scaled -- exactly what the error message itself suggests, and can never mismatch by
    # construction since the index was built from this same channel's encoding value.
    kmerseek search \\
        --ksize ${ksize} \\
        --query ${human_fasta} \\
        --target ${mouse_index} \\
        --min-shared-kmers ${params.min_shared_kmers} \\
        --max-pvalue ${params.max_pvalue} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${output_zst}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Compressed size: \$(du -sh ${output_zst} | cut -f1)" | tee -a ${log_file}
    """
}

// Converts each raw human_vs_mouse.*.results.csv.zst to parquet (dropping the two unused
// md5 columns) and deletes the raw file from searchHumanVsMouse's storeDir -- kept as its own
// (non-containerized) process because kmerseek:main's container has no python/polars, so this
// can't be folded into searchHumanVsMouse's script the way the hp-v040 sibling pipeline does it.
//
// Tradeoff, stated explicitly: because searchHumanVsMouse's storeDir output no longer exists
// on disk after this runs, a future `-resume` that needs a NEW (not-yet-searched) combo is
// unaffected, but re-touching an ALREADY-converted combo would force kmerseek search to redo
// that (expensive) work rather than finding a cached hit -- same tradeoff as manually running
// `nextflow clean`, just automatic. Acceptable here since this pipeline is being superseded by
// the hp-v040 sibling for new sweeps (see that .nf's header comment); this one's raw csv.zst
// files were the ~228GB that prompted this change in the first place.
process convertResultsToParquet {
    tag "${encoding}_k${ksize}"
    storeDir params.outdir

    input:
    tuple val(encoding), val(ksize), path(results_csv_zst)

    output:
    tuple val(encoding), val(ksize), path("human_vs_mouse.${encoding}.k${ksize}.results.parquet")

    script:
    def parquet = "human_vs_mouse.${encoding}.k${ksize}.results.parquet"
    """
    # `pl.scan_csv().sink_parquet()` looks lazy but does NOT stream a .csv.zst source -- it
    # inflates the whole file into RAM before parsing (measured up to 97x expansion on one
    # combo). On 2026-08-10 two concurrent instances of this (hp-kyte-doolittle k20 + k21)
    # hit 79GB + 66GB RSS at once, leaving 129MB free out of 128GB physical RAM -- Jetsam
    # started killing processes, the machine rebooted, and a second attempt right after
    # panicked outright (watchdog timeout: no checkins from watchdogd in 91s, i.e. the OS
    # was so overloaded even its own watchdog daemon starved for CPU). An out-of-band
    # one-combo patch on an earlier crash (hp-thomas-dill-no-c k22, exit 137) left every
    # other combo still exposed to this, which is what let this happen twice more.
    # convert_results_to_parquet_streaming.py instead pipes zstdcat into pyarrow's
    # incremental CSV reader -- measured flat ~3.75GB RSS regardless of file size, verified
    # on the largest combo in this pipeline at 610M rows / 56GB peak RSS for a case with
    # actual composite-score materialization (60x lighter here, this process keeps every
    # column). See scripts/polars_no_stream_compressed_csv memory note for the original
    # incident this script was written for.
    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \\
        /Users/olga/code/2024-kmerseek-analysis/scripts/convert_results_to_parquet_streaming.py \\
        --results-dir . \\
        --zstd-level 9

    # The streaming script skips (rather than touches) sources <=1KB -- kmerseek's
    # well-formed empty-index stub -- so recreate this process's original contract: an
    # empty ${parquet} is the signal evaluate_orthologs.py checks for "no data".
    if [ ! -f "${parquet}" ]; then
        touch "${parquet}"
    fi

    # ${results_csv_zst} here is nextflow's symlink into THIS task's work dir -- the real file
    # searchHumanVsMouse's storeDir wrote lives directly in params.outdir under the same name.
    real_src="${params.outdir}/human_vs_mouse.${encoding}.k${ksize}.results.csv.zst"
    rm -f "\$real_src"
    """
}

process evaluateOrthologs {
    tag "${encoding}_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(encoding), val(ksize), path(results_zst), path(ortholog_pairs)

    output:
    tuple val(encoding), val(ksize), path("ortholog_evaluation.${encoding}.k${ksize}.tsv.zst")
    path "ortholog_evaluation.${encoding}.k${ksize}.summary.txt"
    path "ortholog_evaluation.${encoding}.k${ksize}.roc_data.tsv.zst"
    path "ortholog_evaluation.${encoding}.k${ksize}.mht.csv.zst"
    path "metrics_*.${encoding}.k${ksize}.png"

    script:
    """
    # Pre-filter: keep rows where n_intersecting_hashes > expected_shared_kmers * 100
    # (enrichment >= 100). This safely captures all rows with Poisson p < ~0.001
    # regardless of n_total, while shrinking the file ~1000x. n_total (full row count)
    # is tracked for MHT. ${results_zst} is a Parquet file (from convertResultsToParquet,
    # NOT the raw .csv.zst -- that gets deleted upstream), so this must go through polars
    # scan_parquet/sink_parquet, not zstdcat+awk: zstdcat on non-zstd input silently
    # passes it through byte-for-byte instead of erroring, which let a stale CSV-shaped
    # pre-filter run against raw Parquet bytes for months without ever failing loudly --
    # it just quietly filtered out 100% of real rows.
    if [ ! -s ${results_zst} ]; then
        ln -sf ${results_zst} filtered_results.parquet
        n_total=0
    else
        /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 << 'PYEOF'
import polars as pl
from pathlib import Path

lf = pl.scan_parquet("${results_zst}")
n_total = lf.select(pl.len()).collect().item()

filtered = lf.filter(
    (pl.col("n_intersecting_hashes") >= 2)
    & (
        (pl.col("expected_shared_kmers") == 0)
        | (pl.col("n_intersecting_hashes") / pl.col("expected_shared_kmers") >= 100)
    )
)
filtered.sink_parquet("filtered_results.parquet", compression="zstd")

Path("n_total.txt").write_text(str(n_total))
PYEOF
        n_total=\$(cat n_total.txt 2>/dev/null || echo 0)
    fi
    # Guard against an empty/missing n_total (e.g. a header-only results file with 0 data rows):
    # an unquoted empty \${n_total} would drop the argument and shift ${encoding} into its place.
    n_total=\${n_total:-0}
    evaluate_orthologs.py ${ksize} filtered_results.parquet ${ortholog_pairs} "\${n_total}" "${encoding}"
    """
}

process aggregateResults {
    publishDir params.outdir, mode: 'copy'

    input:
    path summaries

    output:
    path 'kmer_sweep_summary.tsv'
    path 'kmer_sweep_summary.json'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import glob, json, re

    results = []
    for f in sorted(glob.glob('*.summary.txt')):
        m = re.search(r'\\.([^.]+)\\.k(\\d+)\\.summary', f)
        if not m:
            continue
        encoding, ksize = m.group(1), int(m.group(2))
        with open(f) as fh:
            content = fh.read()
        jm = re.search(r'=== JSON SUMMARY ===\\n(\\{[\\s\\S]+\\})', content)
        if jm:
            try:
                d = json.loads(jm.group(1))
                d['encoding'] = encoding
                results.append(d)
                continue
            except json.JSONDecodeError:
                pass
        results.append({'encoding': encoding, 'ksize': ksize})

    results.sort(key=lambda x: (x.get('encoding', ''), x.get('ksize', 0)))

    MHT_METHODS = ['bonferroni', 'bh', 'by', 'two_stage_bh']

    # Build TSV header
    headers = ['encoding', 'ksize', 'total_hits', 'n_ortholog', 'n_non_ortholog']
    for method in MHT_METHODS:
        headers += [f'{method}_rejected', f'{method}_precision', f'{method}_recall']

    with open('kmer_sweep_summary.tsv', 'w') as f:
        f.write('\\t'.join(headers) + '\\n')
        for r in results:
            row = [
                r.get('encoding', ''),
                str(r.get('ksize', '')),
                str(r.get('total_hits', '')),
                str(r.get('n_ortholog', '')),
                str(r.get('n_non_ortholog', '')),
            ]
            mht = r.get('mht', {})
            for method in MHT_METHODS:
                s = mht.get(method, {})
                row += [
                    str(s.get('rejected', '')),
                    f"{s.get('precision', 0):.4f}",
                    f"{s.get('recall', 0):.4f}",
                ]
            f.write('\\t'.join(row) + '\\n')

    with open('kmer_sweep_summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    """
}

process multiQC {
    publishDir params.outdir, mode: 'copy'

    input:
    path sweep_json

    output:
    path "multiqc_report.html"
    path "multiqc_report_data/"

    script:
    """
    make_multiqc_input.py ${sweep_json} mqc_input/

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \\
        mqc_input/ \\
        --config mqc_input/multiqc_config.yaml \\
        --outdir . \\
        --filename multiqc_report.html \\
        --force \\
        --no-megaqc-upload
    """
}

// ---------------------------------------------------------------------------
// Notebook 200 section 2b's composite-metric AUC/AUPRC sweep, moved out of the notebook. It used
// to be a for-loop over every combo inside the notebook itself, checkpointed by rewriting a CSV
// after each combo -- resumable only in the sense that a restarted notebook could skip
// already-done combos, with no parallelism and no protection against losing an in-progress combo
// (2026-08-05: hp_lehninger/hp_lehninger_plus_c k18, 364M/429M rows, thrashed this machine's
// 128GB RAM for 5+ days near-zero-progress before an OOM-kill with nothing saved, since the
// checkpoint only flushed BETWEEN combos). One task per combo here instead: real parallelism
// across combos, and storeDir gives free per-combo resumability without a notebook needing to
// track a "done" set itself. Reads whatever genome-wide result files already exist on disk (this
// run's own outputs, the sibling hp-v040 pipeline's, or an earlier run's), the same way
// ou.load_all_alphabet_ksize_combos and ou.scan_available_columns already do for every notebook
// that calls them -- but see listMetricLeaderboardCombos's barrier-input comment below: it must
// not run until this run's OWN search/convert chain has actually finished, or it snapshots the
// filesystem before this run has produced anything.
// ---------------------------------------------------------------------------

process listMetricLeaderboardCombos {
    // Derives the combo list from whatever genome-wide result files actually exist under
    // params.outdir (+ the hp-v040 sibling pipeline's outdir) -- NOT from any notebook-written
    // CSV. Pipeline steps must never depend on notebook output; only the reverse. An earlier
    // version of this process read notebook 200's own summary CSVs, which meant that notebook
    // had to be run first -- see bin/list_metric_leaderboard_combos.py's docstring.
    input:
    val ready
    // ^ unused in the script -- this process has no REAL data dependency on the search/convert
    // chain (it rescans the filesystem), so without something forcing an edge in the DAG,
    // Nextflow schedules it immediately at t=0 (observed 2026-08-10: it completed 1 second into
    // a 16-hour run). That means it snapshots the filesystem from BEFORE this run's own
    // searchHumanVsMouse/convertResultsToParquet tasks had produced anything -- a same-run combo
    // (e.g. a previously-broken encoding just fixed in this .nf) would need a SECOND `-resume`
    // to actually show up in the leaderboard, even though everything else is storeDir-cached and
    // that second resume is nearly instant. `parquet_results.collect()` below is passed purely
    // as a barrier value to force this to wait for that chain to finish first.

    output:
    path 'combos.csv'

    script:
    """
    list_metric_leaderboard_combos.py --data-dir ${params.outdir} --output combos.csv
    """
}

process computeMetricLeaderboard {
    tag "${dash_encoding}_k${ksize}"
    storeDir "${params.outdir}/200_metric_leaderboard"

    input:
    tuple val(dash_encoding), val(display_encoding), val(ksize)

    output:
    path "200_metric_leaderboard.${dash_encoding}.k${ksize}.csv"

    script:
    """
    compute_metric_leaderboard_combo.py \\
        --dash-encoding ${dash_encoding} \\
        --display-encoding ${display_encoding} \\
        --ksize ${ksize} \\
        --data-dir ${params.outdir} \\
        --output 200_metric_leaderboard.${dash_encoding}.k${ksize}.csv
    """
}

// ---------------------------------------------------------------------------
// Notebook 200 §1's RBH-F1-vs-MGI / OrthoFinder-matched-scope sweep, moved out of the notebook
// the same way §2b was above. Also replaces notebook 206 §1's hand-rolled duplicate of this
// exact sweep for the 6 dash-named HP variants (only 2/6 ever finished there -- "run this cell
// yourself and watch it", never actually completed) -- one canonical, all-combo, per-combo-task
// table instead of two separate partial in-notebook sweeps.
// ---------------------------------------------------------------------------

process computeRbhF1 {
    tag "${dash_encoding}_k${ksize}"
    storeDir "${params.outdir}/200_rbh_f1"

    input:
    tuple val(dash_encoding), val(display_encoding), val(ksize)

    output:
    path "200_rbh_f1.${dash_encoding}.k${ksize}.csv"

    script:
    """
    compute_rbh_f1_combo.py \\
        --dash-encoding ${dash_encoding} \\
        --display-encoding ${display_encoding} \\
        --ksize ${ksize} \\
        --data-dir ${params.outdir} \\
        --of-tsv ${params.of_tsv} \\
        --output 200_rbh_f1.${dash_encoding}.k${ksize}.csv
    """
}

process aggregateRbhF1 {
    publishDir params.outdir, mode: 'copy'

    input:
    path rbh_f1_csvs

    output:
    path '200_rbh_f1_all_combos.csv'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import polars as pl
    from pathlib import Path

    frames = [pl.read_csv(f) for f in sorted(Path('.').glob('200_rbh_f1.*.csv'))]
    non_empty = [f for f in frames if f.height > 0]
    combined = pl.concat(non_empty, how='diagonal_relaxed') if non_empty else pl.DataFrame()
    combined.write_csv('200_rbh_f1_all_combos.csv')
    print(f'{len(non_empty)}/{len(frames)} combo files had rows; {combined.height} total rows')
    """
}

process aggregateMetricLeaderboard {
    publishDir params.outdir, mode: 'copy'

    input:
    path leaderboard_csvs

    output:
    path '200_metric_leaderboard_all_combos.csv'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import polars as pl
    from pathlib import Path

    frames = [pl.read_csv(f) for f in sorted(Path('.').glob('200_metric_leaderboard.*.csv'))]
    non_empty = [f for f in frames if f.height > 0]
    combined = pl.concat(non_empty, how='diagonal_relaxed') if non_empty else pl.DataFrame()
    combined.write_csv('200_metric_leaderboard_all_combos.csv')
    print(f'{len(non_empty)}/{len(frames)} combo files had rows; {combined.height} total rows')
    """
}

workflow {
    // Download and parse ortholog mapping
    ortholog_file = downloadOrthologMapping()
    (ortholog_pairs, ortholog_stats) = parseOrthologMapping(ortholog_file)

    // Encoding × ksize ranges — alphabet size determines useful ksize:
    //   hp variants  k=18-30  (2-letter; 'hp' storeDir results reused). Floor lowered from 20
    //                to 18 on 2026-08-11 -- see file header comment. k15-17 still excluded:
    //                projected ~1-1.7TB for full k15-19 coverage vs. limited free disk at the
    //                time; revisit once there's more headroom.
    //   dayhoff      k=10-20  (6-letter)
    //   protein      k=5-15   (20-letter)
    hp_ksizes      = Channel.of(18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    dayhoff_ksizes = Channel.of(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    protein_ksizes = Channel.of(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    hp_enc_ksize = Channel.of(
        'hp-lehninger',
        'hp-thomas-dill',
        'hp-kyte-doolittle',
        'hp-thomas-dill-no-c',
        'hp-lehninger-plus-c',
        'hp-pbotc-1st-ed'
    ).combine(hp_ksizes)

    dayhoff_enc_ksize = Channel.of('dayhoff').combine(dayhoff_ksizes)
    protein_enc_ksize = Channel.of('protein').combine(protein_ksizes)

    all_enc_ksize = hp_enc_ksize.mix(dayhoff_enc_ksize).mix(protein_enc_ksize)

    // FASTA files are already uncompressed - use directly
    human_decompressed = channel.of(tuple('human', file(params.human_fasta)))
    mouse_decompressed = channel.of(tuple('mouse', file(params.mouse_fasta)))

    // Index mouse database for each (encoding, ksize)
    mouse_index_params = mouse_decompressed
        .combine(all_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(species, fasta, encoding, ksize) }

    indexed = indexDatabase(mouse_index_params)
    index_only = indexed[0]
    // (species, encoding, ksize, index_path)

    // Get mouse indexes by (encoding, ksize)
    mouse_indexes = index_only.map { species, encoding, ksize, index -> tuple(encoding, ksize, index) }

    // Combine human FASTA with mouse indexes by (encoding, ksize)
    human_fasta_enc_ksize = human_decompressed
        .combine(all_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(encoding, ksize, fasta) }

    search_inputs = human_fasta_enc_ksize.join(mouse_indexes, by: [0, 1])
    // (encoding, ksize, human_fasta, mouse_index)

    // Search human against mouse
    search_outputs = searchHumanVsMouse(search_inputs)
    search_results = search_outputs[0]
    // (encoding, ksize, results_csv_zst)

    // Convert to parquet immediately and drop the raw csv.zst (see process comment) --
    // everything downstream reads the parquet.
    parquet_results = convertResultsToParquet(search_results)
    // (encoding, ksize, results_parquet)

    // Evaluate ortholog detection
    eval_inputs = parquet_results.combine(ortholog_pairs)
    // (encoding, ksize, results_parquet, ortholog_pairs)
    eval_outputs = evaluateOrthologs(eval_inputs)

    // Collect all summary files for aggregation
    summaries = eval_outputs[1].collect()
    agg_out = aggregateResults(summaries)

    // MultiQC: single-document summary of the whole encoding x ksize sweep
    multiQC(agg_out[1])

    eval_outputs[0].subscribe { encoding, ksize, eval_file ->
        println("Completed evaluation: ${encoding} k=${ksize} -> ${eval_file}")
    }

    // Notebook 200 §2b's sweep. parquet_results.collect() is a barrier, not a real input --
    // see listMetricLeaderboardCombos's comment: without it, this runs at t=0 and misses
    // everything the search/convert chain above produces during THIS invocation.
    combo_tuples = listMetricLeaderboardCombos(parquet_results.collect())
        .splitCsv(header: true)
        .map { row -> tuple(row.dash_encoding, row.display_encoding, row.ksize as Integer) }
    leaderboard_csvs = computeMetricLeaderboard(combo_tuples)
    aggregateMetricLeaderboard(leaderboard_csvs.collect())

    // Notebook 200 §1's sweep -- same combo list, independent per-combo tasks.
    rbh_f1_csvs = computeRbhF1(combo_tuples)
    aggregateRbhF1(rbh_f1_csvs.collect())
}
