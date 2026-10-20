// Notebook 357: clade-grouped SPLASH on reduced-alphabet protein k-mers (kill test).
//
// UniProt reference proteomes, one species per family, NCBI classes with at least 5 species
// as the samples. For each reduced-alphabet anchor (Dayhoff k=14, HP k=36) the count table
// is targets (the next 10 residues) x classes, with species counts in the cells. The OASIS
// test (refresh-bio/SPLASH) asks whether the target distribution depends on the class. The
// rule, the controls and the pass/fail numbers are notebooks/357_preregistration.md.

nextflow.enable.dsl = 2

params.outdir      = 'results'
params.base_url    = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes'
// Downloaded FASTA files are kept here across runs (storeDir), so a second run with the
// same species does not download them again.
params.fasta_cache = "${System.getenv('SCRATCH') ?: launchDir}/nb357-clade-splash-fasta"
params.settings    = 'dayhoff:14,hp:36'
// Gaps between anchor and target. Gap 5 and 10 only for Dayhoff, as the rule says.
params.gaps        = 'dayhoff:14=0,5,10;hp:36=0'
params.seed        = 357
params.alpha       = 0.05
// Chunks of species for download, anchors and targets: about this many canonical proteins
// each (README column #(1)), so a chunk of eukaryotes and a chunk of bacteria cost about the
// same.
params.proteins_per_chunk = 400000
params.pfam_hmm    = null       // Pfam-A.hmm, pressed (hmmpress) next to it
params.oasis_commit = '00ab197550303e713343b68ea97daf36deb76946'
params.oasis_sha256 = '8116c5c206051b254698448ed74611f64e6cc94fea11452622b544c818207125'
// Pilot: species per class and number of classes; 0 = the full set.
params.max_per_class = 0
params.max_classes   = 0

process getOasis {
    label 'python'
    publishDir "${params.outdir}/provenance", mode: 'copy'

    output:
    path 'OASIS.py'

    script:
    """
    python3 - <<'PY'
import hashlib, urllib.request
url = "https://raw.githubusercontent.com/refresh-bio/SPLASH/${params.oasis_commit}/OASIS/OASIS.py"
data = urllib.request.urlopen(url).read()
h = hashlib.sha256(data).hexdigest()
if h != "${params.oasis_sha256}":
    raise SystemExit(f"OASIS.py at ${params.oasis_commit} has sha256 {h}, expected ${params.oasis_sha256}")
open("OASIS.py", "wb").write(data)
PY
    """

    stub:
    """
    cp ${projectDir}/assets/OASIS.py OASIS.py
    """
}

process selectSpecies {
    label 'python'
    memory '16 GB'
    time '2h'
    publishDir "${params.outdir}/species", mode: 'copy'

    output:
    path 'species.tsv', emit: species
    path 'selection_summary.json', emit: summary
    path 'inputs/inputs.json', emit: inputs
    path 'inputs/reference_proteomes.tsv', emit: proteomes
    path 'inputs/README', emit: readme

    script:
    def pilot = params.max_per_class ? "--max-per-class ${params.max_per_class} --max-classes ${params.max_classes}" : ''
    """
    clade_splash.py inputs --base-url ${params.base_url} --outdir inputs
    clade_splash.py select --proteomes inputs/reference_proteomes.tsv --readme inputs/README \\
        --nodes inputs/nodes.dmp --names inputs/names.dmp --out species.tsv \\
        --summary selection_summary.json ${pilot}
    rm inputs/nodes.dmp inputs/names.dmp
    """

    stub:
    """
    mkdir -p inputs
    cp ${projectDir}/tests/data/species_stub.tsv species.tsv
    echo '{}' > selection_summary.json; echo '{}' > inputs/inputs.json
    touch inputs/reference_proteomes.tsv inputs/README
    """
}

process fetchChunk {
    tag "chunk ${chunk}"
    label 'python'
    storeDir "${params.fasta_cache}/${chunk_key}"
    time { "${2 * task.attempt}h" }

    input:
    tuple val(chunk), val(chunk_key), val(idx), path(species)

    output:
    tuple val(chunk), path('fasta'), path('fetch_stats.tsv')

    script:
    """
    clade_splash.py fetch --species ${species} --idx ${idx.join(' ')} --base-url ${params.base_url} \\
        --outdir fasta --stats fetch_stats.tsv
    """

    stub:
    """
    mkdir -p fasta; printf 'species_idx\\tupid\\tn_proteins\\tn_residues\\tmd5\\turl\\n' > fetch_stats.tsv
    """
}

// Unique anchors of each species in a chunk, split into 256 hash buckets.
process anchors {
    tag "chunk ${chunk} ${setting}"
    label 'python'
    // No measurement yet for this script. A chunk of ~400k proteins is ~160M residues; the
    // unique-anchor tables held before writing are ~12 bytes per residue (2 GB), and the
    // largest single proteome (~60M residues) needs ~8 arrays of 8 bytes per residue in
    // anchor_codes (4 GB). Asked at 16 GB.
    memory { "${16 * task.attempt} GB" }
    time { "${3 * task.attempt}h" }

    input:
    tuple val(chunk), path(fasta), val(setting)

    output:
    tuple val(setting), path("anchors_${chunk}"), emit: dirs
    path "anchors_${chunk}_${setting.replace(':', '_')}.json", emit: stats

    script:
    """
    clade_splash.py anchors --setting ${setting} --fastas ${fasta}/*.fasta.gz \\
        --outdir anchors_${chunk} --stats anchors_${chunk}_${setting.replace(':', '_')}.json
    """

    stub:
    """
    mkdir anchors_${chunk}; echo '{"setting": "${setting}", "letter_counts": [0], "examined": 0, "usable": 0}' > anchors_${chunk}_${setting.replace(':', '_')}.json
    """
}

// Anchors in at least 15 species, one hash bucket at a time.
process candidates {
    tag "${setting} g${group}"
    label 'python'
    // ~1e10 residues / 256 buckets ~ 4e7 (anchor, species) rows, 12 bytes each, plus the
    // group-by, one bucket at a time: asked at 12 GB.
    memory { "${12 * task.attempt} GB" }
    time { "${4 * task.attempt}h" }
    publishDir "${params.outdir}/candidates/${setting.replace(':', '_')}", mode: 'copy', pattern: '*.hist.parquet'

    input:
    tuple val(setting), val(group), path(dirs, stageAs: 'chunk???/*')

    output:
    tuple val(setting), path("cand_${group}.npy"), emit: cand
    path "g${group}.hist.parquet", emit: hist

    script:
    // 16 of the 256 buckets per task: the scheduler's submit rate, not the work, set the
    // pilot's time with one bucket per task.
    def buckets = (group * 16..<(group + 1) * 16).join(' ')
    """
    clade_splash.py candidates --bucket ${buckets} --dirs chunk*/anchors_* \\
        --out cand_${group}.npy --hist g${group}.hist.parquet
    """

    stub:
    """
    touch cand_${group}.npy g${group}.hist.parquet
    """
}

process targets {
    tag "chunk ${chunk} ${setting}"
    label 'python'
    memory { "${16 * task.attempt} GB" }
    time { "${3 * task.attempt}h" }

    input:
    tuple val(chunk), path(fasta), val(setting), val(gaps), path(cands, stageAs: 'cand/*')

    output:
    tuple val(setting), path("targets_${chunk}"), emit: dirs
    tuple val(setting), path("accessions_${chunk}_${setting.replace(':', '_')}.parquet"), emit: acc

    script:
    """
    clade_splash.py targets --setting ${setting} --gaps ${gaps.join(' ')} --candidates cand/*.npy \\
        --fastas ${fasta}/*.fasta.gz --outdir targets_${chunk} --accessions accessions_${chunk}_${setting.replace(':', '_')}.parquet
    """

    stub:
    """
    mkdir targets_${chunk}; touch accessions_${chunk}_${setting.replace(':', '_')}.parquet
    """
}

process test {
    tag "${setting} b${bucket}"
    label 'python'
    // OASIS_pvalue is pure Python per table; the hours scale with the number of anchors in
    // the bucket. No measurement yet: 12 h asked, doubled on retry.
    memory { "${16 * task.attempt} GB" }
    time { "${12 * task.attempt}h" }
    publishDir "${params.outdir}/tests/${setting.replace(':', '_')}", mode: 'copy'

    input:
    tuple val(setting), val(bucket), path(dirs, stageAs: 'chunk???/*'), path(species), path(oasis)

    output:
    path "test_${setting.replace(':', '_')}_b${bucket}.parquet"

    script:
    """
    clade_splash.py test --setting ${setting} --bucket ${bucket} --dirs chunk*/targets_* \\
        --species ${species} --oasis ${oasis} --seed ${params.seed} \\
        --out test_${setting.replace(':', '_')}_b${bucket}.parquet
    """

    stub:
    """
    touch test_${setting.replace(':', '_')}_b${bucket}.parquet
    """
}

process collect {
    label 'python'
    memory '32 GB'
    time '4h'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path tests, stageAs: 'tests/*'

    output:
    path 'all_anchors.parquet', emit: all
    path 'significant.parquet', emit: sig
    path 'counts.tsv', emit: counts

    script:
    """
    clade_splash.py collect --inputs tests/*.parquet --alpha ${params.alpha} \\
        --all-out all_anchors.parquet --sig-out significant.parquet --counts-out counts.tsv
    """

    stub:
    """
    touch all_anchors.parquet significant.parquet counts.tsv
    """
}

process occurrences {
    tag "${setting} b${bucket}"
    label 'python'
    memory { "${16 * task.attempt} GB" }
    time { "${2 * task.attempt}h" }

    input:
    tuple val(setting), val(bucket), path(dirs, stageAs: 'chunk???/*'), path(acc, stageAs: 'acc/*'), path(sig), path(fastas, stageAs: 'fa???/*')

    output:
    path "occ_${setting.replace(':', '_')}_b${bucket}.parquet", emit: occ
    path "prot_${setting.replace(':', '_')}_b${bucket}.fa", emit: fasta

    script:
    def tag = "${setting.replace(':', '_')}_b${bucket}"
    """
    clade_splash.py occurrences --setting ${setting} --bucket ${bucket} --dirs chunk*/targets_* \\
        --accessions acc/*.parquet --significant ${sig} --fasta-dirs fa*/fasta \\
        --out occ_${tag}.parquet --fasta-out prot_${tag}.fa
    """

    stub:
    def tag = "${setting.replace(':', '_')}_b${bucket}"
    """
    touch occ_${tag}.parquet prot_${tag}.fa
    """
}

// hmmscan of every protein that carries a significant anchor against Pfam-A, with Pfam's own
// gathering thresholds (--cut_ga).
process hmmscan {
    tag "part ${part}"
    // HMMER 3.4, biocontainers tag 3.4--h7d74f8d_5 (already in the Sherlock apptainer cache).
    container 'quay.io/biocontainers/hmmer:3.4--h7d74f8d_5'
    cpus 8
    memory '16 GB'
    time { "${8 * task.attempt}h" }

    input:
    tuple val(part), path(fasta), path(hmm), path(pressed)

    output:
    path "pfam_${part}.domtblout"

    script:
    """
    if [ -s ${fasta} ]; then
      hmmscan --cut_ga --cpu ${task.cpus} --noali -o /dev/null --domtblout pfam_${part}.domtblout ${hmm} ${fasta}
    else
      touch pfam_${part}.domtblout
    fi
    """

    stub:
    """
    touch pfam_${part}.domtblout
    """
}

process pfamOverlap {
    label 'python'
    memory '16 GB'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path occ, stageAs: 'occ/*'
    path dom, stageAs: 'dom/*'
    path sig

    output:
    path 'pfam_by_anchor.parquet', emit: anchors
    path 'pfam_by_occurrence.parquet', emit: occ
    path 'top_occurrences.parquet', emit: top

    script:
    """
    clade_splash.py pfam-overlap --occurrences occ/*.parquet --domtblout dom/* \\
        --occ-out pfam_by_occurrence.parquet --out pfam_by_anchor.parquet \\
        --significant ${sig} --top 50 --top-out top_occurrences.parquet
    """

    stub:
    """
    touch pfam_by_anchor.parquet pfam_by_occurrence.parquet top_occurrences.parquet
    """
}

process alignInput {
    label 'python'
    memory '16 GB'

    input:
    path pfam
    path occ, stageAs: 'occ/*'
    path dom, stageAs: 'dom/*'
    path fastas, stageAs: 'prot/*'

    output:
    path 'align_in', emit: dir
    path 'align_manifest.tsv', emit: manifest

    script:
    """
    clade_splash.py align-input --pfam ${pfam} --occurrences occ/*.parquet --domtblout dom/* \\
        --fastas prot/*.fa --outdir align_in --manifest align_manifest.tsv
    """

    stub:
    """
    mkdir align_in; touch align_manifest.tsv
    """
}

process hmmalign {
    container 'quay.io/biocontainers/hmmer:3.4--h7d74f8d_5'
    memory '8 GB'
    time '4h'

    input:
    path dir
    path manifest
    path hmm

    output:
    path 'sto'

    script:
    """
    mkdir sto
    hmmfetch --index ${hmm} > /dev/null
    tail -n +2 ${manifest} | while IFS=\$'\\t' read setting gap anchor fam tag; do
      [ -s ${dir}/\$tag.fa ] || continue
      acc=\$(grep -m1 "^ACC   \$fam\\." ${hmm} | awk '{print \$2}')
      hmmfetch ${hmm} "\$acc" > fam.hmm
      hmmalign --outformat Stockholm fam.hmm ${dir}/\$tag.fa > sto/\$tag.sto
    done
    """

    stub:
    """
    mkdir sto
    """
}

process alignControl {
    label 'python'
    memory '16 GB'
    time '8h'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path sto
    path manifest
    path sig
    path species
    path oasis

    output:
    path 'alignment_control.parquet'

    script:
    """
    clade_splash.py align-control --manifest ${manifest} --sto-dir ${sto} --significant ${sig} \\
        --species ${species} --oasis ${oasis} --seed ${params.seed} --out alignment_control.parquet
    """

    stub:
    """
    touch alignment_control.parquet
    """
}

process totals {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path fstats, stageAs: 'fetch???/*'
    path astats, stageAs: 'anchor/*'

    output:
    path 'residue_totals.json'
    path 'fetch_stats.tsv'

    script:
    """
    clade_splash.py totals --fetch-stats fetch*/fetch_stats.tsv --anchor-stats anchor/* \\
        --stats-out fetch_stats.tsv --out residue_totals.json
    """

    stub:
    """
    echo '{}' > residue_totals.json; touch fetch_stats.tsv
    """
}

workflow {
    if (!params.pfam_hmm) { error "--pfam_hmm is required (pressed Pfam-A.hmm)" }
    def settings = params.settings.tokenize(',')
    def gapsOf = params.gaps.tokenize(';').collectEntries { def kv = it.tokenize('='); [(kv[0]): kv[1].tokenize(',').collect { it as int }] }

    oasis = getOasis()
    sel = selectSpecies()

    // Chunks of species by README protein counts (#(1) column), in species_idx order.
    chunks = sel.species.combine(sel.readme).flatMap { sp, readme ->
        def nprot = [:]
        readme.eachLine { line ->
            def f = line.split('\t')
            if (f.size() >= 5 && f[0] ==~ /UP\d+/) { nprot[f[0]] = f[4] as long }
        }
        def rows = sp.splitCsv(header: true, sep: '\t')
        def out = [], cur = [], n = 0L
        rows.each { r ->
            cur << (r.species_idx as int)
            n += nprot.getOrDefault(r.upid, 0L)
            if (n >= params.proteins_per_chunk) { out << cur; cur = []; n = 0L }
        }
        if (cur) { out << cur }
        // The storeDir key is the species list itself, so a different selection never reuses
        // another selection's files.
        out.withIndex().collect { idx, i -> [i, "${idx.first()}-${idx.last()}-${idx.size()}", idx, sp] }
    }
    fetched = fetchChunk(chunks)

    an = anchors(fetched.map { c, fa, st -> [c, fa] }.combine(Channel.fromList(settings)))

    buckets = Channel.fromList((0..<16).toList())  // groups of 16 buckets
    candIn = an.dirs.groupTuple().combine(buckets).map { s, dirs, b -> [s, b, dirs] }
    cand = candidates(candIn)

    tg = targets(
        fetched.map { c, fa, st -> [c, fa] }
            .combine(cand.cand.groupTuple())
            .map { c, fa, s, cs -> [c, fa, s, gapsOf[s], cs] }
    )

    tbuckets = Channel.fromList((0..<64).toList())
    tests = test(tg.dirs.groupTuple().combine(tbuckets).map { s, dirs, b -> [s, b, dirs] }
                 .combine(sel.species).combine(oasis))
    col = collect(tests.collect())

    allFasta = fetched.map { c, fa, st -> fa }.collect().map { [it] }
    occ = occurrences(
        tg.dirs.groupTuple().join(tg.acc.groupTuple()).combine(tbuckets)
            .map { s, dirs, acc, b -> [s, b, dirs, acc] }
            .combine(col.sig)
            .combine(allFasta)
    )

    hmm = file(params.pfam_hmm)
    pressed = files("${params.pfam_hmm}.h3?")
    parts = occ.fasta.collectFile(name: 'proteins.fa').splitFasta(by: 2000, file: true)
        .toList().flatMap { fs -> fs.withIndex().collect { f, i -> [i, f, hmm, pressed] } }
        // No significant anchor means no proteins: one empty part keeps the rest of the DAG
        // running, so the notebook still gets (empty) Pfam tables.
        .ifEmpty { [0, file("${projectDir}/assets/empty.fa"), hmm, pressed] }
    dom = hmmscan(parts)

    po = pfamOverlap(occ.occ.collect(), dom.collect(), col.sig)
    ai = alignInput(po.anchors, occ.occ.collect(), dom.collect(), occ.fasta.collect())
    sto = hmmalign(ai.dir, ai.manifest, hmm)
    alignControl(sto, ai.manifest, col.sig, sel.species, oasis)

    totals(fetched.map { c, fa, st -> st }.collect(), an.stats.collect())
}
