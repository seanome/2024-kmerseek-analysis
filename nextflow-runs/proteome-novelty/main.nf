// Notebook 343: does one more proteome add new local sequence?
//
// For each clade of the QfO 2020_04 panel (assets/proteomes.tsv) and each alphabet x k
// setting, count the words that recur in at least two proteomes, the same for synthetic
// proteomes from a first-order Markov chain, and the same for MMseqs2 families. Then fit
// Heaps' law and score how well it predicts what the next hidden proteome adds. The rule
// is notebooks/343_preregistration.md.

nextflow.enable.dsl = 2

params.proteomes = "${projectDir}/assets/proteomes.tsv"
params.qfo_dir   = null
params.outdir    = 'results'
params.scaled    = 1
params.n_splits  = 200
// alphabet:k. Budgets matched in bits; see the pre-registered rule.
params.settings  = 'hp_pbotc_1st_ed:21,hp_pbotc_1st_ed:30,hp_pbotc_1st_ed:42,protein:5,protein:7,protein:10'
params.primary   = 'hp_pbotc_1st_ed:30'
params.mmseqs_args = '--min-seq-id 0.3 -c 0.8 --cov-mode 0'

// One row per proteome: [proteome, fasta path, n_sequences, n_residues, [clades]]
def readTable() {
    def rows = []
    file(params.proteomes).splitCsv(header: true, sep: '\t').each { r ->
        rows << [r.proteome, file("${params.qfo_dir}/${r.fasta}"), r.n_sequences as long,
                 r.n_residues as long, r.clades ? r.clades.tokenize(',') : []]
    }
    rows
}

// Each real FASTA must be the file the table was built from.
process checkFasta {
    tag "${proteome}"
    label 'python'

    input:
    tuple val(proteome), path(fasta), val(n_seq), val(n_res)

    output:
    tuple val(proteome), path(fasta)

    script:
    """
    python3 - <<'PY'
n_seq = n_res = 0
with open("${fasta}") as fh:
    for line in fh:
        if line.startswith(">"):
            n_seq += 1
        else:
            n_res += len(line.strip())
if (n_seq, n_res) != (${n_seq}, ${n_res}):
    raise SystemExit(f"${fasta}: {n_seq} sequences / {n_res} residues, table says ${n_seq} / ${n_res}")
PY
    """

    stub:
    """
    true
    """
}

process nullProteome {
    tag "${proteome}"
    label 'python'

    input:
    tuple val(proteome), path(fasta), val(seed)

    output:
    tuple val(proteome), path("null_${proteome}.fa.gz")

    script:
    """
    novelty.py null --fasta ${fasta} --seed ${seed} --out null_${proteome}.fa.gz
    """

    stub:
    """
    echo | gzip > null_${proteome}.fa.gz
    """
}

// Words of one clade under one setting -> (mask, n_words). `fastas` are staged by name and
// passed in clade order, so bit i of a mask is proteome i of `members`.
process cladeMasks {
    tag "${clade} ${kind} ${alphabet} k${ksize}"
    label 'python'
    // Measured 2026-10-18 on the laptop: Fungi (50.1M residues), protein k10, 2.9 GB peak
    // and 19 s, about 57 bytes per residue. Asked at 80 bytes per residue plus 2 GB.
    memory { "${(long) Math.ceil(2 + 80e-9 * n_res) * task.attempt} GB" }
    time   { "${4 * task.attempt}h" }

    input:
    tuple val(clade), val(kind), val(alphabet), val(ksize), val(members), val(names), path(fastas, stageAs: 'in/*'), val(n_res)

    output:
    tuple val(clade), val("${kind}.${alphabet}.${ksize}"), val(members), path("${clade}.${kind}.${alphabet}.${ksize}.masks.parquet"), emit: masks
    tuple val(clade), path("${clade}.${kind}.${alphabet}.${ksize}.pairs.parquet"), optional: true, emit: pairs
    path "${clade}.${kind}.${alphabet}.${ksize}.summary.json", emit: summary

    script:
    def prefix = "${clade}.${kind}.${alphabet}.${ksize}"
    def pairs = (kind == 'real' && "${alphabet}:${ksize}" == params.primary) ? "--pairs-out ${prefix}.pairs.parquet" : ''
    """
    novelty.py masks --fastas ${names.collect { "in/${it}" }.join(' ')} \\
        --alphabet ${alphabet} --ksize ${ksize} --scaled ${params.scaled} \\
        --out ${prefix}.masks.parquet ${pairs} --summary ${prefix}.summary.json
    """

    stub:
    def prefix = "${clade}.${kind}.${alphabet}.${ksize}"
    """
    touch ${prefix}.masks.parquet ${prefix}.summary.json
    ${kind == 'real' && "${alphabet}:${ksize}" == params.primary ? "touch ${prefix}.pairs.parquet" : ''}
    """
}

process mmseqsCluster {
    tag "${clade}"
    container 'quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0'
    // No history for this pipeline: a model, not a measurement. Eukaryota (408M residues,
    // 884_238 proteins) gets 58 GB and 25 h.
    cpus   16
    memory { "${Math.max(16L, (long) Math.ceil(8 + 120e-9 * n_res)) * task.attempt} GB" }
    time   { "${(long) Math.ceil(4 + 0.05e-6 * n_res) * task.attempt}h" }

    input:
    tuple val(clade), val(members), val(names), path(fastas, stageAs: 'in/*'), val(n_res)

    output:
    tuple val(clade), val(members), path("${clade}_cluster.tsv")

    script:
    def pairs = [members, names].transpose().collect { m, n -> "${m} in/${n}" }.join(' ')
    """
    relabel_for_mmseqs.sh ${pairs} > clade.fasta
    mmseqs easy-cluster clade.fasta ${clade} tmp ${params.mmseqs_args} --threads ${task.cpus}
    """

    stub:
    """
    touch ${clade}_cluster.tsv
    """
}

process mmseqsMasks {
    tag "${clade}"
    label 'python'

    input:
    tuple val(clade), val(members), path(tsv)

    output:
    tuple val(clade), val('real.mmseqs.family'), val(members), path("${clade}.real.mmseqs.family.masks.parquet")

    script:
    """
    novelty.py mmseqs --clusters ${tsv} --proteomes ${members.join(' ')} --out ${clade}.real.mmseqs.family.masks.parquet
    """

    stub:
    """
    touch ${clade}.real.mmseqs.family.masks.parquet
    """
}

process rarefy {
    tag "${clade} ${setting}"
    label 'python'
    // Each finished clade x setting is readable before the slowest MMseqs2 job ends.
    publishDir "${params.outdir}/parts", mode: 'copy'

    input:
    tuple val(clade), val(setting), val(members), path(masks)

    output:
    path "${clade}.${setting}.curve.parquet", emit: curve
    tuple val(clade), val(setting), path("${clade}.${setting}.last.parquet"), emit: last
    path "${clade}.${setting}.splits.parquet", emit: splits
    path "${clade}.${setting}.fit.json", emit: fit

    script:
    def p = "${clade}.${setting}"
    """
    novelty.py rarefy --masks ${masks} --clade ${clade} --setting ${setting} \\
        --proteomes ${members.join(' ')} --n-splits ${params.n_splits} \\
        --curve-out ${p}.curve.parquet --last-out ${p}.last.parquet \\
        --splits-out ${p}.splits.parquet --fit-out ${p}.fit.json
    """

    stub:
    def p = "${clade}.${setting}"
    """
    touch ${p}.curve.parquet ${p}.last.parquet ${p}.splits.parquet ${p}.fit.json
    """
}

// The specimen that adds the most newly recurring words when added last, and its longest
// stretches of them, at the primary setting.
process examples {
    tag "${clade}"
    label 'python'
    publishDir "${params.outdir}/parts", mode: 'copy'
    memory { "${(long) Math.ceil(4 + 40e-9 * n_res) * task.attempt} GB" }
    time   { "${2 * task.attempt}h" }

    input:
    tuple val(clade), path(last), path(pairs), val(members), val(names), path(fastas, stageAs: 'in/*'), val(n_res)

    output:
    path "${clade}.examples.parquet"

    script:
    def (alphabet, ksize) = params.primary.tokenize(':')
    """
    novelty.py examples --fastas ${names.collect { "in/${it}" }.join(' ')} \\
        --proteomes ${members.join(' ')} --clade ${clade} --specimen auto --last ${last} \\
        --pairs ${pairs} --alphabet ${alphabet} --ksize ${ksize} --top 3 --out ${clade}.examples.parquet
    """

    stub:
    """
    touch ${clade}.examples.parquet
    """
}

process collect {
    label 'python'
    publishDir params.outdir, mode: 'copy'

    input:
    path curves, stageAs: 'curve/*'
    path lasts, stageAs: 'last/*'
    path splits, stageAs: 'splits/*'
    path fits, stageAs: 'fit/*'
    path summaries, stageAs: 'summary/*'
    path examples, stageAs: 'examples/*'

    output:
    path "343_*"

    script:
    """
    python3 - <<'PY'
import glob, json
import polars as pl
for name in ["curve", "last", "splits", "examples"]:
    pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{name}/*.parquet"))], how="diagonal_relaxed").write_parquet(f"343_{name}.parquet")
pl.DataFrame([json.load(open(f)) for f in sorted(glob.glob("fit/*.json"))]).write_parquet("343_fit.parquet")
rows = []
for f in sorted(glob.glob("summary/*.json")):
    d = json.load(open(f))
    clade, kind, alphabet, k = f.split("/")[1].removesuffix(".summary.json").split(".")
    rows.append({"clade": clade, "kind": kind, "alphabet": alphabet, "ksize": int(k),
                 "n_windows": sum(d["n_windows"]), "n_distinct_words": d["n_distinct_words"], "n_masks": d["n_masks"]})
pl.DataFrame(rows).write_parquet("343_word_summary.parquet")
PY
    """

    stub:
    """
    touch 343_curve.parquet 343_last.parquet 343_splits.parquet 343_examples.parquet 343_fit.parquet 343_word_summary.parquet
    """
}

workflow {
    if (!params.qfo_dir) { error "--qfo_dir is required (the QfO 2020_04 release folder)" }

    def table = readTable()
    def clades = table.collectMany { it[4] }.unique()
    def settings = params.settings.tokenize(',').collect { def s = it.tokenize(':'); [s[0], s[1] as int] }

    checked = checkFasta(Channel.fromList(table.collect { [it[0], it[1], it[2], it[3]] }))
    seeds = table.withIndex().collectEntries { r, i -> [(r[0]): 343000 + i] }
    nulls = nullProteome(checked.map { p, f -> [p, f, seeds[p]] })

    // Clade inputs: members in table order, with the staged file names in the same order.
    def residues = table.collectEntries { [(it[0]): it[3]] }
    def members = clades.collectEntries { c -> [(c): table.findAll { c in it[4] }.collect { it[0] }] }
    def bundle = { ch ->
        ch.toList().flatMap { pairs ->
            def byName = pairs.collectEntries { [(it[0]): it[1]] }
            clades.collect { c ->
                def m = members[c]
                def fs = m.collect { byName[it] }
                [c, m, fs.collect { it.name }, fs, m.sum { residues[it] }]
            }
        }
    }
    realClades = bundle(checked)
    nullClades = bundle(nulls)

    maskJobs = realClades.map { [it, 'real'] }.mix(nullClades.map { [it, 'null'] })
        .combine(Channel.fromList(settings))
        .map { b, kind, alphabet, k -> [b[0], kind, alphabet, k, b[1], b[2], b[3], b[4]] }
    cm = cladeMasks(maskJobs)

    fam = mmseqsMasks(mmseqsCluster(realClades))
    rr = rarefy(cm.masks.mix(fam))

    primary = "real.${params.primary.replace(':', '.')}"
    ex = examples(
        rr.last.filter { it[1] == primary }.map { c, s, f -> [c, f] }
            .join(cm.pairs)
            .join(realClades.map { [it[0], it[1], it[2], it[3], it[4]] })
    )

    collect(rr.curve.collect(), rr.last.map { it[2] }.collect(), rr.splits.collect(),
            rr.fit.collect(), cm.summary.collect(), ex.collect())
}
