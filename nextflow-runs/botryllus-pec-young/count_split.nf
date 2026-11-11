// Which part of the counts causes the gap with the paper's gene lists?
// Counts read pairs per gene the way HTSeq does (featureCounts: a pair counts only if it
// overlaps one gene and aligns to one place) on the STAR alignments of nf-core/rnaseq, for
// each genome that has been aligned to, then fits every count table the same way
// (each-alone-checks/count_split.R).
//
//   this_genome   $out/rnaseq               April 2024 genome, FUN genes (make run)
//   paper_genome  $out/rnaseq-paper-genome  the paper's genome and g genes (make run-count-split)
//
// Run by run_count_split.sh inside the Nextflow head job.

params.out     = null   // output root, $SCRATCH/botryllus-pec-young
params.repo    = null   // nextflow-runs/botryllus-pec-young in the clone
params.genomes = 'this_genome,paper_genome'
// The lab's own featureCounts build on Sherlock.
params.featurecounts = '/home/groups/ayeletv/subread-2.0.6-Linux-x86_64/bin/featureCounts'
// The DESeq2 1.42.0 image nf-core/differentialabundance 2.0.0 pulled for this run.
params.deseq2_image  = '/scratch/users/olgabot/apptainer-cache/quay.io-biocontainers-bioconductor-deseq2@sha256-2d299c5f045c65879e8da189bba17857963d50703af5410003e7e84883deef13.img'

process FEATURECOUNTS {
    tag "${genome} strand ${strand}"
    // No earlier featureCounts run on these files to size from. The 38 alignment files of
    // the April 2024 genome total 15 GB; the largest is 3.6 GB (69.9 million read pairs).
    cpus 8
    memory { 16.GB * task.attempt }
    time { 2.h * task.attempt }

    input:
    tuple val(genome), path(gtf), path(bams), val(strand)

    output:
    path "${genome}__featurecounts_s${strand}.tsv", emit: counts
    path "${genome}__featurecounts_s${strand}.tsv.summary", emit: summary

    script:
    """
    ${params.featurecounts} -T ${task.cpus} -p --countReadPairs -s ${strand} \\
        -t exon -g gene_id -a ${gtf} -o ${genome}__featurecounts_s${strand}.tsv ${bams}
    """
}

process FITS {
    // 36 fits took 12 min 48 s and 2.1 GB in job 51911916; this is at most 36 fits.
    cpus 1
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    container params.deseq2_image
    publishDir "${params.out}/count-split", mode: 'copy'

    input:
    path tables
    path inputs
    path script

    output:
    path "count_split_*.tsv"
    path "count_split.out"

    script:
    """
    Rscript ${script} > count_split.out 2>&1
    """
}

workflow {
    def dirs = [this_genome: "${params.out}/rnaseq", paper_genome: "${params.out}/rnaseq-paper-genome"]
    def genomes = params.genomes.tokenize(',')
    // Only the paper's 38 libraries.
    def wanted = file("${params.repo}/samples_paper.csv").splitCsv(header: true).collect { it.sample } as Set

    by_genome = Channel.fromList(genomes).map { genome ->
        def gtf = file("${dirs[genome]}/genome/*.filtered.gtf")
        def bams = file("${dirs[genome]}/star_salmon/*.bam").findAll { it.name.tokenize('.')[0] in wanted }
        assert gtf.size() == 1 : "expected one filtered GTF in ${dirs[genome]}/genome, found ${gtf.size()}"
        assert bams.size() == wanted.size() : "expected ${wanted.size()} alignment files in ${dirs[genome]}/star_salmon, found ${bams.size()}"
        tuple(genome, gtf[0], bams)
    }
    FEATURECOUNTS(by_genome.combine(Channel.of(0, 1, 2)))

    // Salmon's tables from nf-core/rnaseq, renamed <genome>__salmon.tsv.
    salmon = Channel.fromList(genomes).flatMap { genome ->
        file("${workDir}/salmon-tables").mkdirs()
        [['gene_counts', ''], ['gene_lengths', '_lengths']].collect { which, suffix ->
            def f = file("${dirs[genome]}/star_salmon/salmon.merged.${which}.tsv", checkIfExists: true)
            f.copyTo("${workDir}/salmon-tables/${genome}__salmon${suffix}.tsv")
            file("${workDir}/salmon-tables/${genome}__salmon${suffix}.tsv")
        }
    }
    // The paper's own counts and groups (export_paper_counts.py), the libraries and the
    // paper's listed genes with their matched gene in this run.
    inputs = Channel.fromList([
        "${params.out}/size-factors/paper_counts.tsv",
        "${params.out}/size-factors/paper_groups.tsv",
        "${params.repo}/samples_paper.csv",
        "${params.repo}/each-alone-comparison/paper/paper_gene_overlap_genes.tsv",
    ]).map { file(it, checkIfExists: true) }

    FITS(
        FEATURECOUNTS.out.counts.mix(salmon).collect(),
        inputs.collect(),
        file("${params.repo}/each-alone-checks/count_split.R", checkIfExists: true),
    )
}
