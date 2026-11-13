#!/usr/bin/env python3
"""Does STAR's default 66% length rule or a 30-base minimum give counts closer to the paper's?

For one library (one SLURM array task): take the first 500,000 trimmed read pairs of the
rnaseq-paper-genome-length20 run, align them with STAR twice on the paper's genome, and count
read pairs per gene with featureCounts the way count_split.nf does (pairs, strand ignored,
exons, gene_id, the paper's filtered GTF):

  default   STAR's own rule, the run's setting: the aligned part must cover 66% of the pair
  relaxed   --outFilterMatchNminOverLread 0 --outFilterScoreMinOverLread 0 --outFilterMatchNmin 30:
            any alignment with 30 matching bases is kept

Writes <outdir>/<library>.tsv with the pairs counted for a gene in each mode. The full-library
estimate for "relaxed" is the full run's counted pairs times relaxed/default on the subsample.

Both modes use one STAR pass (the run used two) and the same reads, so their ratio is fair.
Stdlib only: Sherlock's /usr/bin/python3 is 3.6. Run inside a SLURM job, not on a login node.
"""
import argparse
import gzip
import os
import subprocess

BASE = "/scratch/users/olgabot/botryllus-pec-young"
LAUNCH = BASE + "/rnaseq-paper-genome-length20-launch"
IMAGE = (
    "/scratch/users/olgabot/apptainer-cache/community-cr-prod.seqera.io-docker-registry-v2-"
    "blobs-sha256-26-268b4c9c6cbf8fa6606c9b7fd4fafce18bf2c931d1a809a0ce51b105ec06c89d-data.img"
)
INDEX = LAUNCH + "/work/6e/0bd518ea6d8b6550968972f863b652/star"
GTF = BASE + "/rnaseq-paper-genome-length20/genome/nc_nsi_nf.filtered.gtf"
FEATURECOUNTS = "/home/groups/ayeletv/subread-2.0.6-Linux-x86_64/bin/featureCounts"
# Written from the run's trace: library, read 1 path, read 2 path.
READS = BASE + "/star-too-short-test/trimmed_reads.tsv"
BASE_ARGS = [
    "--runRNGseed", "0", "--outFilterMultimapNmax", "20", "--alignSJDBoverhangMin", "1",
    "--outSAMtype", "BAM", "Unsorted", "--twopassMode", "None",
    "--outSAMattributes", "NH", "HI", "AS", "NM", "MD",
]
RELAXED = ["--outFilterScoreMinOverLread", "0", "--outFilterMatchNminOverLread", "0",
           "--outFilterMatchNmin", "30"]


def subsample(src, dst, n_pairs):
    with gzip.open(src, "rt") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= 4 * n_pairs:
                break
            fout.write(line)


def counted_pairs(summary):
    """Assigned pairs and all pairs featureCounts saw, from its .summary file."""
    with open(summary) as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh][1:]
    values = {r[0]: int(r[1]) for r in rows}
    return values["Assigned"], sum(values.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--pairs", type=int, default=500000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()

    with open(READS) as fh:
        library, r1, r2 = sorted(line.rstrip("\n").split("\t") for line in fh)[args.index]
    work = os.path.join(args.outdir, library)
    os.makedirs(work, exist_ok=True)
    fq = [os.path.join(work, "r1.fq"), os.path.join(work, "r2.fq")]
    subsample(r1, fq[0], args.pairs)
    subsample(r2, fq[1], args.pairs)

    out = []
    for mode, extra in (("default", []), ("relaxed", RELAXED)):
        prefix = os.path.join(work, mode + ".")
        subprocess.check_call(
            ["apptainer", "exec", "--no-home", "-B", "/scratch/users/olgabot", IMAGE, "STAR",
             "--genomeDir", INDEX, "--readFilesIn"] + fq
            + ["--runThreadN", str(args.threads), "--outFileNamePrefix", prefix] + BASE_ARGS + extra)
        subprocess.check_call(
            [FEATURECOUNTS, "-T", str(args.threads), "-p", "--countReadPairs", "-s", "0",
             "-t", "exon", "-g", "gene_id", "-a", GTF, "-o", prefix + "counts.tsv",
             prefix + "Aligned.out.bam"])
        assigned, seen = counted_pairs(prefix + "counts.tsv.summary")
        out.append((mode, assigned, seen))
        os.remove(prefix + "Aligned.out.bam")
    for f in fq:
        os.remove(f)

    with open(os.path.join(args.outdir, library + ".tsv"), "w") as fh:
        fh.write("library\tmode\tpairs_counted_for_a_gene\tpairs_in_featurecounts\n")
        for mode, assigned, seen in out:
            fh.write("{}\t{}\t{}\t{}\n".format(library, mode, assigned, seen))


if __name__ == "__main__":
    main()
