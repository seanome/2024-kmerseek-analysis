#!/usr/bin/env python3
"""Why does STAR call 20-98% of read pairs "too short" in the botryllus-pec-young runs?

STAR keeps a pair only when the aligned bases cover at least 66% of the two reads' combined
length (--outFilterMatchNminOverLread 0.66). If only one read of a pair aligns, it covers about
half, and the pair is reported as "unmapped: too short". This script tells apart the causes by
aligning a subsample of each library four ways, with the same STAR index and container as the
rnaseq-paper-genome-length20 run:

  paired        both reads together, the run's settings (the baseline)
  mate1, mate2  each read alone, as single-end reads, with the same 66% rule per read
  relaxed       both reads together, the 66% rule off (--outFilterMatchNmin 30 instead), so a pair
                with only one aligned read is kept and shows up as a singleton in flagstat

From mate1 and mate2 it counts, per pair, whether read 1, read 2, both or neither aligned alone.
"Neither" pairs are written out for a later BLAST, because those reads do not come from this
genome as assembled.

Differences from the pipeline's STAR call: one pass (--twopassMode None) and no GTF given at
mapping time (the index was built with it). Both change all four modes alike.

Stdlib only: Sherlock's /usr/bin/python3 is 3.6. Run inside a SLURM job, not on a login node.
"""
import argparse
import gzip
import os
import subprocess
import sys

LAUNCH = "/scratch/users/olgabot/botryllus-pec-young/rnaseq-paper-genome-length20-launch"
IMAGE = (
    "/scratch/users/olgabot/apptainer-cache/community-cr-prod.seqera.io-docker-registry-v2-"
    "blobs-sha256-26-268b4c9c6cbf8fa6606c9b7fd4fafce18bf2c931d1a809a0ce51b105ec06c89d-data.img"
)
INDEX = LAUNCH + "/work/6e/0bd518ea6d8b6550968972f863b652/star"
TRIMMED = {
    "SRR32233964": LAUNCH + "/work/f2/80ba96170bdd7e30a5167b320b7302/SRR32233964_trimmed_{}_val_{}.fq.gz",
    "SRR32233984": LAUNCH + "/work/05/dd34547dd1d58b97f0c82fa0c30cc0/SRR32233984_trimmed_{}_val_{}.fq.gz",
}
# Same filters as the pipeline's call (nf-core/rnaseq 3.27.0 ALIGN_STAR), minus two-pass.
BASE_ARGS = [
    "--runRNGseed", "0", "--outFilterMultimapNmax", "20", "--alignSJDBoverhangMin", "1",
    "--outSAMtype", "BAM", "Unsorted", "--twopassMode", "None",
]


def container(cmd):
    binds = ["-B", "/scratch/users/olgabot"]
    return ["apptainer", "exec", "--no-home"] + binds + [IMAGE] + cmd


def subsample(sample, n_pairs, outdir):
    """First n_pairs pairs of the trimmed reads (Trim Galore keeps the two files in step)."""
    paths = []
    for mate in (1, 2):
        src = TRIMMED[sample].format(mate, mate)
        dst = os.path.join(outdir, "{}_{}.fq".format(sample, mate))
        with gzip.open(src, "rt") as fin, open(dst, "w") as fout:
            for i, line in enumerate(fin):
                if i >= 4 * n_pairs:
                    break
                fout.write(line)
        paths.append(dst)
    return paths


def run_star(mode, reads, prefix, threads):
    extra = []
    if mode == "relaxed":
        extra = ["--outFilterScoreMinOverLread", "0", "--outFilterMatchNminOverLread", "0",
                 "--outFilterMatchNmin", "30", "--outSAMunmapped", "Within"]
    if mode in ("mate1", "mate2"):
        extra = ["--outReadsUnmapped", "Fastx"]
    cmd = ["STAR", "--genomeDir", INDEX, "--readFilesIn"] + reads + [
        "--runThreadN", str(threads), "--outFileNamePrefix", prefix] + BASE_ARGS + extra
    subprocess.check_call(container(cmd))


def final_log(prefix):
    out = {}
    with open(prefix + "Log.final.out") as fh:
        for line in fh:
            if "|" in line:
                key, val = line.split("|", 1)
                out[key.strip()] = val.strip()
    return out


def read_ids_from_fastx(path):
    ids = set()
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                ids.add(line[1:].split()[0])
    return ids


def flagstat(bam, threads):
    text = subprocess.check_output(
        container(["samtools", "flagstat", "-@", str(threads), bam]),
        universal_newlines=True)
    out = {}
    for line in text.splitlines():
        n = int(line.split()[0])
        if "primary mapped" in line:
            out["primary mapped reads"] = n
        elif "with itself and mate mapped" in line:
            out["reads with mate also mapped"] = n
        elif "singletons" in line:
            out["singleton reads (mate not mapped)"] = n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pairs", type=int, default=500000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--samples", nargs="+", default=sorted(TRIMMED))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for sample in args.samples:
        sdir = os.path.join(args.outdir, sample)
        os.makedirs(sdir, exist_ok=True)
        r1, r2 = subsample(sample, args.pairs, sdir)
        reads = {"paired": [r1, r2], "relaxed": [r1, r2], "mate1": [r1], "mate2": [r2]}
        logs = {}
        for mode in ("paired", "mate1", "mate2", "relaxed"):
            prefix = os.path.join(sdir, mode + ".")
            run_star(mode, reads[mode], prefix, args.threads)
            logs[mode] = final_log(prefix)
            for key in ("Number of input reads", "Average input read length",
                        "Uniquely mapped reads %", "% of reads mapped to multiple loci",
                        "% of reads unmapped: too short", "% of reads unmapped: other",
                        "% of reads mapped to too many loci"):
                rows.append((sample, mode, key, logs[mode].get(key, "")))

        # Per pair: which reads align on their own?
        un1 = read_ids_from_fastx(os.path.join(sdir, "mate1.Unmapped.out.mate1"))
        un2 = read_ids_from_fastx(os.path.join(sdir, "mate2.Unmapped.out.mate1"))
        n = int(logs["mate1"]["Number of input reads"])
        neither = len(un1 & un2)
        only2 = len(un1 - un2)
        only1 = len(un2 - un1)
        both = n - neither - only1 - only2
        for key, val in (("pairs: both reads align alone", both),
                         ("pairs: only read 1 aligns alone", only1),
                         ("pairs: only read 2 aligns alone", only2),
                         ("pairs: neither read aligns alone", neither)):
            rows.append((sample, "mate1+mate2", key, "{} ({:.2f}%)".format(val, 100.0 * val / n)))

        fs = flagstat(os.path.join(sdir, "relaxed.Aligned.out.bam"), args.threads)
        for key, val in sorted(fs.items()):
            rows.append((sample, "relaxed", key, str(val)))

        # Keep the reads that align nowhere, for BLAST; drop the big files.
        with open(os.path.join(sdir, "neither_ids.txt"), "w") as fh:
            fh.write("\n".join(sorted(un1 & un2)) + "\n")
        for f in os.listdir(sdir):
            if f.endswith(".bam") or f.endswith(".fq"):
                os.remove(os.path.join(sdir, f))

    out = os.path.join(args.outdir, "star_too_short_test.tsv")
    with open(out, "w") as fh:
        fh.write("sample\tmode\tmeasure\tvalue\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    sys.stdout.write(open(out).read())


if __name__ == "__main__":
    main()
