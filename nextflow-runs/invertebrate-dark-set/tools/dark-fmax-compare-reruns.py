import gzip, glob, sys

O, H = sys.argv[1], "/scratch/users/olgabot/invertebrate-dark-set/results"
for sp in ("borrelia", "yeast", "human", "worm"):
    ids = set(l.strip() for l in open("%s/%s/all_ids.txt" % (O, sp)))
    for t in ("phmmer", "jackhmmer", "mmseqs2"):
        new = set()
        with gzip.open("%s/tools/%s.%s.tsv.gz" % (O, sp, t), "rt") as fh:
            for l in fh:
                p = l.rstrip("\n").split("\t")
                new.add((p[0], p[1], p[4], p[5], p[7]))
        old = set()
        for f in glob.glob("%s/%s/hits/%s.chunk_*.%s.tsv.gz" % (H, sp, sp, t)):
            with gzip.open(f, "rt") as fh:
                for l in fh:
                    p = l.rstrip("\n").split("\t")
                    if p[0] in ids:
                        old.add((p[0], p[1], p[2], p[3], p[5]))
        print(
            sp,
            t,
            "original",
            len(old),
            "rerun",
            len(new),
            "shared",
            len(old & new),
            "only original",
            len(old - new),
            "only rerun",
            len(new - old),
            flush=True,
        )
